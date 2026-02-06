import logging
import re
from pathlib import Path
from typing import List
from datetime import datetime

from .models import Metadata, Section, ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser:
    def parse(self, pdf_path: Path, document_id: str) -> ParsedDocument:
        import fitz
        
        doc = fitz.open(pdf_path)
        
        try:
            metadata = self._extract_metadata(doc)
            pages_text = [doc[i].get_text("text") for i in range(doc.page_count)]
            full_text = "\n\n".join(pages_text)
            sections = self._identify_sections(pages_text)
            
            return ParsedDocument(
                document_id=document_id,
                metadata=metadata,
                sections=sections,
                full_text=full_text,
                page_count=doc.page_count,
                parsing_date=datetime.now().isoformat()
            )
        finally:
            doc.close()
    
    def _extract_metadata(self, doc) -> Metadata:
        metadata = Metadata()
        
        pdf_meta = doc.metadata
        if pdf_meta:
            metadata.title = pdf_meta.get('title', 'Unknown') or 'Unknown'
            author = pdf_meta.get('author')
            metadata.authors = [author] if author else ['Unknown']
        
        if doc.page_count > 0:
            text = doc[0].get_text("text")
            
            if metadata.title == 'Unknown':
                for line in text.split('\n')[:10]:
                    if len(line.strip()) > 10:
                        metadata.title = line.strip()
                        break
            
            match = re.search(r'Abstract\s+(.*?)(?=\n\n|\nIntroduction)', 
                            text, re.IGNORECASE | re.DOTALL)
            if match:
                metadata.abstract = match.group(1).strip()[:500]
        
        return metadata
    
    def _identify_sections(self, pages_text: List[str]) -> List[Section]:
        sections = []
        keywords = ['abstract', 'introduction', 'methods', 'results', 'conclusion']
        current = None
        
        for page_num, page_text in enumerate(pages_text):
            for line in page_text.split('\n'):
                line_lower = line.strip().lower()
                
                if any(line_lower == kw or line_lower.startswith(kw) for kw in keywords):
                    if current:
                        current.end_page = page_num
                        sections.append(current)
                    
                    current = Section(
                        heading=line.strip(),
                        content="",
                        start_page=page_num,
                        end_page=page_num
                    )
                elif current:
                    current.content += line + "\n"
        
        if current:
            current.end_page = len(pages_text) - 1
            sections.append(current)
        
        return sections
