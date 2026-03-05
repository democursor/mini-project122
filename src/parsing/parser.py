import logging
import re
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import gc

from .models import Metadata, Section, ParsedDocument

logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """Custom exception for PDF parsing errors"""
    pass


class PDFParser:
    """Optimized PDF parser with robust error handling for large documents"""
    
    def __init__(self, max_pages: int = 1000, chunk_size: int = 50):
        """
        Initialize parser with limits for large documents
        
        Args:
            max_pages: Maximum number of pages to process
            chunk_size: Number of pages to process at once (for memory optimization)
        """
        self.max_pages = max_pages
        self.chunk_size = chunk_size
    
    def parse(self, pdf_path: Path, document_id: str) -> ParsedDocument:
        """
        Parse PDF with optimized memory handling and error recovery
        
        Args:
            pdf_path: Path to PDF file
            document_id: Unique document identifier
            
        Returns:
            ParsedDocument object
            
        Raises:
            ParsingError: If parsing fails
        """
        import fitz
        
        if not pdf_path.exists():
            raise ParsingError(f"PDF file not found: {pdf_path}")
        
        doc = None
        try:
            # Open document with error handling
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                raise ParsingError(f"Failed to open PDF: {str(e)}")
            
            # Check page count
            if doc.page_count == 0:
                raise ParsingError("PDF has no pages")
            
            if doc.page_count > self.max_pages:
                logger.warning(f"Document has {doc.page_count} pages, limiting to {self.max_pages}")
            
            # Extract metadata with error handling
            metadata = self._extract_metadata_safe(doc)
            
            # Process pages in chunks for large documents
            pages_text = self._extract_text_chunked(doc)
            
            # Build full text efficiently
            full_text = self._build_full_text(pages_text)
            
            # Identify sections
            sections = self._identify_sections(pages_text)
            
            return ParsedDocument(
                document_id=document_id,
                metadata=metadata,
                sections=sections,
                full_text=full_text,
                page_count=min(doc.page_count, self.max_pages),
                parsing_date=datetime.now().isoformat()
            )
            
        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing PDF: {str(e)}", exc_info=True)
            raise ParsingError(f"Unexpected parsing error: {str(e)}")
        finally:
            if doc:
                doc.close()
            # Force garbage collection for large documents
            gc.collect()
    
    def _extract_text_chunked(self, doc) -> List[str]:
        """
        Extract text from pages in chunks to optimize memory usage
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of page texts
        """
        pages_text = []
        total_pages = min(doc.page_count, self.max_pages)
        
        for chunk_start in range(0, total_pages, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_pages)
            
            logger.debug(f"Processing pages {chunk_start+1} to {chunk_end}")
            
            for page_num in range(chunk_start, chunk_end):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")
                    pages_text.append(text if text else "")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num+1}: {str(e)}")
                    pages_text.append("")  # Add empty string for failed pages
            
            # Clear memory after each chunk
            if chunk_end < total_pages:
                gc.collect()
        
        return pages_text
    
    def _build_full_text(self, pages_text: List[str]) -> str:
        """
        Build full text efficiently with memory optimization
        
        Args:
            pages_text: List of page texts
            
        Returns:
            Combined full text
        """
        try:
            # Use join for memory efficiency
            return "\n\n".join(pages_text)
        except MemoryError:
            logger.error("Out of memory building full text, truncating")
            # Fallback: use first half of pages
            half = len(pages_text) // 2
            return "\n\n".join(pages_text[:half])
    
    def _extract_metadata_safe(self, doc) -> Metadata:
        """
        Extract metadata with error handling
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Metadata object
        """
        try:
            return self._extract_metadata(doc)
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {str(e)}, using defaults")
            return Metadata(title="Unknown", authors=["Unknown"])
    
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
