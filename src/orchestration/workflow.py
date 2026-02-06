import logging
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from src.ingestion import PDFValidator, PDFStorage, PDFUploader
from src.parsing import PDFParser
from src.chunking import SemanticChunker, ChunkingConfig
from src.extraction import ConceptExtractor

logger = logging.getLogger(__name__)


class ProcessingState(TypedDict):
    document_id: str
    file_path: Optional[Path]
    status: str
    error_message: Optional[str]
    parsed_data: Optional[dict]
    chunks: Optional[list]
    concepts: Optional[list]


class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.validator = PDFValidator(config.get('storage.max_file_size_mb', 50))
        self.storage = PDFStorage(config.get('storage.pdf_directory', './data/pdfs'))
        self.uploader = PDFUploader(self.validator, self.storage)
        self.parser = PDFParser()
        self.chunker = SemanticChunker(ChunkingConfig())
        self.extractor = ConceptExtractor()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(ProcessingState)
        
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("chunk", self._chunk_node)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("complete", self._complete_node)
        workflow.add_node("error", self._error_node)
        
        workflow.set_entry_point("parse")
        
        workflow.add_conditional_edges(
            "parse",
            self._check_parse_result,
            {
                "success": "chunk",
                "error": "error"
            }
        )
        
        workflow.add_conditional_edges(
            "chunk",
            self._check_chunk_result,
            {
                "success": "extract",
                "error": "error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract",
            self._check_extract_result,
            {
                "success": "complete",
                "error": "error"
            }
        )
        
        workflow.add_edge("complete", END)
        workflow.add_edge("error", END)
        
        return workflow.compile()
    
    def _parse_node(self, state: ProcessingState) -> ProcessingState:
        try:
            doc_id = state["document_id"]
            file_path = self.storage.retrieve(doc_id)
            
            parsed = self.parser.parse(file_path, doc_id)
            
            state["parsed_data"] = parsed.to_dict()
            state["status"] = "parsed"
            logger.info(f"Parsed document: {doc_id}")
            
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Parse error: {str(e)}"
            logger.error(f"Parse error: {e}")
        
        return state
    
    def _chunk_node(self, state: ProcessingState) -> ProcessingState:
        try:
            doc_id = state["document_id"]
            parsed_data = state["parsed_data"]
            
            all_chunks = []
            for section in parsed_data.get('sections', []):
                chunks = self.chunker.chunk_document(
                    doc_id,
                    section['content'],
                    section['heading']
                )
                all_chunks.extend([c.to_dict() for c in chunks])
            
            state["chunks"] = all_chunks
            state["status"] = "chunked"
            logger.info(f"Created {len(all_chunks)} chunks")
            
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Chunking error: {str(e)}"
            logger.error(f"Chunking error: {e}")
        
        return state
    
    def _extract_node(self, state: ProcessingState) -> ProcessingState:
        try:
            doc_id = state["document_id"]
            chunks = state["chunks"]
            
            all_concepts = []
            for chunk in chunks[:5]:  # Limit to first 5 chunks for demo
                result = self.extractor.extract(
                    chunk['chunk_id'],
                    doc_id,
                    chunk['text']
                )
                all_concepts.append(result.to_dict())
            
            state["concepts"] = all_concepts
            state["status"] = "extracted"
            logger.info(f"Extracted concepts from {len(all_concepts)} chunks")
            
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Extraction error: {str(e)}"
            logger.error(f"Extraction error: {e}")
        
        return state
    
    def _complete_node(self, state: ProcessingState) -> ProcessingState:
        state["status"] = "complete"
        logger.info(f"Processing complete: {state['document_id']}")
        return state
    
    def _error_node(self, state: ProcessingState) -> ProcessingState:
        logger.error(f"Processing failed: {state.get('error_message')}")
        return state
    
    def _check_parse_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "parsed" else "error"
    
    def _check_chunk_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "chunked" else "error"
    
    def _check_extract_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "extracted" else "error"
    
    def process_document(self, document_id: str) -> ProcessingState:
        initial_state = ProcessingState(
            document_id=document_id,
            file_path=None,
            status="uploaded",
            error_message=None,
            parsed_data=None,
            chunks=None,
            concepts=None
        )
        
        result = self.workflow.invoke(initial_state)
        return result
