"""
Document service - handles document upload and processing
"""
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import BackgroundTasks
import asyncio
import uuid
import json
from datetime import datetime

from src.ingestion.validator import PDFValidator
from src.ingestion.storage import PDFStorage
from src.orchestration.workflow import DocumentProcessor
from src.api.models import DocumentMetadata
from src.utils.config import Config

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for document management"""
    
    def __init__(self):
        self.config = Config()
        max_size = self.config.get('storage.max_file_size_mb', 50)
        pdf_dir = self.config.get('storage.pdf_directory', './data/pdfs')
        
        self.validator = PDFValidator(max_file_size_mb=max_size)
        self.storage = PDFStorage(base_dir=pdf_dir)
        self.processor = DocumentProcessor(self.config)
        self.metadata_file = Path('./data/documents_metadata.json')
        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self):
        """Ensure metadata file exists"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            self.metadata_file.write_text('{}')
    
    def _load_metadata(self) -> dict:
        """Load all document metadata"""
        try:
            return json.loads(self.metadata_file.read_text())
        except:
            return {}
    
    def _save_metadata(self, metadata: dict):
        """Save all document metadata"""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))
    
    async def upload_and_process(
        self,
        file_path: Path,
        filename: str,
        background_tasks: BackgroundTasks
    ) -> str:
        """
        Upload and process a document
        
        Args:
            file_path: Path to uploaded file
            filename: Original filename
            background_tasks: FastAPI background tasks
            
        Returns:
            Document ID
        """
        try:
            # Validate PDF
            validation_result = self.validator.validate(file_path)
            if not validation_result.is_valid:
                raise ValueError(f"Invalid PDF file: {', '.join(validation_result.errors)}")
            
            # Generate document ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Store PDF
            pdf_path = self.storage.store(file_path, doc_id)
            logger.info(f"Document stored with ID: {doc_id}")
            
            # Save metadata
            metadata = self._load_metadata()
            metadata[doc_id] = {
                'document_id': doc_id,
                'filename': filename,
                'upload_date': datetime.now().isoformat(),
                'status': 'processing',
                'pdf_path': str(pdf_path)
            }
            self._save_metadata(metadata)
            
            # Process in background
            background_tasks.add_task(
                self._process_document_sync,
                doc_id,
                pdf_path
            )
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error in upload_and_process: {str(e)}")
            raise
    
    def _process_document_sync(self, doc_id: str, pdf_path: Path):
        """
        Synchronous wrapper for document processing
        (runs in background task)
        """
        try:
            logger.info(f"Starting background processing for document {doc_id}")
            self.processor.process_document(doc_id)
            
            # Update status
            metadata = self._load_metadata()
            if doc_id in metadata:
                metadata[doc_id]['status'] = 'completed'
                self._save_metadata(metadata)
            
            logger.info(f"Completed processing for document {doc_id}")
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            # Update status to failed
            metadata = self._load_metadata()
            if doc_id in metadata:
                metadata[doc_id]['status'] = 'failed'
                metadata[doc_id]['error'] = str(e)
                self._save_metadata(metadata)
    
    async def list_documents(self) -> List[DocumentMetadata]:
        """
        List all documents
        
        Returns:
            List of document metadata
        """
        try:
            metadata = self._load_metadata()
            
            result = []
            for doc_id, doc in metadata.items():
                result.append(DocumentMetadata(
                    document_id=doc_id,
                    filename=doc.get('filename', 'Unknown'),
                    title=doc.get('title'),
                    authors=doc.get('authors'),
                    year=doc.get('year'),
                    pages=doc.get('pages'),
                    upload_date=doc.get('upload_date'),
                    status=doc.get('status', 'unknown')
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None
        """
        try:
            metadata = self._load_metadata()
            doc = metadata.get(document_id)
            
            if not doc:
                return None
            
            return DocumentMetadata(
                document_id=document_id,
                filename=doc.get('filename', 'Unknown'),
                title=doc.get('title'),
                authors=doc.get('authors'),
                year=doc.get('year'),
                pages=doc.get('pages'),
                upload_date=doc.get('upload_date'),
                status=doc.get('status', 'unknown')
            )
            
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated data
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # Check if document exists
            metadata = self._load_metadata()
            if document_id not in metadata:
                return False
            
            # Delete from ChromaDB (vector embeddings)
            try:
                from src.vector.store import VectorStore
                vector_store = VectorStore(
                    persist_directory=self.config.get('vector.persist_directory', './data/chroma')
                )
                vector_store.delete_document_chunks(document_id)
                logger.info(f"Deleted vector embeddings for document {document_id}")
            except Exception as e:
                logger.error(f"Error deleting from ChromaDB: {e}")
            
            # Delete from Neo4j (knowledge graph)
            try:
                from src.graph.builder import KnowledgeGraphBuilder
                graph_builder = KnowledgeGraphBuilder(
                    uri=self.config.get('neo4j.uri'),
                    user=self.config.get('neo4j.user'),
                    password=self.config.get('neo4j.password'),
                    database=self.config.get('neo4j.database')
                )
                graph_builder.delete_paper_node(document_id)
                graph_builder.close()
                logger.info(f"Deleted knowledge graph data for document {document_id}")
            except Exception as e:
                logger.error(f"Error deleting from Neo4j: {e}")
            
            # Delete from file storage
            success = self.storage.delete(document_id)
            
            if success:
                # Remove from metadata
                del metadata[document_id]
                self._save_metadata(metadata)
                logger.info(f"Document {document_id} fully deleted from all systems")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
