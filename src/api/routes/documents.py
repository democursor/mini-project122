"""
Document management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from typing import List
import logging
import tempfile
import shutil
from pathlib import Path

from src.api.models import DocumentUploadResponse, DocumentListResponse, DocumentMetadata
from src.services.document_service import DocumentService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
document_service = DocumentService()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF document for processing
    
    The document will be:
    1. Validated
    2. Stored
    3. Parsed
    4. Chunked
    5. Concepts extracted
    6. Knowledge graph built
    7. Embeddings generated
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)
        
        # Process document
        doc_id = await document_service.upload_and_process(
            file_path=tmp_path,
            filename=file.filename,
            background_tasks=background_tasks
        )
        
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status="processing",
            message="Document uploaded successfully and processing started"
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents with their metadata
    """
    try:
        documents = await document_service.list_documents()
        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    """
    Get metadata for a specific document
    """
    try:
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all associated data
    """
    try:
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
