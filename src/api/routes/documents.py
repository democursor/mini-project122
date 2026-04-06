"""
Document management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
from typing import List
import logging
import tempfile
import shutil
from pathlib import Path

from src.api.models import DocumentUploadResponse, DocumentListResponse, DocumentMetadata
from src.services.document_service import DocumentService
from src.auth.dependencies import get_current_user
from src.auth.supabase_db import DocumentDB

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
document_service = DocumentService()

def get_document_db():
    return DocumentDB()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
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
        document_db = get_document_db()
        user_id = current_user["user_id"]
        
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
        
        # Store in Supabase
        document_db.upsert_document({
            "id": doc_id,
            "user_id": user_id,
            "filename": file.filename,
            "status": "processing"
        })
        
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
async def list_documents(current_user: dict = Depends(get_current_user)):
    """
    List all uploaded documents with their metadata
    """
    try:
        document_db = get_document_db()
        user_id = current_user["user_id"]
        
        # Get documents from Supabase for this user
        db_docs = document_db.list_documents(user_id)
        db_doc_ids = {doc["id"] for doc in db_docs}
        
        # Get documents from document service
        all_documents = await document_service.list_documents()
        
        # Filter to only show documents belonging to this user
        user_documents = [doc for doc in all_documents if doc.document_id in db_doc_ids]
        
        # If no documents from service but have in DB, return DB docs
        if not user_documents and db_docs:
            user_documents = [
                DocumentMetadata(
                    document_id=doc["id"],
                    filename=doc["filename"],
                    title=doc.get("title"),
                    authors=doc.get("authors", []),
                    status=doc.get("status", "unknown"),
                    pages=doc.get("pages"),
                    keywords=doc.get("keywords", []),
                    abstract=doc.get("abstract"),
                    year=doc.get("year"),
                    upload_date=doc.get("upload_date")
                )
                for doc in db_docs
            ]
        
        return DocumentListResponse(
            documents=user_documents,
            total=len(user_documents)
        )
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get metadata for a specific document
    """
    try:
        document_db = get_document_db()
        # Verify user owns this document
        db_doc = document_db.get_document(document_id, current_user["user_id"])
        if not db_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
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
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a document and all associated data
    """
    try:
        document_db = get_document_db()
        # Delete from Supabase
        db_success = document_db.delete_document(document_id, current_user["user_id"])
        if not db_success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from document service
        success = await document_service.delete_document(document_id)
        if not success:
            logger.warning(f"Document {document_id} deleted from DB but not from service")
        
        return {"message": "Document deleted successfully", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
