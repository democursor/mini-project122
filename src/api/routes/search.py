"""
Semantic search endpoints
"""
from fastapi import APIRouter, HTTPException
import logging

from src.api.models import SearchRequest, SearchResponse
from src.services.search_service import SearchService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
search_service = SearchService()

@router.post("/", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search across all documents
    
    Uses vector embeddings to find semantically similar content
    """
    try:
        results = await search_service.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results)
        )
    
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/{document_id}")
async def find_similar_documents(document_id: str, top_k: int = 10):
    """
    Find documents similar to a given document
    """
    try:
        results = await search_service.find_similar(document_id, top_k)
        return {
            "document_id": document_id,
            "similar_documents": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error finding similar documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
