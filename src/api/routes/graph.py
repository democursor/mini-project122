"""
Knowledge graph endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from src.api.models import (
    GraphStats, PaperNode, ConceptNode, RelatedPaper,
    ConceptSearchRequest, ConceptSearchResponse
)
from src.services.graph_service import GraphService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
graph_service = GraphService()

@router.get("/stats", response_model=GraphStats)
async def get_graph_statistics():
    """
    Get knowledge graph statistics
    """
    try:
        stats = await graph_service.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting graph statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/papers", response_model=List[PaperNode])
async def list_papers(limit: int = Query(default=50, ge=1, le=200)):
    """
    List all papers in the knowledge graph
    """
    try:
        papers = await graph_service.list_papers(limit)
        return papers
    except Exception as e:
        logger.error(f"Error listing papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/papers/{paper_id}/related", response_model=List[RelatedPaper])
async def get_related_papers(paper_id: str, limit: int = Query(default=10, ge=1, le=50)):
    """
    Find papers related to a given paper based on shared concepts
    """
    try:
        related = await graph_service.find_related_papers(paper_id, limit)
        return related
    except Exception as e:
        logger.error(f"Error finding related papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/concepts", response_model=List[ConceptNode])
async def list_top_concepts(limit: int = Query(default=50, ge=1, le=200)):
    """
    List top concepts by frequency
    """
    try:
        concepts = await graph_service.list_top_concepts(limit)
        return concepts
    except Exception as e:
        logger.error(f"Error listing concepts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/concepts/search", response_model=ConceptSearchResponse)
async def search_concept(request: ConceptSearchRequest):
    """
    Search for papers and related concepts by concept name
    """
    try:
        result = await graph_service.search_concept(
            concept_name=request.concept_name,
            limit=request.limit
        )
        return result
    except Exception as e:
        logger.error(f"Error searching concept: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
