"""
Search service - handles semantic search operations
"""
import logging
from typing import List, Optional, Dict, Any

from src.vector.search import SemanticSearchEngine
from src.api.models import SearchResult
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class SearchService:
    """Service for semantic search"""
    
    def __init__(self):
        self.config = load_config()
        self.search_engine = SemanticSearchEngine(self.config)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of search results
        """
        try:
            # Perform search
            results = self.search_engine.search(query, top_k=top_k)
            
            # Convert to API models
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    document_id=result.get('document_id', ''),
                    title=result.get('title', 'Untitled'),
                    authors=result.get('authors'),
                    excerpt=result.get('text', '')[:500],  # Limit excerpt length
                    score=result.get('score', 0.0),
                    chunk_id=result.get('chunk_id', '')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise
    
    async def find_similar(
        self,
        document_id: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document
        
        Args:
            document_id: Document ID
            top_k: Number of results
            
        Returns:
            List of similar documents
        """
        try:
            # Get document chunks
            results = self.search_engine.find_similar_documents(document_id, top_k=top_k)
            
            # Convert to API models
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    document_id=result.get('document_id', ''),
                    title=result.get('title', 'Untitled'),
                    authors=result.get('authors'),
                    excerpt=result.get('text', '')[:500],
                    score=result.get('score', 0.0),
                    chunk_id=result.get('chunk_id', '')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise
