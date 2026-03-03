"""Retrieval component for RAG system."""

import time
import logging
from typing import List, Dict, Any
from src.rag.models import RAGContext
from src.vector.search import SemanticSearchEngine

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieval component for RAG system."""
    
    def __init__(self, search_engine: SemanticSearchEngine, max_context_tokens: int = 3000):
        """
        Initialize RAG retriever.
        
        Args:
            search_engine: Semantic search engine instance
            max_context_tokens: Maximum tokens for context (leave room for query and response)
        """
        self.search_engine = search_engine
        self.max_context_tokens = max_context_tokens
        logger.info(f"RAGRetriever initialized with max_context_tokens={max_context_tokens}")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> RAGContext:
        """
        Retrieve relevant context for RAG generation.
        
        Strategy:
        1. Perform semantic search
        2. Select diverse, high-quality chunks
        3. Ensure context fits within token limits
        4. Format for LLM consumption
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            RAGContext with retrieved chunks and metadata
        """
        start_time = time.time()
        logger.info(f"Retrieving context for query: {query[:100]}...")
        
        try:
            # Retrieve more candidates for diversity
            candidates = self.search_engine.search(query, top_k * 2)
            logger.debug(f"Retrieved {len(candidates)} candidate chunks")
            
            # Select diverse, high-quality chunks
            selected_chunks = self._select_diverse_chunks(candidates, top_k)
            logger.debug(f"Selected {len(selected_chunks)} diverse chunks")
            
            # Ensure context fits token limits
            context_chunks = self._fit_context_to_limits(selected_chunks)
            logger.debug(f"Fitted {len(context_chunks)} chunks within token limits")
            
            retrieval_time = time.time() - start_time
            context_length = sum(len(chunk["text"]) for chunk in context_chunks)
            
            logger.info(f"Context retrieval completed in {retrieval_time:.2f}s, "
                       f"context_length={context_length} chars")
            
            return RAGContext(
                query=query,
                retrieved_chunks=context_chunks,
                total_chunks_found=len(candidates),
                retrieval_time=retrieval_time,
                context_length=context_length
            )
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            raise
    
    def _select_diverse_chunks(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Select diverse chunks to avoid redundancy.
        
        Args:
            candidates: List of candidate chunks
            top_k: Number of chunks to select
            
        Returns:
            List of diverse chunks
        """
        selected = []
        used_documents = []
        
        for chunk in candidates:
            doc_id = chunk["metadata"].get("document_id", "unknown")
            
            # Limit chunks per document for diversity (max 2 per document)
            if used_documents.count(doc_id) < 2:
                selected.append(chunk)
                used_documents.append(doc_id)
                
                if len(selected) >= top_k:
                    break
        
        return selected
    
    def _fit_context_to_limits(self, chunks: List[Dict]) -> List[Dict]:
        """
        Ensure context fits within token limits.
        
        Args:
            chunks: List of chunks to fit
            
        Returns:
            List of chunks that fit within token limits
        """
        fitted_chunks = []
        current_tokens = 0
        
        for chunk in chunks:
            # Rough token estimate: words * 1.3
            chunk_tokens = len(chunk["text"].split()) * 1.3
            
            if current_tokens + chunk_tokens <= self.max_context_tokens:
                fitted_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                logger.debug(f"Stopping at {len(fitted_chunks)} chunks to stay within token limit")
                break
        
        return fitted_chunks
