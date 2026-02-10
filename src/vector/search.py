"""Semantic search engine"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .embedder import EmbeddingGenerator
from .store import VectorStore


@dataclass
class SearchQuery:
    """Represents a processed search query"""
    original_query: str
    processed_query: str
    embedding: np.ndarray
    filters: Dict[str, Any]
    top_k: int = 10


class QueryProcessor:
    """Processes and optimizes search queries"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)
    
    def process_query(self, query: str, filters: Optional[Dict] = None,
                     top_k: int = 10) -> SearchQuery:
        """Process raw query into structured search query"""
        processed_query = self._clean_query(query)
        extracted_filters = self._extract_filters(processed_query)
        
        combined_filters = filters or {}
        combined_filters.update(extracted_filters)
        
        query_embedding = self.embedding_generator.generate_single_embedding(processed_query)
        
        return SearchQuery(
            original_query=query,
            processed_query=processed_query,
            embedding=query_embedding,
            filters=combined_filters,
            top_k=top_k
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        cleaned = re.sub(r'\s+', ' ', query.strip())
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
        cleaned = cleaned.lower()
        return cleaned
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from query text"""
        filters = {}
        
        # Year filters
        year_match = re.search(r'after (\d{4})', query)
        if year_match:
            filters["year"] = {"$gte": int(year_match.group(1))}
        
        year_match = re.search(r'before (\d{4})', query)
        if year_match:
            filters["year"] = {"$lte": int(year_match.group(1))}
        
        year_match = re.search(r'in (\d{4})', query)
        if year_match:
            filters["year"] = int(year_match.group(1))
        
        return filters


class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self, vector_store: VectorStore, 
                 query_processor: QueryProcessor):
        self.vector_store = vector_store
        self.query_processor = query_processor
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, top_k: int = 10,
              filters: Optional[Dict] = None) -> List[Dict]:
        """Perform semantic search"""
        search_query = self.query_processor.process_query(
            query, filters, top_k * 2
        )
        
        candidates = self.vector_store.search_similar(
            query_embedding=search_query.embedding,
            top_k=search_query.top_k,
            filters=search_query.filters
        )
        
        ranked_results = self._rerank_results(search_query, candidates)
        final_results = ranked_results[:top_k]
        enhanced_results = self._enhance_results(final_results, search_query)
        
        self.logger.info(f"Search completed: {len(final_results)} results for '{query}'")
        return enhanced_results
    
    def _rerank_results(self, query: SearchQuery, 
                       candidates: List[Dict]) -> List[Dict]:
        """Re-rank search results using multiple signals"""
        for result in candidates:
            base_score = result["similarity_score"]
            
            term_boost = self._calculate_term_frequency_boost(
                query.processed_query, result["document"]
            )
            
            section_boost = self._calculate_section_boost(result["metadata"])
            
            final_score = (
                base_score * 0.7 +
                term_boost * 0.2 +
                section_boost * 0.1
            )
            
            result["final_score"] = final_score
            result["ranking_factors"] = {
                "similarity": base_score,
                "term_frequency": term_boost,
                "section": section_boost
            }
        
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    
    def _calculate_term_frequency_boost(self, query: str, document: str) -> float:
        """Calculate boost based on query term frequency in document"""
        query_terms = query.lower().split()
        document_lower = document.lower()
        
        term_matches = sum(1 for term in query_terms if term in document_lower)
        
        if not query_terms:
            return 0.0
        
        return term_matches / len(query_terms)
    
    def _calculate_section_boost(self, metadata: Dict) -> float:
        """Calculate boost based on document section"""
        section_weights = {
            "abstract": 1.0,
            "introduction": 0.8,
            "conclusion": 0.7,
            "results": 0.6,
            "methods": 0.5,
            "references": 0.2
        }
        
        section = metadata.get("section_heading", "").lower()
        
        for section_name, weight in section_weights.items():
            if section_name in section:
                return weight
        
        return 0.5
    
    def _enhance_results(self, results: List[Dict], query: SearchQuery) -> List[Dict]:
        """Enhance results with additional metadata and formatting"""
        enhanced = []
        
        for result in results:
            enhanced_result = {
                "chunk_id": result["chunk_id"],
                "document_id": result["metadata"]["document_id"],
                "text": result["document"],
                "similarity_score": result["similarity_score"],
                "final_score": result["final_score"],
                "metadata": result["metadata"],
                "ranking_factors": result.get("ranking_factors", {})
            }
            
            enhanced.append(enhanced_result)
        
        return enhanced
