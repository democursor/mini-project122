"""
Query intent classification for agentic RAG system.
"""
import logging
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query intent types."""
    SIMPLE = "simple"  # Direct factual question
    COMPLEX = "complex"  # Multi-faceted question requiring synthesis
    COMPARATIVE = "comparative"  # Comparison between concepts/papers


class QueryClassifier:
    """Classifies user queries by intent."""
    
    # Keywords that indicate complex queries
    COMPLEX_INDICATORS = [
        'detailed report', 'comprehensive', 'explain in detail',
        'tell me everything', 'all about', 'overview',
        'how does', 'why does', 'what are the implications',
        'analyze', 'discuss', 'elaborate', 'describe in depth'
    ]
    
    # Keywords that indicate comparative queries
    COMPARATIVE_INDICATORS = [
        'compare', 'difference between', 'versus', 'vs',
        'contrast', 'similarities', 'better than', 'worse than',
        'advantages', 'disadvantages', 'pros and cons'
    ]
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent.
        
        Args:
            query: User query string
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()
        
        # Check for comparative intent
        if any(indicator in query_lower for indicator in self.COMPARATIVE_INDICATORS):
            logger.info(f"Classified query as COMPARATIVE")
            return QueryIntent.COMPARATIVE
        
        # Check for complex intent
        if any(indicator in query_lower for indicator in self.COMPLEX_INDICATORS):
            logger.info(f"Classified query as COMPLEX")
            return QueryIntent.COMPLEX
        
        # Check query length and structure
        word_count = len(query.split())
        has_multiple_questions = query.count('?') > 1
        
        if word_count > 15 or has_multiple_questions:
            logger.info(f"Classified query as COMPLEX (length={word_count}, multiple_questions={has_multiple_questions})")
            return QueryIntent.COMPLEX
        
        # Default to simple
        logger.info(f"Classified query as SIMPLE")
        return QueryIntent.SIMPLE
    
    def get_classification_metadata(self, query: str) -> Dict[str, Any]:
        """
        Get detailed classification metadata.
        
        Args:
            query: User query string
            
        Returns:
            Dict with classification details
        """
        intent = self.classify(query)
        query_lower = query.lower()
        
        metadata = {
            'intent': intent.value,
            'word_count': len(query.split()),
            'question_count': query.count('?'),
            'matched_indicators': []
        }
        
        # Find matched indicators
        if intent == QueryIntent.COMPARATIVE:
            metadata['matched_indicators'] = [
                ind for ind in self.COMPARATIVE_INDICATORS 
                if ind in query_lower
            ]
        elif intent == QueryIntent.COMPLEX:
            metadata['matched_indicators'] = [
                ind for ind in self.COMPLEX_INDICATORS 
                if ind in query_lower
            ]
        
        return metadata
