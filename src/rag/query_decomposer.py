"""
Query decomposition for complex queries in agentic RAG system.
"""
import logging
from typing import List
from src.rag.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decomposes complex queries into sub-queries."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize query decomposer.
        
        Args:
            llm_client: LLM client for query decomposition
        """
        self.llm_client = llm_client
    
    def decompose(self, query: str, min_subqueries: int = 3, max_subqueries: int = 6) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Args:
            query: Original complex query
            min_subqueries: Minimum number of sub-queries
            max_subqueries: Maximum number of sub-queries
            
        Returns:
            List of sub-queries
        """
        prompt = self._build_decomposition_prompt(query, min_subqueries, max_subqueries)
        
        try:
            response = self.llm_client.generate_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse sub-queries from response
            sub_queries = self._parse_subqueries(response)
            
            # Validate count
            if len(sub_queries) < min_subqueries:
                logger.warning(f"Only {len(sub_queries)} sub-queries generated, expected {min_subqueries}")
                # Add the original query as fallback
                sub_queries.append(query)
            
            if len(sub_queries) > max_subqueries:
                logger.info(f"Truncating {len(sub_queries)} sub-queries to {max_subqueries}")
                sub_queries = sub_queries[:max_subqueries]
            
            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            # Fallback: return original query
            return [query]
    
    def _build_decomposition_prompt(self, query: str, min_count: int, max_count: int) -> str:
        """Build prompt for query decomposition."""
        return f"""You are a research assistant helping to break down complex questions into simpler sub-questions.

Original Question: {query}

Break this question down into {min_count}-{max_count} specific sub-questions that, when answered together, would provide a comprehensive answer to the original question.

Guidelines:
- Each sub-question should focus on a specific aspect
- Sub-questions should be clear and answerable independently
- Cover different angles: definitions, mechanisms, impacts, applications, etc.
- Number each sub-question (1., 2., 3., etc.)
- Keep sub-questions concise and focused

Sub-questions:"""
    
    def _parse_subqueries(self, response: str) -> List[str]:
        """
        Parse sub-queries from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of parsed sub-queries
        """
        sub_queries = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.) and bullet points
            cleaned = line
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*', '•']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            
            # Skip if too short or looks like a header
            if len(cleaned) < 10 or cleaned.endswith(':'):
                continue
            
            sub_queries.append(cleaned)
        
        return sub_queries
