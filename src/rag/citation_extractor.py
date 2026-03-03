"""Citation extraction and validation for RAG responses."""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extracts and validates citations from LLM responses."""
    
    def __init__(self):
        """Initialize citation extractor."""
        self.citation_pattern = r'\[([^\]]+)\]'
        logger.info("CitationExtractor initialized")
    
    def extract_citations(self, response: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract citations from LLM response and validate against context.
        
        Args:
            response: LLM generated response
            context_chunks: Context chunks used for generation
            
        Returns:
            Dictionary with extracted citations and validation results
        """
        try:
            # Find all citations in response
            citations = re.findall(self.citation_pattern, response)
            logger.debug(f"Found {len(citations)} citations in response")
            
            if not citations:
                return {
                    "total_citations": 0,
                    "citations": [],
                    "citation_accuracy": 0.0
                }
            
            # Validate citations against context
            validated_citations = []
            for citation in citations:
                validation = self._validate_citation(citation, context_chunks)
                validated_citations.append(validation)
            
            valid_count = sum(1 for c in validated_citations if c["valid"])
            accuracy = valid_count / len(citations) if citations else 0.0
            
            logger.info(f"Citation extraction complete: {valid_count}/{len(citations)} valid, "
                       f"accuracy={accuracy:.2%}")
            
            return {
                "total_citations": len(citations),
                "citations": validated_citations,
                "citation_accuracy": accuracy
            }
            
        except Exception as e:
            logger.error(f"Error extracting citations: {e}", exc_info=True)
            return {
                "total_citations": 0,
                "citations": [],
                "citation_accuracy": 0.0,
                "error": str(e)
            }
    
    def _validate_citation(self, citation: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a single citation against context.
        
        Args:
            citation: Citation text to validate
            context_chunks: Context chunks to validate against
            
        Returns:
            Validation result dictionary
        """
        citation_lower = citation.lower()
        
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            title = metadata.get("title", "").lower()
            authors = metadata.get("authors", [])
            
            # Normalize authors to lowercase strings
            if isinstance(authors, list):
                authors_lower = [str(a).lower() for a in authors]
            else:
                authors_lower = [str(authors).lower()]
            
            # Check if citation matches title (at least one significant word)
            title_words = [w for w in title.split() if len(w) > 3]
            citation_words = [w for w in citation_lower.split() if len(w) > 3]
            title_match = any(word in title for word in citation_words)
            
            # Check if citation matches any author
            author_match = any(author_name in citation_lower for author_name in authors_lower)
            
            if title_match or author_match:
                logger.debug(f"Citation '{citation}' validated against paper: {metadata.get('title')}")
                return {
                    "citation": citation,
                    "valid": True,
                    "matched_paper": {
                        "title": metadata.get("title"),
                        "authors": metadata.get("authors"),
                        "document_id": metadata.get("document_id")
                    }
                }
        
        logger.debug(f"Citation '{citation}' could not be validated")
        return {
            "citation": citation,
            "valid": False,
            "matched_paper": None
        }
