"""
Graph searcher for concept-based query expansion
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class GraphSearcher:
    """Search and retrieve concepts from knowledge graph"""
    
    def __init__(self, neo4j_client):
        """Initialize graph searcher with Neo4j client"""
        self.client = neo4j_client
    
    def get_related_concepts(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Get related concepts from the knowledge graph based on query
        
        Args:
            query: Search query
            limit: Maximum number of concepts to return
            
        Returns:
            List of concept dictionaries with 'name' field
        """
        try:
            if not self.client.driver:
                return []
            
            # Extract keywords from query (simple approach)
            keywords = [word.lower() for word in query.split() if len(word) > 3]
            
            if not keywords:
                return []
            
            # Search for concepts matching keywords
            with self.client.driver.session(database=self.client.database) as session:
                # Build query to find concepts containing any keyword
                cypher_query = """
                MATCH (c:Concept)
                WHERE ANY(keyword IN $keywords WHERE toLower(c.name) CONTAINS keyword 
                      OR toLower(c.normalized_name) CONTAINS keyword)
                RETURN c.name as name, c.normalized_name as normalized_name
                LIMIT $limit
                """
                
                result = session.run(cypher_query, keywords=keywords, limit=limit)
                concepts = [{'name': record['name']} for record in result]
                
                logger.debug(f"Found {len(concepts)} related concepts for query")
                return concepts
                
        except Exception as e:
            logger.warning(f"Error retrieving concepts from graph: {e}")
            return []
