"""
Graph service - handles knowledge graph operations
"""
import logging
from typing import List, Optional

from src.graph.queries import GraphQueryEngine
from src.api.models import GraphStats, PaperNode, ConceptNode, RelatedPaper, ConceptSearchResponse
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class GraphService:
    """Service for knowledge graph operations"""
    
    def __init__(self):
        self.config = load_config()
        try:
            self.query_engine = GraphQueryEngine(self.config)
            self.neo4j_available = True
        except Exception as e:
            logger.warning(f"Neo4j not available: {str(e)}")
            self.neo4j_available = False
    
    async def get_statistics(self) -> GraphStats:
        """
        Get knowledge graph statistics
        
        Returns:
            Graph statistics
        """
        try:
            if not self.neo4j_available:
                return GraphStats(
                    total_papers=0,
                    total_concepts=0,
                    total_mentions=0,
                    total_relationships=0
                )
            
            stats = self.query_engine.get_graph_statistics()
            
            return GraphStats(
                total_papers=stats.get('total_papers', 0),
                total_concepts=stats.get('total_concepts', 0),
                total_mentions=stats.get('total_mentions', 0),
                total_relationships=stats.get('total_relationships', 0)
            )
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise
    
    async def list_papers(self, limit: int = 50) -> List[PaperNode]:
        """
        List all papers in the graph
        
        Args:
            limit: Maximum number of papers
            
        Returns:
            List of paper nodes
        """
        try:
            if not self.neo4j_available:
                return []
            
            papers = self.query_engine.get_all_papers(limit=limit)
            
            result = []
            for paper in papers:
                result.append(PaperNode(
                    id=paper.get('id', ''),
                    title=paper.get('title', 'Untitled'),
                    authors=paper.get('authors'),
                    year=paper.get('year'),
                    abstract=paper.get('abstract')
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing papers: {str(e)}")
            raise
    
    async def find_related_papers(
        self,
        paper_id: str,
        limit: int = 10
    ) -> List[RelatedPaper]:
        """
        Find papers related to a given paper
        
        Args:
            paper_id: Paper ID
            limit: Maximum number of results
            
        Returns:
            List of related papers
        """
        try:
            if not self.neo4j_available:
                return []
            
            related = self.query_engine.find_related_papers(paper_id, limit=limit)
            
            result = []
            for item in related:
                paper_data = item.get('paper', {})
                result.append(RelatedPaper(
                    paper=PaperNode(
                        id=paper_data.get('id', ''),
                        title=paper_data.get('title', 'Untitled'),
                        authors=paper_data.get('authors'),
                        year=paper_data.get('year'),
                        abstract=paper_data.get('abstract')
                    ),
                    shared_concepts=item.get('shared_concepts', 0),
                    similarity_score=item.get('similarity_score', 0.0)
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding related papers: {str(e)}")
            raise
    
    async def list_top_concepts(self, limit: int = 50) -> List[ConceptNode]:
        """
        List top concepts by frequency
        
        Args:
            limit: Maximum number of concepts
            
        Returns:
            List of concept nodes
        """
        try:
            if not self.neo4j_available:
                return []
            
            concepts = self.query_engine.get_top_concepts(limit=limit)
            
            result = []
            for concept in concepts:
                result.append(ConceptNode(
                    name=concept.get('name', ''),
                    frequency=concept.get('frequency', 0),
                    type=concept.get('type')
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing concepts: {str(e)}")
            raise
    
    async def search_concept(
        self,
        concept_name: str,
        limit: int = 10
    ) -> ConceptSearchResponse:
        """
        Search for papers and related concepts by concept name
        
        Args:
            concept_name: Concept to search for
            limit: Maximum number of results
            
        Returns:
            Concept search response
        """
        try:
            if not self.neo4j_available:
                return ConceptSearchResponse(
                    concept=concept_name,
                    papers=[],
                    related_concepts=[]
                )
            
            # Find papers mentioning this concept
            papers_data = self.query_engine.find_papers_by_concept(concept_name, limit=limit)
            
            papers = []
            for paper in papers_data:
                papers.append(PaperNode(
                    id=paper.get('id', ''),
                    title=paper.get('title', 'Untitled'),
                    authors=paper.get('authors'),
                    year=paper.get('year'),
                    abstract=paper.get('abstract')
                ))
            
            # Find related concepts
            related_data = self.query_engine.find_related_concepts(concept_name, limit=limit)
            
            related_concepts = []
            for concept in related_data:
                related_concepts.append(ConceptNode(
                    name=concept.get('name', ''),
                    frequency=concept.get('frequency', 0),
                    type=concept.get('type')
                ))
            
            return ConceptSearchResponse(
                concept=concept_name,
                papers=papers,
                related_concepts=related_concepts
            )
            
        except Exception as e:
            logger.error(f"Error searching concept: {str(e)}")
            raise
