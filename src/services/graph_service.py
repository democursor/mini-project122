"""
Graph service - handles knowledge graph operations
"""
import logging
from typing import List, Optional

from src.graph.queries import GraphQueryEngine
from src.api.models import GraphStats, PaperNode, ConceptNode, RelatedPaper, ConceptSearchResponse

logger = logging.getLogger(__name__)

class GraphService:
    """Service for knowledge graph operations"""
    
    def __init__(self):
        from src.utils.config import Config
        self.config = Config()
        try:
            from neo4j import GraphDatabase
            
            uri = self.config.get('neo4j.uri', 'neo4j://localhost:7687')
            user = self.config.get('neo4j.user', 'neo4j')
            password = self.config.get('neo4j.password', '')
            database = self.config.get('neo4j.database', 'neo4j')
            
            # Create Neo4j driver
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test connection with specific database
            with driver.session(database=database) as session:
                session.run("RETURN 1")
            
            # Create query engine with driver
            self.query_engine = GraphQueryEngine(driver, database)
            self.neo4j_available = True
            logger.info(f"Neo4j connected successfully to {uri}/{database}")
            
        except Exception as e:
            logger.warning(f"Neo4j not available: {str(e)}")
            self.neo4j_available = False
            self.query_engine = None
    
    async def get_statistics(self) -> GraphStats:
        """
        Get knowledge graph statistics
        
        Returns:
            Graph statistics
        """
        if not self.neo4j_available or not self.query_engine:
            logger.info("Neo4j not available, returning empty statistics")
            return GraphStats(
                total_papers=0,
                total_concepts=0,
                total_mentions=0,
                total_relationships=0
            )
        
        try:
            stats = self.query_engine.get_graph_statistics()
            
            return GraphStats(
                total_papers=stats.get('papers', 0),
                total_concepts=stats.get('concepts', 0),
                total_mentions=stats.get('mentions', 0),
                total_relationships=stats.get('relationships', 0)
            )
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return GraphStats(
                total_papers=0,
                total_concepts=0,
                total_mentions=0,
                total_relationships=0
            )
    
    async def list_papers(self, limit: int = 50) -> List[PaperNode]:
        """
        List all papers in the graph
        
        Args:
            limit: Maximum number of papers
            
        Returns:
            List of paper nodes
        """
        if not self.neo4j_available or not self.query_engine:
            logger.info("Neo4j not available, returning empty paper list")
            return []
        
        try:
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
            return []
    
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
        if not self.neo4j_available or not self.query_engine:
            logger.info("Neo4j not available, returning empty related papers list")
            return []
        
        try:
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
            return []
    
    async def list_top_concepts(self, limit: int = 50) -> List[ConceptNode]:
        """
        List top concepts by frequency
        
        Args:
            limit: Maximum number of concepts
            
        Returns:
            List of concept nodes
        """
        if not self.neo4j_available or not self.query_engine:
            logger.info("Neo4j not available, returning empty concepts list")
            return []
        
        try:
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
            return []
    
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
        if not self.neo4j_available or not self.query_engine:
            logger.info("Neo4j not available, returning empty concept search results")
            return ConceptSearchResponse(
                concept=concept_name,
                papers=[],
                related_concepts=[]
            )
        
        try:
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
            return ConceptSearchResponse(
                concept=concept_name,
                papers=[],
                related_concepts=[]
            )
