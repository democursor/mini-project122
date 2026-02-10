import logging
from typing import List, Dict
from neo4j import GraphDatabase


logger = logging.getLogger(__name__)


class GraphQueryEngine:
    """Executes common graph queries"""
    
    def __init__(self, driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database
    
    def get_all_papers(self, limit: int = 10) -> List[Dict]:
        """Get all papers in the graph"""
        with self.driver.session(database=self.database) as session:
            return session.execute_read(self._get_all_papers_tx, limit)
    
    def _get_all_papers_tx(self, tx, limit: int):
        """Transaction to get all papers"""
        query = """
        MATCH (p:Paper)
        RETURN p.id as id, p.title as title, p.year as year, p.page_count as page_count
        ORDER BY p.updated_at DESC
        LIMIT $limit
        """
        result = tx.run(query, limit=limit)
        return [dict(record) for record in result]
    
    def get_all_concepts(self, limit: int = 20) -> List[Dict]:
        """Get all concepts ordered by frequency"""
        with self.driver.session(database=self.database) as session:
            return session.execute_read(self._get_all_concepts_tx, limit)
    
    def _get_all_concepts_tx(self, tx, limit: int):
        """Transaction to get all concepts"""
        query = """
        MATCH (c:Concept)
        RETURN c.name as name, c.type as type, c.frequency as frequency
        ORDER BY c.frequency DESC
        LIMIT $limit
        """
        result = tx.run(query, limit=limit)
        return [dict(record) for record in result]
    
    def find_papers_by_concept(self, concept_name: str) -> List[Dict]:
        """Find papers mentioning a specific concept"""
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                self._find_papers_by_concept_tx, 
                concept_name.lower()
            )
    
    def _find_papers_by_concept_tx(self, tx, concept_name: str):
        """Transaction to find papers by concept"""
        query = """
        MATCH (p:Paper)-[r:MENTIONS]->(c:Concept {normalized_name: $concept_name})
        RETURN p.id as id, p.title as title, p.year as year, 
               r.frequency as mentions, r.confidence as confidence
        ORDER BY r.frequency DESC
        """
        result = tx.run(query, concept_name=concept_name)
        return [dict(record) for record in result]
    
    def find_related_papers(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Find papers related through shared concepts"""
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                self._find_related_papers_tx, 
                paper_id, 
                limit
            )
    
    def _find_related_papers_tx(self, tx, paper_id: str, limit: int):
        """Transaction to find related papers"""
        query = """
        MATCH (source:Paper {id: $paper_id})
        MATCH (source)-[:MENTIONS]->(concept:Concept)<-[:MENTIONS]-(related:Paper)
        WHERE related <> source
        
        WITH related, count(concept) as shared_concepts,
             collect(concept.name)[0..5] as sample_concepts
        
        RETURN related.id as id,
               related.title as title,
               related.year as year,
               shared_concepts,
               sample_concepts
        ORDER BY shared_concepts DESC
        LIMIT $limit
        """
        result = tx.run(query, paper_id=paper_id, limit=limit)
        return [dict(record) for record in result]
    
    def find_related_concepts(self, concept_name: str, limit: int = 10) -> List[Dict]:
        """Find concepts related to a given concept"""
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                self._find_related_concepts_tx,
                concept_name.lower(),
                limit
            )
    
    def _find_related_concepts_tx(self, tx, concept_name: str, limit: int):
        """Transaction to find related concepts"""
        query = """
        MATCH (c:Concept {normalized_name: $concept_name})-[r:RELATED_TO]-(related:Concept)
        RETURN related.name as name,
               related.type as type,
               related.frequency as frequency,
               r.strength as relationship_strength,
               r.papers_count as papers_count
        ORDER BY r.strength DESC, r.papers_count DESC
        LIMIT $limit
        """
        result = tx.run(query, concept_name=concept_name, limit=limit)
        return [dict(record) for record in result]
    
    def get_graph_statistics(self) -> Dict:
        """Get overall graph statistics"""
        with self.driver.session(database=self.database) as session:
            return session.execute_read(self._get_statistics_tx)
    
    def _get_statistics_tx(self, tx):
        """Transaction to get graph statistics"""
        query = """
        MATCH (p:Paper)
        WITH count(p) as paper_count
        MATCH (c:Concept)
        WITH paper_count, count(c) as concept_count
        MATCH ()-[r:MENTIONS]->()
        WITH paper_count, concept_count, count(r) as mentions_count
        MATCH ()-[rel:RELATED_TO]-()
        RETURN paper_count, concept_count, mentions_count, count(rel)/2 as relationships_count
        """
        result = tx.run(query)
        record = result.single()
        if record:
            return {
                'papers': record['paper_count'],
                'concepts': record['concept_count'],
                'mentions': record['mentions_count'],
                'relationships': record['relationships_count']
            }
        return {'papers': 0, 'concepts': 0, 'mentions': 0, 'relationships': 0}
