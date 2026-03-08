import logging
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from .models import PaperNode, ConceptNode


logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds and maintains the research knowledge graph"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.database = database
            self.logger = logger
            self._verify_connection()
            self._create_constraints()
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _verify_connection(self):
        """Verify Neo4j connection"""
        with self.driver.session(database=self.database) as session:
            result = session.run("RETURN 1 as test")
            result.single()
    
    def _create_constraints(self):
        """Create unique constraints for nodes"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
                session.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.normalized_name IS UNIQUE")
                logger.info("Created graph constraints")
            except Exception as e:
                logger.warning(f"Constraint creation warning: {e}")
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def create_paper_node(self, paper: PaperNode) -> str:
        """Create paper node in graph"""
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(self._create_paper_tx, paper)
            logger.info(f"Created paper node: {paper.id}")
            return result
    
    def _create_paper_tx(self, tx, paper: PaperNode):
        """Transaction to create paper node"""
        query = """
        MERGE (p:Paper {id: $paper_id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.year = $year,
            p.page_count = $page_count,
            p.updated_at = datetime()
        RETURN p.id as paper_id
        """
        
        result = tx.run(query,
            paper_id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            page_count=paper.page_count
        )
        
        record = result.single()
        return record["paper_id"] if record else None
    
    def create_concept_node(self, concept: ConceptNode):
        """Create or update concept node"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._create_concept_tx, concept)
            logger.debug(f"Created/updated concept: {concept.name}")
    
    def _create_concept_tx(self, tx, concept: ConceptNode):
        """Transaction to create concept node"""
        query = """
        MERGE (c:Concept {normalized_name: $normalized_name})
        ON CREATE SET 
            c.id = $concept_id,
            c.name = $name,
            c.type = $type,
            c.frequency = 1
        ON MATCH SET
            c.frequency = c.frequency + 1
        RETURN c.id as concept_id
        """
        
        tx.run(query,
            concept_id=concept.id,
            name=concept.name,
            normalized_name=concept.normalized_name,
            type=concept.type
        )
    
    def create_mentions_relationship(self, paper_id: str, concept_name: str, 
                                    frequency: int = 1, confidence: float = 1.0):
        """Create MENTIONS relationship between paper and concept"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._create_mentions_tx, 
                paper_id, concept_name, frequency, confidence
            )
    
    def _create_mentions_tx(self, tx, paper_id: str, concept_name: str, 
                           frequency: int, confidence: float):
        """Transaction to create MENTIONS relationship"""
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (c:Concept {normalized_name: $concept_name})
        MERGE (p)-[r:MENTIONS]->(c)
        ON CREATE SET
            r.frequency = $frequency,
            r.confidence = $confidence
        ON MATCH SET
            r.frequency = r.frequency + $frequency
        """
        
        tx.run(query,
            paper_id=paper_id,
            concept_name=concept_name,
            frequency=frequency,
            confidence=confidence
        )
    
    def create_related_concepts(self, concept1: str, concept2: str, strength: float = 0.5):
        """Create RELATED_TO relationship between concepts"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._create_related_tx,
                concept1, concept2, strength
            )
    
    def _create_related_tx(self, tx, concept1: str, concept2: str, strength: float):
        """Transaction to create RELATED_TO relationship"""
        query = """
        MATCH (c1:Concept {normalized_name: $concept1})
        MATCH (c2:Concept {normalized_name: $concept2})
        WHERE c1 <> c2
        MERGE (c1)-[r:RELATED_TO]-(c2)
        ON CREATE SET
            r.strength = $strength,
            r.papers_count = 1
        ON MATCH SET
            r.papers_count = r.papers_count + 1,
            r.strength = (r.strength + $strength) / 2
        """
        
        tx.run(query,
            concept1=concept1,
            concept2=concept2,
            strength=strength
        )
    
    def build_from_parsed_data(self, parsed_json: Dict):
        """Build graph from parsed JSON data"""
        try:
            # Extract paper data
            doc_id = parsed_json.get('document_id')
            parsed_data = parsed_json.get('parsed_data', {})
            metadata = parsed_data.get('metadata', {})
            
            # Create paper node
            paper = PaperNode(
                id=doc_id,
                title=metadata.get('title', 'Unknown'),
                abstract=metadata.get('abstract', ''),
                year=metadata.get('year'),
                page_count=parsed_data.get('page_count', 0)
            )
            self.create_paper_node(paper)
            
            # Extract and create concept nodes
            concepts_data = parsed_json.get('concepts', [])
            all_concepts = []
            
            for chunk_concepts in concepts_data:
                # Process entities
                for entity in chunk_concepts.get('entities', []):
                    concept_name = entity.get('text', '').lower().strip()
                    if concept_name:
                        concept = ConceptNode(
                            id=f"concept_{concept_name.replace(' ', '_')}",
                            name=entity.get('text', ''),
                            normalized_name=concept_name,
                            type=entity.get('label', 'UNKNOWN')
                        )
                        self.create_concept_node(concept)
                        self.create_mentions_relationship(
                            doc_id, 
                            concept_name,
                            frequency=1,
                            confidence=entity.get('confidence', 1.0)
                        )
                        all_concepts.append(concept_name)
                
                # Process keyphrases
                for keyphrase in chunk_concepts.get('keyphrases', []):
                    phrase = keyphrase.get('phrase', '').lower().strip()
                    if phrase:
                        concept = ConceptNode(
                            id=f"concept_{phrase.replace(' ', '_')}",
                            name=keyphrase.get('phrase', ''),
                            normalized_name=phrase,
                            type='KEYPHRASE'
                        )
                        self.create_concept_node(concept)
                        self.create_mentions_relationship(
                            doc_id,
                            phrase,
                            frequency=1,
                            confidence=keyphrase.get('score', 0.5)
                        )
                        all_concepts.append(phrase)
            
            # Create concept-concept relationships (co-occurrence)
            for i, concept1 in enumerate(all_concepts):
                for concept2 in all_concepts[i+1:]:
                    if concept1 != concept2:
                        self.create_related_concepts(concept1, concept2, strength=0.5)
            
            logger.info(f"Built graph for document {doc_id} with {len(all_concepts)} concepts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build graph from parsed data: {e}")
            return False
    
    def clear_graph(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._clear_graph_tx)
            logger.warning("Cleared entire graph")
    
    def _clear_graph_tx(self, tx):
        """Transaction to clear graph"""
        tx.run("MATCH (n) DETACH DELETE n")
    
    def delete_paper_node(self, paper_id: str):
        """Delete a paper node and all its relationships"""
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(self._delete_paper_tx, paper_id)
            logger.info(f"Deleted paper node and relationships: {paper_id}")
            return result
    
    def _delete_paper_tx(self, tx, paper_id: str):
        """Transaction to delete paper node"""
        query = """
        MATCH (p:Paper {id: $paper_id})
        DETACH DELETE p
        RETURN count(p) as deleted_count
        """
        result = tx.run(query, paper_id=paper_id)
        record = result.single()
        return record["deleted_count"] if record else 0
