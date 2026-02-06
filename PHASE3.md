# Phase 3: Knowledge Graph Construction

## Overview

Phase 3 transforms the extracted concepts and entities from Phase 2 into a comprehensive knowledge graph. This phase creates a network of interconnected nodes representing papers, concepts, authors, and venues, enabling powerful graph-based queries and discovery.

**Learning Objectives:**
- Understand graph database concepts and design
- Learn Cypher query language for Neo4j
- Master graph schema design and relationships
- Explore graph algorithms and traversal patterns
- Build scalable knowledge representation systems

**Key Concepts:**
- Graph databases vs relational databases
- Nodes, relationships, and properties
- Graph schema design
- Cypher query language
- Graph algorithms (centrality, community detection)
- Incremental graph updates

---

## Table of Contents

1. [Knowledge Graph Module](#knowledge-graph-module)
2. [Graph Schema Design](#graph-schema-design)
3. [Graph Construction Process](#graph-construction-process)
4. [Graph Queries and Analytics](#graph-queries-and-analytics)
5. [Learning Outcomes](#learning-outcomes)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Success Criteria](#success-criteria)

---

## Knowledge Graph Module

### Purpose

The Knowledge Graph Module creates and maintains a graph database that represents the relationships between papers, concepts, authors, and venues. This enables discovery of hidden connections and provides a foundation for intelligent search and recommendation.

**Why Knowledge Graphs Matter:**
- **Relationship Discovery:** Find connections between seemingly unrelated papers
- **Concept Evolution:** Track how concepts develop over time
- **Author Networks:** Identify collaboration patterns and expertise areas
- **Research Trends:** Discover emerging topics and declining areas

### 12.1 Graph Schema Design

#### Core Node Types

**1. Paper Nodes**
```cypher
// Paper node structure
CREATE (p:Paper {
  id: "doc_abc123",
  title: "Attention Is All You Need",
  abstract: "The dominant sequence transduction models...",
  year: 2017,
  venue: "NeurIPS",
  doi: "10.48550/arXiv.1706.03762",
  page_count: 15,
  citation_count: 0,  // Will be updated
  created_at: datetime(),
  updated_at: datetime()
})
```

**2. Concept Nodes**
```cypher
// Concept node structure
CREATE (c:Concept {
  id: "concept_transformer",
  name: "transformer",
  normalized_name: "transformer architecture",
  type: "METHOD",  // METHOD, DATASET, METRIC, DOMAIN
  frequency: 1,    // Number of papers mentioning this concept
  first_seen: date("2017-06-12"),
  last_seen: date("2024-01-15"),
  description: "Neural network architecture based on attention mechanisms"
})
```

**3. Author Nodes**
```cypher
// Author node structure
CREATE (a:Author {
  id: "author_vaswani",
  name: "Ashish Vaswani",
  normalized_name: "ashish vaswani",
  paper_count: 0,     // Will be calculated
  h_index: 0,         // Will be calculated
  first_publication: date("2017-06-12"),
  last_publication: date("2024-01-15"),
  primary_affiliation: "Google"
})
```

**4. Venue Nodes**
```cypher
// Venue node structure
CREATE (v:Venue {
  id: "venue_neurips",
  name: "NeurIPS",
  full_name: "Conference on Neural Information Processing Systems",
  type: "CONFERENCE",  // CONFERENCE, JOURNAL, WORKSHOP
  year_founded: 1987,
  paper_count: 0,      // Will be calculated
  impact_factor: 0.0   // Will be calculated
})
```

---

#### Relationship Types

**1. Paper-Concept Relationships**
```cypher
// Paper mentions concept
(p:Paper)-[:MENTIONS {
  frequency: 5,           // How many times mentioned
  confidence: 0.95,       // Extraction confidence
  context: "transformer architecture",
  first_mention_page: 1,
  extraction_method: "NER"
}]->(c:Concept)
```

**2. Paper-Author Relationships**
```cypher
// Author authored paper
(a:Author)-[:AUTHORED {
  position: 1,            // Author position (1st, 2nd, etc.)
  is_corresponding: true,
  affiliation: "Google Research"
}]->(p:Paper)
```

**3. Paper-Venue Relationships**
```cypher
// Paper published in venue
(p:Paper)-[:PUBLISHED_IN {
  year: 2017,
  volume: 30,
  pages: "5998-6008"
}]->(v:Venue)
```

**4. Paper-Paper Relationships**
```cypher
// Paper cites another paper
(p1:Paper)-[:CITES {
  citation_context: "building on the work of...",
  page_number: 2
}]->(p2:Paper)
```

**5. Concept-Concept Relationships**
```cypher
// Concepts are related (co-occurrence)
(c1:Concept)-[:RELATED_TO {
  strength: 0.8,          // Co-occurrence strength
  papers_count: 15,       // Number of papers with both concepts
  relationship_type: "co_occurrence"
}]->(c2:Concept)

// Concept evolution
(c1:Concept)-[:EVOLVED_TO {
  confidence: 0.7,
  time_gap_years: 2
}]->(c2:Concept)
```

**6. Author-Author Relationships**
```cypher
// Authors collaborated
(a1:Author)-[:COLLABORATED_WITH {
  papers_count: 3,
  first_collaboration: date("2017-06-12"),
  last_collaboration: date("2020-03-15"),
  venues: ["NeurIPS", "ICML"]
}]->(a2:Author)
```

---

### 12.2 Graph Construction Process

#### Graph Builder Implementation

```python
from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging

class KnowledgeGraphBuilder:
    """Builds and maintains the research knowledge graph"""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def create_paper_node(self, document: ParsedDocument) -> str:
        """
        Create paper node in graph.
        
        Returns:
            Paper node ID
        """
        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_paper_tx, document
            )
            return result
    
    def _create_paper_tx(self, tx, document: ParsedDocument):
        """Transaction to create paper node"""
        query = """
        MERGE (p:Paper {id: $paper_id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.year = $year,
            p.venue = $venue,
            p.doi = $doi,
            p.page_count = $page_count,
            p.created_at = datetime(),
            p.updated_at = datetime()
        RETURN p.id as paper_id
        """
        
        result = tx.run(query,
            paper_id=document.document_id,
            title=document.metadata.title,
            abstract=document.metadata.abstract,
            year=document.metadata.year,
            venue=document.metadata.venue,
            doi=document.metadata.doi,
            page_count=document.page_count
        )
        
        return result.single()["paper_id"]
    
    def create_concept_nodes(self, extraction_results: List[ConceptExtractionResult]):
        """Create concept nodes from extraction results"""
        with self.driver.session() as session:
            for result in extraction_results:
                # Create concept nodes
                for entity in result.entities:
                    session.write_transaction(
                        self._create_concept_tx, entity
                    )
                
                for keyphrase in result.keyphrases:
                    session.write_transaction(
                        self._create_keyphrase_concept_tx, keyphrase
                    )
    
    def _create_concept_tx(self, tx, entity: Entity):
        """Transaction to create concept node from entity"""
        query = """
        MERGE (c:Concept {normalized_name: $normalized_name})
        ON CREATE SET 
            c.id = $concept_id,
            c.name = $name,
            c.type = $type,
            c.frequency = 1,
            c.first_seen = date(),
            c.last_seen = date()
        ON MATCH SET
            c.frequency = c.frequency + 1,
            c.last_seen = date()
        RETURN c.id as concept_id
        """
        
        concept_id = f"concept_{entity.normalized_form or entity.text}".replace(" ", "_")
        
        tx.run(query,
            concept_id=concept_id,
            name=entity.text,
            normalized_name=entity.normalized_form or entity.text,
            type=entity.label
        )
    
    def create_relationships(self, document_id: str, 
                           extraction_results: List[ConceptExtractionResult]):
        """Create relationships between paper and concepts"""
        with self.driver.session() as session:
            for result in extraction_results:
                # Paper-Concept relationships
                for entity in result.entities:
                    session.write_transaction(
                        self._create_paper_concept_relationship_tx,
                        document_id, entity
                    )
                
                # Concept-Concept relationships
                for relationship in result.relationships:
                    session.write_transaction(
                        self._create_concept_relationship_tx,
                        relationship
                    )
    
    def _create_paper_concept_relationship_tx(self, tx, document_id: str, entity: Entity):
        """Create MENTIONS relationship between paper and concept"""
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (c:Concept {normalized_name: $concept_name})
        MERGE (p)-[r:MENTIONS]->(c)
        ON CREATE SET
            r.frequency = 1,
            r.confidence = $confidence,
            r.extraction_method = "NER"
        ON MATCH SET
            r.frequency = r.frequency + 1
        """
        
        tx.run(query,
            paper_id=document_id,
            concept_name=entity.normalized_form or entity.text,
            confidence=entity.confidence
        )
```

---

#### Deduplication Strategy

```python
class GraphDeduplicator:
    """Handles deduplication in knowledge graph"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
    
    def deduplicate_concepts(self):
        """
        Merge duplicate concept nodes.
        
        Strategy:
        1. Find concepts with similar names
        2. Calculate similarity scores
        3. Merge concepts above threshold
        4. Update all relationships
        """
        with self.graph.driver.session() as session:
            # Find potential duplicates
            duplicates = session.read_transaction(self._find_duplicate_concepts)
            
            for group in duplicates:
                if len(group) > 1:
                    # Merge concepts in group
                    primary_concept = group[0]  # Keep first as primary
                    for duplicate in group[1:]:
                        session.write_transaction(
                            self._merge_concepts_tx, primary_concept, duplicate
                        )
    
    def _find_duplicate_concepts(self, tx):
        """Find concepts that might be duplicates"""
        query = """
        MATCH (c:Concept)
        RETURN c.normalized_name as name, collect(c.id) as concept_ids
        """
        
        result = tx.run(query)
        
        # Group similar concepts
        groups = []
        processed = set()
        
        for record in result:
            name = record["name"]
            concept_ids = record["concept_ids"]
            
            if name not in processed:
                # Find similar names
                similar_group = [concept_ids[0]]  # Start with first concept
                
                # Simple similarity check (can be improved)
                for other_record in result:
                    other_name = other_record["name"]
                    if (other_name != name and 
                        other_name not in processed and
                        self._are_similar(name, other_name)):
                        similar_group.extend(other_record["concept_ids"])
                        processed.add(other_name)
                
                if len(similar_group) > 1:
                    groups.append(similar_group)
                
                processed.add(name)
        
        return groups
    
    def _are_similar(self, name1: str, name2: str) -> bool:
        """Check if two concept names are similar"""
        # Simple similarity check
        # In production, use more sophisticated methods
        
        # Check if one is abbreviation of other
        if name1.lower() in name2.lower() or name2.lower() in name1.lower():
            return True
        
        # Check edit distance
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        return similarity > 0.8
    
    def _merge_concepts_tx(self, tx, primary_id: str, duplicate_id: str):
        """Merge duplicate concept into primary concept"""
        query = """
        // Find both concepts
        MATCH (primary:Concept {id: $primary_id})
        MATCH (duplicate:Concept {id: $duplicate_id})
        
        // Update primary concept frequency
        SET primary.frequency = primary.frequency + duplicate.frequency
        
        // Move all relationships from duplicate to primary
        WITH primary, duplicate
        MATCH (duplicate)<-[r:MENTIONS]-(p:Paper)
        MERGE (p)-[new_r:MENTIONS]->(primary)
        ON CREATE SET new_r = properties(r)
        ON MATCH SET new_r.frequency = new_r.frequency + r.frequency
        DELETE r
        
        // Move concept-concept relationships
        WITH primary, duplicate
        MATCH (duplicate)-[r:RELATED_TO]-(other:Concept)
        WHERE other <> primary
        MERGE (primary)-[new_r:RELATED_TO]-(other)
        ON CREATE SET new_r = properties(r)
        ON MATCH SET new_r.strength = (new_r.strength + r.strength) / 2
        DELETE r
        
        // Delete duplicate concept
        DELETE duplicate
        """
        
        tx.run(query, primary_id=primary_id, duplicate_id=duplicate_id)
```

---

### 12.3 Concept Aggregation

#### Popularity Metrics

```python
class ConceptAnalyzer:
    """Analyzes concept popularity and trends"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
    
    def calculate_concept_popularity(self):
        """Calculate popularity metrics for all concepts"""
        with self.graph.driver.session() as session:
            session.write_transaction(self._update_concept_metrics_tx)
    
    def _update_concept_metrics_tx(self, tx):
        """Update concept popularity metrics"""
        query = """
        MATCH (c:Concept)<-[r:MENTIONS]-(p:Paper)
        WITH c, count(p) as paper_count, sum(r.frequency) as total_mentions
        SET c.paper_count = paper_count,
            c.total_mentions = total_mentions,
            c.popularity_score = paper_count * log(total_mentions + 1)
        """
        tx.run(query)
    
    def get_trending_concepts(self, time_window_days: int = 365) -> List[Dict]:
        """Get concepts trending in recent time window"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._get_trending_concepts_tx, time_window_days
            )
    
    def _get_trending_concepts_tx(self, tx, time_window_days: int):
        """Find trending concepts"""
        query = """
        MATCH (c:Concept)<-[:MENTIONS]-(p:Paper)
        WHERE p.created_at > datetime() - duration({days: $days})
        WITH c, count(p) as recent_papers
        WHERE recent_papers >= 3
        MATCH (c)<-[:MENTIONS]-(all_p:Paper)
        WITH c, recent_papers, count(all_p) as total_papers
        RETURN c.name as concept,
               c.type as type,
               recent_papers,
               total_papers,
               (recent_papers * 1.0 / total_papers) as trend_ratio
        ORDER BY trend_ratio DESC, recent_papers DESC
        LIMIT 20
        """
        
        result = tx.run(query, days=time_window_days)
        return [dict(record) for record in result]
```

---

### 12.4 Bidirectional Relationships

#### Relationship Management

```python
class RelationshipManager:
    """Manages bidirectional relationships in graph"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
    
    def create_bidirectional_relationship(self, node1_id: str, node2_id: str,
                                        relationship_type: str, properties: Dict):
        """Create bidirectional relationship between nodes"""
        with self.graph.driver.session() as session:
            session.write_transaction(
                self._create_bidirectional_tx,
                node1_id, node2_id, relationship_type, properties
            )
    
    def _create_bidirectional_tx(self, tx, node1_id: str, node2_id: str,
                                relationship_type: str, properties: Dict):
        """Create bidirectional relationship transaction"""
        query = f"""
        MATCH (n1 {{id: $node1_id}})
        MATCH (n2 {{id: $node2_id}})
        MERGE (n1)-[r1:{relationship_type}]->(n2)
        MERGE (n2)-[r2:{relationship_type}]->(n1)
        SET r1 += $properties
        SET r2 += $properties
        """
        
        tx.run(query,
            node1_id=node1_id,
            node2_id=node2_id,
            properties=properties
        )
```
### 12.5 Incremental Updates

#### Update Strategy

```python
class IncrementalGraphUpdater:
    """Handles incremental updates to knowledge graph"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
        self.logger = logging.getLogger(__name__)
    
    def update_graph_with_new_paper(self, document: ParsedDocument,
                                   extraction_results: List[ConceptExtractionResult]):
        """
        Add new paper to graph without rebuilding.
        
        Steps:
        1. Create paper node
        2. Create/update concept nodes
        3. Create relationships
        4. Update aggregated metrics
        5. Detect new concept relationships
        """
        try:
            with self.graph.driver.session() as session:
                # Start transaction
                with session.begin_transaction() as tx:
                    # 1. Create paper node
                    paper_id = self._create_paper_node_tx(tx, document)
                    
                    # 2. Create/update concepts
                    concept_ids = self._update_concepts_tx(tx, extraction_results)
                    
                    # 3. Create relationships
                    self._create_paper_relationships_tx(tx, paper_id, extraction_results)
                    
                    # 4. Update metrics
                    self._update_metrics_tx(tx, concept_ids)
                    
                    # 5. Detect new relationships
                    self._detect_new_relationships_tx(tx, concept_ids)
                    
                    # Commit transaction
                    tx.commit()
                    
            self.logger.info(f"Successfully added paper {paper_id} to graph")
            
        except Exception as e:
            self.logger.error(f"Failed to update graph: {e}")
            raise
    
    def _detect_new_relationships_tx(self, tx, concept_ids: List[str]):
        """Detect new concept-concept relationships"""
        query = """
        // Find concepts that co-occur with new concepts
        UNWIND $concept_ids as concept_id
        MATCH (new_concept:Concept {id: concept_id})<-[:MENTIONS]-(p:Paper)
        MATCH (p)-[:MENTIONS]->(other_concept:Concept)
        WHERE other_concept.id <> concept_id
        
        // Calculate co-occurrence strength
        WITH new_concept, other_concept, count(p) as cooccurrence_count
        WHERE cooccurrence_count >= 2
        
        // Create or update relationship
        MERGE (new_concept)-[r:RELATED_TO]-(other_concept)
        ON CREATE SET 
            r.strength = cooccurrence_count * 0.1,
            r.papers_count = cooccurrence_count,
            r.relationship_type = "co_occurrence"
        ON MATCH SET
            r.papers_count = cooccurrence_count,
            r.strength = cooccurrence_count * 0.1
        """
        
        tx.run(query, concept_ids=concept_ids)
    
    def remove_paper_from_graph(self, paper_id: str):
        """Remove paper and update graph accordingly"""
        with self.graph.driver.session() as session:
            session.write_transaction(self._remove_paper_tx, paper_id)
    
    def _remove_paper_tx(self, tx, paper_id: str):
        """Remove paper and clean up orphaned nodes"""
        query = """
        // Find paper and its relationships
        MATCH (p:Paper {id: $paper_id})
        
        // Update concept frequencies
        MATCH (p)-[r:MENTIONS]->(c:Concept)
        SET c.frequency = c.frequency - r.frequency
        
        // Remove paper and all its relationships
        DETACH DELETE p
        
        // Clean up concepts with zero frequency
        MATCH (c:Concept)
        WHERE c.frequency <= 0
        DETACH DELETE c
        """
        
        tx.run(query, paper_id=paper_id)
```

---

### 12.6 Graph Queries

#### Common Query Patterns

```python
class GraphQueryEngine:
    """Executes common graph queries"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
    
    def find_related_papers(self, paper_id: str, max_hops: int = 2) -> List[Dict]:
        """Find papers related through shared concepts"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._find_related_papers_tx, paper_id, max_hops
            )
    
    def _find_related_papers_tx(self, tx, paper_id: str, max_hops: int):
        """Find related papers transaction"""
        query = f"""
        MATCH (source:Paper {{id: $paper_id}})
        MATCH (source)-[:MENTIONS]->(concept:Concept)<-[:MENTIONS]-(related:Paper)
        WHERE related <> source
        
        WITH related, count(concept) as shared_concepts,
             collect(concept.name) as shared_concept_names
        
        RETURN related.id as paper_id,
               related.title as title,
               related.year as year,
               shared_concepts,
               shared_concept_names
        ORDER BY shared_concepts DESC
        LIMIT 20
        """
        
        result = tx.run(query, paper_id=paper_id)
        return [dict(record) for record in result]
    
    def find_concept_clusters(self, min_cluster_size: int = 5) -> List[Dict]:
        """Find clusters of related concepts"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._find_concept_clusters_tx, min_cluster_size
            )
    
    def _find_concept_clusters_tx(self, tx, min_cluster_size: int):
        """Find concept clusters using community detection"""
        query = """
        // Use Louvain algorithm for community detection
        CALL gds.louvain.stream('concept-graph')
        YIELD nodeId, communityId
        
        WITH communityId, collect(gds.util.asNode(nodeId)) as concepts
        WHERE size(concepts) >= $min_size
        
        RETURN communityId,
               [c in concepts | c.name] as concept_names,
               size(concepts) as cluster_size
        ORDER BY cluster_size DESC
        """
        
        result = tx.run(query, min_size=min_cluster_size)
        return [dict(record) for record in result]
    
    def find_influential_concepts(self, limit: int = 20) -> List[Dict]:
        """Find most influential concepts using centrality measures"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._find_influential_concepts_tx, limit
            )
    
    def _find_influential_concepts_tx(self, tx, limit: int):
        """Find influential concepts using PageRank"""
        query = """
        // Calculate PageRank centrality
        CALL gds.pageRank.stream('concept-graph')
        YIELD nodeId, score
        
        WITH gds.util.asNode(nodeId) as concept, score
        
        RETURN concept.name as concept_name,
               concept.type as concept_type,
               concept.paper_count as paper_count,
               score as influence_score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        result = tx.run(query, limit=limit)
        return [dict(record) for record in result]
    
    def find_research_evolution(self, concept_name: str) -> List[Dict]:
        """Track how a concept has evolved over time"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._find_research_evolution_tx, concept_name
            )
    
    def _find_research_evolution_tx(self, tx, concept_name: str):
        """Track concept evolution over time"""
        query = """
        MATCH (c:Concept {normalized_name: $concept_name})<-[:MENTIONS]-(p:Paper)
        
        WITH p.year as year, count(p) as paper_count
        WHERE year IS NOT NULL
        
        RETURN year, paper_count
        ORDER BY year
        """
        
        result = tx.run(query, concept_name=concept_name)
        return [dict(record) for record in result]
    
    def find_author_expertise(self, author_name: str) -> List[Dict]:
        """Find author's areas of expertise"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._find_author_expertise_tx, author_name
            )
    
    def _find_author_expertise_tx(self, tx, author_name: str):
        """Find author expertise areas"""
        query = """
        MATCH (a:Author {normalized_name: $author_name})-[:AUTHORED]->(p:Paper)
        MATCH (p)-[r:MENTIONS]->(c:Concept)
        
        WITH c, sum(r.frequency) as total_mentions, count(p) as paper_count
        
        RETURN c.name as concept,
               c.type as concept_type,
               total_mentions,
               paper_count,
               (total_mentions * paper_count) as expertise_score
        ORDER BY expertise_score DESC
        LIMIT 10
        """
        
        result = tx.run(query, author_name=author_name.lower())
        return [dict(record) for record in result]
```

---

## Graph Queries and Analytics

### Advanced Analytics

#### Research Trend Analysis

```python
class ResearchTrendAnalyzer:
    """Analyzes research trends using graph data"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
    
    def analyze_emerging_topics(self, years_back: int = 3) -> List[Dict]:
        """Identify emerging research topics"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._analyze_emerging_topics_tx, years_back
            )
    
    def _analyze_emerging_topics_tx(self, tx, years_back: int):
        """Find concepts with rapid growth"""
        current_year = 2024  # Would be dynamic in real implementation
        
        query = """
        MATCH (c:Concept)<-[:MENTIONS]-(p:Paper)
        WHERE p.year >= $start_year
        
        WITH c, p.year as year, count(p) as yearly_count
        ORDER BY c.id, year
        
        WITH c, collect({year: year, count: yearly_count}) as yearly_data
        WHERE size(yearly_data) >= 2
        
        // Calculate growth rate
        WITH c, yearly_data,
             yearly_data[0].count as first_year_count,
             yearly_data[-1].count as last_year_count
        
        WHERE first_year_count > 0
        
        WITH c, yearly_data,
             (last_year_count * 1.0 / first_year_count) as growth_rate
        
        WHERE growth_rate > 2.0  // At least 2x growth
        
        RETURN c.name as concept,
               c.type as concept_type,
               growth_rate,
               yearly_data
        ORDER BY growth_rate DESC
        LIMIT 15
        """
        
        result = tx.run(query, start_year=current_year - years_back)
        return [dict(record) for record in result]
    
    def analyze_declining_topics(self, years_back: int = 5) -> List[Dict]:
        """Identify declining research topics"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._analyze_declining_topics_tx, years_back
            )
    
    def _analyze_declining_topics_tx(self, tx, years_back: int):
        """Find concepts with declining interest"""
        current_year = 2024
        
        query = """
        MATCH (c:Concept)<-[:MENTIONS]-(p:Paper)
        WHERE p.year >= $start_year AND c.paper_count >= 10
        
        WITH c, p.year as year, count(p) as yearly_count
        ORDER BY c.id, year
        
        WITH c, collect({year: year, count: yearly_count}) as yearly_data
        WHERE size(yearly_data) >= 3
        
        // Calculate decline rate
        WITH c, yearly_data,
             yearly_data[0].count as first_year_count,
             yearly_data[-1].count as last_year_count
        
        WHERE first_year_count > last_year_count
        
        WITH c, yearly_data,
             (first_year_count * 1.0 / last_year_count) as decline_rate
        
        WHERE decline_rate > 1.5  // At least 50% decline
        
        RETURN c.name as concept,
               c.type as concept_type,
               decline_rate,
               yearly_data
        ORDER BY decline_rate DESC
        LIMIT 15
        """
        
        result = tx.run(query, start_year=current_year - years_back)
        return [dict(record) for record in result]
```

---

#### Collaboration Network Analysis

```python
class CollaborationAnalyzer:
    """Analyzes author collaboration networks"""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder
    
    def find_collaboration_clusters(self) -> List[Dict]:
        """Find clusters of collaborating authors"""
        with self.graph.driver.session() as session:
            return session.read_transaction(self._find_collaboration_clusters_tx)
    
    def _find_collaboration_clusters_tx(self, tx):
        """Find author collaboration clusters"""
        query = """
        // Create collaboration relationships
        MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE a1 <> a2
        
        WITH a1, a2, count(p) as collaborations
        WHERE collaborations >= 2
        
        // Use community detection
        CALL gds.louvain.stream({
            nodeQuery: 'MATCH (a:Author) RETURN id(a) as id',
            relationshipQuery: 'MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author) WHERE a1 <> a2 RETURN id(a1) as source, id(a2) as target, count(p) as weight'
        })
        YIELD nodeId, communityId
        
        WITH communityId, collect(gds.util.asNode(nodeId)) as authors
        WHERE size(authors) >= 3
        
        RETURN communityId,
               [a in authors | a.name] as author_names,
               size(authors) as cluster_size
        ORDER BY cluster_size DESC
        """
        
        result = tx.run(query)
        return [dict(record) for record in result]
    
    def find_key_collaborators(self, author_name: str) -> List[Dict]:
        """Find key collaborators for an author"""
        with self.graph.driver.session() as session:
            return session.read_transaction(
                self._find_key_collaborators_tx, author_name
            )
    
    def _find_key_collaborators_tx(self, tx, author_name: str):
        """Find author's key collaborators"""
        query = """
        MATCH (a:Author {normalized_name: $author_name})-[:AUTHORED]->(p:Paper)
        MATCH (p)<-[:AUTHORED]-(collaborator:Author)
        WHERE collaborator <> a
        
        WITH collaborator, count(p) as joint_papers,
             collect(p.title) as paper_titles
        
        RETURN collaborator.name as collaborator_name,
               joint_papers,
               paper_titles
        ORDER BY joint_papers DESC
        LIMIT 10
        """
        
        result = tx.run(query, author_name=author_name.lower())
        return [dict(record) for record in result]
```

---

## Learning Outcomes

### Skills Learned in Phase 3

**1. Graph Database Design**
- Graph schema design principles
- Node and relationship modeling
- Property graph concepts
- Graph vs relational database trade-offs

**2. Neo4j and Cypher**
- Cypher query language
- Graph database operations (CREATE, MATCH, MERGE)
- Transaction management
- Index and constraint creation

**3. Graph Algorithms**
- Centrality measures (PageRank, betweenness)
- Community detection (Louvain algorithm)
- Path finding algorithms
- Graph traversal patterns

**4. Knowledge Representation**
- Ontology design for research domains
- Concept hierarchies and relationships
- Entity resolution and deduplication
- Incremental knowledge base updates

**5. Graph Analytics**
- Trend analysis using temporal data
- Collaboration network analysis
- Influence and authority metrics
- Research landscape visualization

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: "Neo4j connection failed"**
- **Cause:** Neo4j not running or wrong credentials
- **Solutions:**
  - Start Neo4j service: `neo4j start`
  - Check connection URI and credentials
  - Verify firewall settings (port 7687)

**Issue 2: "Out of memory during graph construction"**
- **Cause:** Large dataset, insufficient heap memory
- **Solutions:**
  - Increase Neo4j heap size in neo4j.conf
  - Process data in smaller batches
  - Use PERIODIC COMMIT for large imports

**Issue 3: "Slow query performance"**
- **Cause:** Missing indexes, inefficient queries
- **Solutions:**
  - Create indexes on frequently queried properties
  - Use EXPLAIN/PROFILE to analyze queries
  - Optimize Cypher queries (avoid Cartesian products)

**Issue 4: "Duplicate nodes created"**
- **Cause:** Inconsistent node identification
- **Solutions:**
  - Use MERGE instead of CREATE
  - Implement proper deduplication logic
  - Create unique constraints

**Issue 5: "Graph becomes disconnected"**
- **Cause:** Missing relationships, data quality issues
- **Solutions:**
  - Implement relationship validation
  - Add bidirectional relationships where appropriate
  - Regular graph integrity checks

---

## Success Criteria

Phase 3 is successful when:

✅ **Graph Schema**
- All node types are properly defined
- Relationships capture domain semantics
- Properties support required queries

✅ **Graph Construction**
- Papers, concepts, and authors are correctly represented
- Relationships are created accurately
- Deduplication works effectively

✅ **Query Performance**
- Common queries execute in < 5 seconds
- Indexes are properly configured
- Memory usage is reasonable

✅ **Graph Analytics**
- Trend analysis identifies meaningful patterns
- Collaboration networks are accurate
- Centrality measures highlight important concepts

✅ **Incremental Updates**
- New papers can be added without rebuilding
- Metrics are updated correctly
- Graph integrity is maintained

---

## Next Steps

After completing Phase 3, you'll have:
- A comprehensive knowledge graph of research literature
- Powerful graph-based queries and analytics
- Understanding of graph database design and operations

**Phase 4** will build on this foundation by:
- Creating vector embeddings for semantic search
- Implementing fast similarity search with ChromaDB
- Enabling hybrid search (graph + vector)
- Supporting complex research queries

---

**Phase 3 demonstrates advanced database design skills and graph analytics capabilities - highly valued for roles in knowledge management, recommendation systems, and data engineering.**