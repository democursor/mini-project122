"""
Build knowledge graph from existing parsed data
"""
import json
import logging
from pathlib import Path
from src.utils.config import Config
from src.graph import KnowledgeGraphBuilder, GraphQueryEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_graph_from_parsed_data():
    """Build knowledge graph from all parsed JSON files"""
    print("=" * 60)
    print("Building Knowledge Graph from Parsed Data")
    print("=" * 60)
    
    # Load config
    config = Config()
    
    # Check Neo4j config
    if not config.get('neo4j'):
        print("\n✗ Neo4j configuration not found")
        return
    
    # Connect to Neo4j
    try:
        builder = KnowledgeGraphBuilder(
            uri=config.get('neo4j.uri'),
            user=config.get('neo4j.user'),
            password=config.get('neo4j.password'),
            database=config.get('neo4j.database', 'neo4j')
        )
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
        return
    
    # Find parsed JSON files
    parsed_dir = Path('./data/parsed')
    json_files = list(parsed_dir.glob('*.json'))
    
    if not json_files:
        print("\n✗ No parsed JSON files found in data/parsed/")
        builder.close()
        return
    
    print(f"\nFound {len(json_files)} parsed documents")
    
    # Process each document
    successful = 0
    failed = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_id = data.get('document_id')
            
            # Skip if no concepts
            if not data.get('concepts') or len(data.get('concepts', [])) == 0:
                print(f"⊘ Skipping {doc_id} (no concepts)")
                continue
            
            # Build graph
            success = builder.build_from_parsed_data(data)
            
            if success:
                print(f"✓ Processed {doc_id}")
                successful += 1
            else:
                print(f"✗ Failed {doc_id}")
                failed += 1
                
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
            failed += 1
    
    # Show statistics
    print("\n" + "=" * 60)
    print("Graph Construction Complete")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {len(json_files) - successful - failed}")
    
    # Query statistics
    query_engine = GraphQueryEngine(builder.driver, builder.database)
    stats = query_engine.get_graph_statistics()
    
    print("\nGraph Statistics:")
    print(f"  Papers: {stats['papers']}")
    print(f"  Concepts: {stats['concepts']}")
    print(f"  Mentions: {stats['mentions']}")
    print(f"  Relationships: {stats['relationships']}")
    
    print("\nTop 10 Concepts:")
    concepts = query_engine.get_all_concepts(limit=10)
    for i, concept in enumerate(concepts, 1):
        print(f"  {i}. {concept['name']} ({concept['type']}): {concept['frequency']} mentions")
    
    print("\n✓ Knowledge graph built successfully!")
    print("\nOpen Neo4j Browser to explore: http://localhost:7474")
    print("\nExample Cypher queries:")
    print("  MATCH (p:Paper) RETURN p LIMIT 10")
    print("  MATCH (c:Concept) RETURN c.name, c.frequency ORDER BY c.frequency DESC LIMIT 20")
    print("  MATCH (p:Paper)-[:MENTIONS]->(c:Concept) RETURN p.title, c.name LIMIT 50")
    
    builder.close()


if __name__ == "__main__":
    build_graph_from_parsed_data()
