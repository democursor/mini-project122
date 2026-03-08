"""
Test script for Phase 3: Knowledge Graph Construction
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


def test_neo4j_connection(config):
    """Test Neo4j connection"""
    print("\n=== Testing Neo4j Connection ===")
    try:
        builder = KnowledgeGraphBuilder(
            uri=config.get('neo4j.uri'),
            user=config.get('neo4j.user'),
            password=config.get('neo4j.password'),
            database=config.get('neo4j.database', 'neo4j')
        )
        print("✓ Connected to Neo4j successfully")
        builder.close()
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        print("\nMake sure Neo4j is running:")
        print("  Docker: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
        print("  Or install Neo4j Desktop from https://neo4j.com/download/")
        return False


def test_graph_construction(config):
    """Test building graph from parsed data"""
    print("\n=== Testing Graph Construction ===")
    
    builder = KnowledgeGraphBuilder(
        uri=config.get('neo4j.uri'),
        user=config.get('neo4j.user'),
        password=config.get('neo4j.password'),
        database=config.get('neo4j.database', 'neo4j')
    )
    
    try:
        # Find parsed JSON files
        parsed_dir = Path('./data/parsed')
        json_files = list(parsed_dir.glob('*.json'))
        
        if not json_files:
            print("✗ No parsed JSON files found in data/parsed/")
            return False
        
        print(f"Found {len(json_files)} parsed documents")
        
        # Process first document
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doc_id = data.get('document_id')
        print(f"\nProcessing document: {doc_id}")
        
        # Build graph
        success = builder.build_from_parsed_data(data)
        
        if success:
            print("✓ Graph constructed successfully")
        else:
            print("✗ Graph construction failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during graph construction: {e}")
        return False
    finally:
        builder.close()


def test_graph_queries(config):
    """Test graph queries"""
    print("\n=== Testing Graph Queries ===")
    
    builder = KnowledgeGraphBuilder(
        uri=config.get('neo4j.uri'),
        user=config.get('neo4j.user'),
        password=config.get('neo4j.password'),
        database=config.get('neo4j.database', 'neo4j')
    )
    
    query_engine = GraphQueryEngine(builder.driver, builder.database)
    
    try:
        # Get statistics
        print("\n1. Graph Statistics:")
        stats = query_engine.get_graph_statistics()
        print(f"   Papers: {stats['papers']}")
        print(f"   Concepts: {stats['concepts']}")
        print(f"   Mentions: {stats['mentions']}")
        print(f"   Relationships: {stats['relationships']}")
        
        # Get all papers
        print("\n2. All Papers:")
        papers = query_engine.get_all_papers(limit=5)
        for paper in papers:
            print(f"   - {paper['title'][:60]}... ({paper['year']})")
        
        # Get top concepts
        print("\n3. Top Concepts:")
        concepts = query_engine.get_all_concepts(limit=10)
        for concept in concepts:
            print(f"   - {concept['name']} ({concept['type']}): {concept['frequency']} mentions")
        
        # Find related papers (if we have papers)
        if papers:
            paper_id = papers[0]['id']
            print(f"\n4. Papers Related to '{papers[0]['title'][:40]}...':")
            related = query_engine.find_related_papers(paper_id, limit=5)
            if related:
                for rel in related:
                    print(f"   - {rel['title'][:60]}... (shared: {rel['shared_concepts']} concepts)")
            else:
                print("   No related papers found")
        
        # Find papers by concept (if we have concepts)
        if concepts:
            concept_name = concepts[0]['name']
            print(f"\n5. Papers Mentioning '{concept_name}':")
            papers_with_concept = query_engine.find_papers_by_concept(concept_name)
            for paper in papers_with_concept[:5]:
                print(f"   - {paper['title'][:60]}... (mentions: {paper['mentions']})")
        
        print("\n✓ All queries executed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error during queries: {e}")
        return False
    finally:
        builder.close()


def main():
    """Run all Phase 3 tests"""
    print("=" * 60)
    print("PHASE 3: Knowledge Graph Construction - Test Suite")
    print("=" * 60)
    
    # Load config
    config = Config()
    
    # Check if Neo4j config exists
    if not config.get('neo4j'):
        print("\n✗ Neo4j configuration not found in config/default.yaml")
        print("Please add Neo4j configuration and ensure Neo4j is running")
        return
    
    # Run tests
    tests = [
        ("Neo4j Connection", lambda: test_neo4j_connection(config)),
        ("Graph Construction", lambda: test_graph_construction(config)),
        ("Graph Queries", lambda: test_graph_queries(config))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Phase 3 implementation successful!")
        print("\nNext steps:")
        print("  1. Open Neo4j Browser: http://localhost:7474")
        print("  2. Run Cypher queries to explore the graph")
        print("  3. Visualize relationships between papers and concepts")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
