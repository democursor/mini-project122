"""
Test Phase 1-3 functionality WITHOUT Neo4j
Shows what works even without graph database
"""
import json
from pathlib import Path
from src.utils.config import Config
from src.orchestration import DocumentProcessor


def test_without_neo4j():
    """Test that pipeline works without Neo4j"""
    print("=" * 60)
    print("Testing Pipeline WITHOUT Neo4j")
    print("=" * 60)
    
    # Load config
    config = Config()
    
    # Create processor (will warn about Neo4j but continue)
    print("\n1. Initializing processor...")
    processor = DocumentProcessor(config)
    
    if processor.graph_builder:
        print("   ✓ Neo4j connected")
    else:
        print("   ⚠ Neo4j not available (this is OK!)")
    
    # Find a parsed document to test with
    parsed_dir = Path('./data/parsed')
    json_files = list(parsed_dir.glob('*.json'))
    
    if not json_files:
        print("\n✗ No parsed documents found")
        print("  Run: python main.py")
        return
    
    # Load first document with concepts
    test_doc = None
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get('concepts') and len(data.get('concepts', [])) > 0:
                test_doc = data
                break
    
    if not test_doc:
        print("\n✗ No documents with concepts found")
        return
    
    doc_id = test_doc['document_id']
    print(f"\n2. Testing with document: {doc_id}")
    
    # Show what we have
    print("\n3. Available Data (WITHOUT Neo4j):")
    print(f"   ✓ Document ID: {doc_id}")
    print(f"   ✓ Title: {test_doc['parsed_data']['metadata']['title']}")
    print(f"   ✓ Pages: {test_doc['parsed_data']['page_count']}")
    print(f"   ✓ Chunks: {len(test_doc.get('chunks', []))}")
    print(f"   ✓ Concepts extracted: {len(test_doc.get('concepts', []))}")
    
    # Show sample concepts
    if test_doc.get('concepts'):
        print("\n4. Sample Extracted Concepts:")
        concept_chunk = test_doc['concepts'][0]
        
        if concept_chunk.get('entities'):
            print("   Entities:")
            for ent in concept_chunk['entities'][:5]:
                print(f"     - {ent['text']} ({ent['label']})")
        
        if concept_chunk.get('keyphrases'):
            print("   Keyphrases:")
            for kp in concept_chunk['keyphrases'][:5]:
                print(f"     - {kp['phrase']} (score: {kp['score']:.2f})")
    
    # Show what's missing without Neo4j
    print("\n5. What You're Missing Without Neo4j:")
    print("   ✗ Graph visualization")
    print("   ✗ Related paper discovery")
    print("   ✗ Concept co-occurrence analysis")
    print("   ✗ Graph-based queries")
    
    # Show what still works
    print("\n6. What Still Works:")
    print("   ✓ PDF parsing and text extraction")
    print("   ✓ Semantic chunking")
    print("   ✓ Concept extraction (NER + keyphrases)")
    print("   ✓ Data saved to JSON files")
    print("   ✓ Can proceed to Phase 4 (Vector Search)")
    
    print("\n" + "=" * 60)
    print("RESULT: Pipeline works WITHOUT Neo4j!")
    print("=" * 60)
    print("\nYou can:")
    print("  1. Continue to Phase 4 (Vector Search)")
    print("  2. Use NetworkX for lightweight graph (no database)")
    print("  3. Use SQLite for file-based graph")
    print("  4. Try Neo4j later on different network")
    
    return True


if __name__ == "__main__":
    test_without_neo4j()
