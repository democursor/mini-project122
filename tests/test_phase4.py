"""Test Phase 4: Vector Storage and Semantic Search"""
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.vector import EmbeddingGenerator, EmbeddingConfig, VectorStore
from src.vector.search import SemanticSearchEngine, QueryProcessor
from src.vector.models import ChunkEmbedding


def test_embedding_generation():
    """Test 1: Embedding Generation"""
    print("\n=== Test 1: Embedding Generation ===")
    
    config = EmbeddingConfig()
    generator = EmbeddingGenerator(config)
    
    # Test single embedding
    text = "Machine learning is a subset of artificial intelligence"
    embedding = generator.generate_single_embedding(text)
    
    print(f"✓ Generated single embedding")
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  Embedding shape: {embedding.shape}")
    
    # Test batch embeddings
    texts = [
        "Deep learning uses neural networks",
        "Natural language processing analyzes text",
        "Computer vision processes images"
    ]
    embeddings = generator.generate_embeddings(texts)
    
    print(f"✓ Generated batch embeddings")
    print(f"  Number of embeddings: {len(embeddings)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    return True


def test_vector_storage():
    """Test 2: Vector Storage with ChromaDB"""
    print("\n=== Test 2: Vector Storage ===")
    
    vector_store = VectorStore("./data/chroma_test")
    
    # Create test embeddings
    config = EmbeddingConfig()
    generator = EmbeddingGenerator(config)
    
    test_chunks = [
        {
            "chunk_id": "test_chunk_1",
            "document_id": "test_doc_1",
            "text": "Machine learning algorithms learn from data",
            "section_heading": "Introduction"
        },
        {
            "chunk_id": "test_chunk_2",
            "document_id": "test_doc_1",
            "text": "Neural networks are inspired by biological neurons",
            "section_heading": "Background"
        },
        {
            "chunk_id": "test_chunk_3",
            "document_id": "test_doc_2",
            "text": "Natural language processing enables computers to understand text",
            "section_heading": "Abstract"
        }
    ]
    
    # Generate embeddings
    texts = [chunk["text"] for chunk in test_chunks]
    embeddings = generator.generate_embeddings(texts)
    
    # Create ChunkEmbedding objects
    chunk_embeddings = []
    for chunk, embedding in zip(test_chunks, embeddings):
        chunk_embedding = ChunkEmbedding(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            text=chunk["text"],
            embedding=embedding,
            embedding_model=config.model_name,
            section_heading=chunk["section_heading"]
        )
        chunk_embeddings.append(chunk_embedding)
    
    # Store embeddings
    vector_store.add_embeddings(chunk_embeddings)
    
    print(f"✓ Stored {len(chunk_embeddings)} embeddings")
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"✓ Collection stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Collection name: {stats['collection_name']}")
    
    return True


def test_semantic_search():
    """Test 3: Semantic Search"""
    print("\n=== Test 3: Semantic Search ===")
    
    # Initialize components
    config = EmbeddingConfig()
    generator = EmbeddingGenerator(config)
    vector_store = VectorStore("./data/chroma_test")
    query_processor = QueryProcessor(generator)
    search_engine = SemanticSearchEngine(vector_store, query_processor)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about natural language processing"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_engine.search(query, top_k=2)
        
        print(f"✓ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Similarity: {result['similarity_score']:.3f}")
            print(f"    Final Score: {result['final_score']:.3f}")
            print(f"    Text: {result['text'][:80]}...")
            print(f"    Section: {result['metadata'].get('section_heading', 'N/A')}")
    
    return True


def test_query_processing():
    """Test 4: Query Processing"""
    print("\n=== Test 4: Query Processing ===")
    
    config = EmbeddingConfig()
    generator = EmbeddingGenerator(config)
    processor = QueryProcessor(generator)
    
    # Test query cleaning
    raw_query = "  What   is   machine learning???  "
    processed = processor._clean_query(raw_query)
    print(f"✓ Query cleaning:")
    print(f"  Raw: '{raw_query}'")
    print(f"  Processed: '{processed}'")
    
    # Test filter extraction
    query_with_year = "papers about transformers after 2020"
    filters = processor._extract_filters(query_with_year)
    print(f"\n✓ Filter extraction:")
    print(f"  Query: '{query_with_year}'")
    print(f"  Extracted filters: {filters}")
    
    # Test full query processing
    search_query = processor.process_query("machine learning algorithms", top_k=5)
    print(f"\n✓ Full query processing:")
    print(f"  Original: '{search_query.original_query}'")
    print(f"  Processed: '{search_query.processed_query}'")
    print(f"  Embedding shape: {search_query.embedding.shape}")
    print(f"  Top K: {search_query.top_k}")
    
    return True


def test_integration_with_workflow():
    """Test 5: Integration with Existing Workflow"""
    print("\n=== Test 5: Workflow Integration ===")
    
    from src.orchestration import DocumentProcessor
    
    config = Config()
    processor = DocumentProcessor(config)
    
    # Check if vector store is initialized
    print(f"✓ Vector store initialized: {processor.vector_store is not None}")
    print(f"✓ Embedding generator initialized: {processor.embedding_generator is not None}")
    
    # Check vector store stats
    stats = processor.vector_store.get_collection_stats()
    print(f"✓ Vector store stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    
    return True


def main():
    """Run all Phase 4 tests"""
    # Setup logging
    setup_logging(log_level='INFO', log_file='./data/logs/test_phase4.log')
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("PHASE 4 TESTING: Vector Storage and Semantic Search")
    print("=" * 60)
    
    tests = [
        ("Embedding Generation", test_embedding_generation),
        ("Vector Storage", test_vector_storage),
        ("Semantic Search", test_semantic_search),
        ("Query Processing", test_query_processing),
        ("Workflow Integration", test_integration_with_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name}: FAILED")
            print(f"  Error: {e}")
            logger.error(f"Test failed: {test_name}", exc_info=True)
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All Phase 4 tests passed!")
        print("\nPhase 4 Components:")
        print("  ✓ Embedding generation (sentence-transformers)")
        print("  ✓ Vector storage (ChromaDB)")
        print("  ✓ Semantic search engine")
        print("  ✓ Query processing and filtering")
        print("  ✓ Integration with workflow")
        print("\nNext: Test with real documents using main.py")
    else:
        print(f"\n✗ {failed} test(s) failed. Please review errors above.")


if __name__ == "__main__":
    main()
