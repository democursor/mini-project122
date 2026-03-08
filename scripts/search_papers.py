"""
Semantic search interface for research papers
"""
from src.utils.config import Config
from src.vector import EmbeddingGenerator, EmbeddingConfig, VectorStore
from src.vector.search import SemanticSearchEngine, QueryProcessor


def search_papers():
    """Interactive semantic search interface"""
    print("=" * 60)
    print("SEMANTIC SEARCH - Research Papers")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    embedding_generator = EmbeddingGenerator(EmbeddingConfig())
    vector_store = VectorStore(config.get('vector.persist_directory', './data/chroma'))
    query_processor = QueryProcessor(embedding_generator)
    search_engine = SemanticSearchEngine(vector_store, query_processor)
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"\n📊 Vector Store Statistics:")
    print(f"   Total chunks indexed: {stats['total_chunks']}")
    print(f"   Collection: {stats['collection_name']}")
    
    if stats['total_chunks'] == 0:
        print("\n⚠️  No documents indexed yet!")
        print("   Run 'python main.py' to index documents first.")
        return
    
    print("\n" + "=" * 60)
    print("Enter your search query (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        query = input("\n🔍 Search: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        try:
            # Perform search
            results = search_engine.search(query, top_k=5)
            
            if not results:
                print("\n❌ No results found.")
                continue
            
            print(f"\n✓ Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{'─' * 60}")
                print(f"Result {i}:")
                print(f"  📄 Document: {result['document_id']}")
                print(f"  📊 Similarity: {result['similarity_score']:.3f}")
                print(f"  ⭐ Final Score: {result['final_score']:.3f}")
                
                if result['metadata'].get('section_heading'):
                    print(f"  📑 Section: {result['metadata']['section_heading']}")
                
                print(f"\n  📝 Text:")
                text = result['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"     {text}")
                
                # Show ranking factors
                factors = result.get('ranking_factors', {})
                if factors:
                    print(f"\n  🎯 Ranking Factors:")
                    print(f"     Similarity: {factors.get('similarity', 0):.3f}")
                    print(f"     Term Frequency: {factors.get('term_frequency', 0):.3f}")
                    print(f"     Section Boost: {factors.get('section', 0):.3f}")
                
                print()
        
        except Exception as e:
            print(f"\n❌ Search error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    search_papers()
