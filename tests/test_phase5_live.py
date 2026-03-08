"""Quick test of Phase 5 with real Google Gemini API."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.vector.store import VectorStore
from src.vector.embedder import EmbeddingGenerator, EmbeddingConfig
from src.vector.search import SemanticSearchEngine, QueryProcessor
from src.rag.retriever import RAGRetriever
from src.rag.llm_client import LLMClient
from src.rag.assistant import ResearchAssistant


def main():
    """Test Phase 5 with a real question."""
    print("\n" + "="*80)
    print("PHASE 5 LIVE TEST: RAG with Google Gemini")
    print("="*80)
    
    # Load config
    config = Config().config
    setup_logging(
        log_level=config["logging"]["level"],
        log_file=config["logging"]["file"]
    )
    
    # Check API key based on provider
    provider = config["rag"]["llm_provider"]
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not set!")
            return
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set!")
            return
    
    print(f"✓ Provider: {provider}")
    print(f"✓ Model: {config['rag']['llm_model']}")
    print(f"✓ API Key found: {api_key[:20]}...")
    
    try:
        print("\n1. Initializing components...")
        
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory=config["vector"]["persist_directory"]
        )
        print("  ✓ Vector store loaded")
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(
            config=EmbeddingConfig(
                model_name=config["vector"]["embedding_model"],
                batch_size=config["vector"]["batch_size"]
            )
        )
        print("  ✓ Embedding generator loaded")
        
        # Initialize search engine
        query_processor = QueryProcessor(embedding_generator)
        search_engine = SemanticSearchEngine(vector_store, query_processor)
        print("  ✓ Search engine initialized")
        
        # Initialize RAG components
        retriever = RAGRetriever(
            search_engine=search_engine,
            max_context_tokens=config["rag"]["max_context_tokens"]
        )
        print("  ✓ RAG retriever initialized")
        
        llm_client = LLMClient(
            provider=config["rag"]["llm_provider"],
            model=config["rag"]["llm_model"],
            api_key=api_key
        )
        print("  ✓ LLM client initialized")
        
        assistant = ResearchAssistant(
            retriever=retriever,
            llm_client=llm_client
        )
        print("  ✓ Research assistant ready")
        
        # Test with a simple question
        print("\n2. Testing with a research question...")
        question = "What is this paper about?"
        print(f"   Question: {question}")
        
        print("\n3. Generating response (this may take 5-10 seconds)...")
        response = assistant.ask_question(question, top_k=3)
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        print(f"\n🤖 Answer:\n{response.answer}\n")
        
        print(f"📚 Sources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            metadata = source.get("metadata", {})
            print(f"  [{i}] Document: {metadata.get('document_id', 'Unknown')[:20]}...")
            print(f"      Score: {source.get('final_score', 0):.3f}")
        
        print(f"\n📝 Citations:")
        citations = response.citations
        print(f"  Total: {citations['total_citations']}")
        print(f"  Accuracy: {citations['citation_accuracy']:.1%}")
        
        print(f"\n⏱️  Performance:")
        stats = response.retrieval_stats
        print(f"  Retrieval time: {stats['retrieval_time']:.2f}s")
        print(f"  Chunks retrieved: {stats['chunks_retrieved']}/{stats['total_found']}")
        
        print("\n" + "="*80)
        print("✅ PHASE 5 LIVE TEST PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
