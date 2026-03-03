"""Interactive chat interface for AI Research Assistant."""

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


def print_separator():
    """Print separator line."""
    print("\n" + "="*80 + "\n")


def print_sources(sources):
    """Print source documents."""
    print("\n📚 Sources:")
    for i, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("title", "Unknown")
        authors = metadata.get("authors", ["Unknown"])
        if isinstance(authors, list):
            authors_str = ", ".join(authors[:2])
            if len(authors) > 2:
                authors_str += " et al."
        else:
            authors_str = str(authors)
        
        print(f"  [{i}] {title}")
        print(f"      Authors: {authors_str}")
        print(f"      Score: {source.get('score', 0):.3f}")


def print_citations(citations):
    """Print citation analysis."""
    if citations["total_citations"] == 0:
        print("\n📝 No citations found in response")
        return
    
    print(f"\n📝 Citations: {citations['total_citations']} total, "
          f"Accuracy: {citations['citation_accuracy']:.1%}")
    
    for i, citation in enumerate(citations["citations"][:5], 1):
        status = "✓" if citation["valid"] else "✗"
        print(f"  {status} {citation['citation']}")


def main():
    """Run interactive chat assistant."""
    print("🤖 AI Research Assistant")
    print_separator()
    
    # Load configuration
    config = Config().config
    setup_logging(
        log_level=config["logging"]["level"],
        log_file=config["logging"]["file"]
    )
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") if config["rag"]["llm_provider"] == "openai" else os.getenv("GOOGLE_API_KEY")
    if not api_key and config["rag"]["llm_provider"] in ["openai", "google"]:
        print(f"⚠️  Warning: API key not found for {config['rag']['llm_provider']}")
        if config["rag"]["llm_provider"] == "openai":
            print("   Set it with: set OPENAI_API_KEY=your-key-here")
        else:
            print("   Set it with: set GOOGLE_API_KEY=your-key-here")
        print("   Or switch to Ollama in config/default.yaml")
        print_separator()
        return
    
    print("Initializing components...")
    
    try:
        # Initialize vector store and search
        vector_store = VectorStore(
            persist_directory=config["vector"]["persist_directory"]
        )
        
        embedding_generator = EmbeddingGenerator(
            config=EmbeddingConfig(
                model_name=config["vector"]["embedding_model"],
                batch_size=config["vector"]["batch_size"]
            )
        )
        
        query_processor = QueryProcessor(embedding_generator)
        search_engine = SemanticSearchEngine(vector_store, query_processor)
        
        # Initialize RAG components
        retriever = RAGRetriever(
            search_engine=search_engine,
            max_context_tokens=config["rag"]["max_context_tokens"]
        )
        
        llm_client = LLMClient(
            provider=config["rag"]["llm_provider"],
            model=config["rag"]["llm_model"],
            api_key=api_key
        )
        
        assistant = ResearchAssistant(
            retriever=retriever,
            llm_client=llm_client
        )
        
        print("✓ Assistant ready!")
        print_separator()
        
        # Interactive loop
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - 'clear' to clear conversation history")
        print("  - 'history' to show conversation length")
        print("  - 'quit' or 'exit' to quit")
        print_separator()
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if question.lower() == 'clear':
                    assistant.clear_conversation()
                    print("✓ Conversation history cleared")
                    continue
                
                if question.lower() == 'history':
                    length = assistant.get_conversation_length()
                    print(f"📊 Conversation length: {length} turns")
                    continue
                
                # Process question
                print("\n🔍 Searching and generating response...")
                
                response = assistant.ask_question(
                    question,
                    top_k=config["rag"]["top_k_retrieval"],
                    use_conversation_context=(assistant.get_conversation_length() > 0)
                )
                
                # Print answer
                print_separator()
                print("🤖 Assistant:")
                print(response.answer)
                
                # Print sources
                print_sources(response.sources)
                
                # Print citations
                print_citations(response.citations)
                
                # Print stats
                stats = response.retrieval_stats
                print(f"\n⏱️  Retrieval: {stats['retrieval_time']:.2f}s, "
                      f"Chunks: {stats['chunks_retrieved']}/{stats['total_found']}")
                
                print_separator()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit")
    
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("\nMake sure:")
        print("  1. Vector store is populated (run main.py first)")
        print("  2. OPENAI_API_KEY is set (or use Ollama)")
        print("  3. All dependencies are installed")


if __name__ == "__main__":
    main()
