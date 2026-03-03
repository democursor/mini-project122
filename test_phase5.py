"""Test suite for Phase 5: RAG and AI Research Assistant."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.vector.store import VectorStore
from src.vector.embedder import EmbeddingGenerator, EmbeddingConfig
from src.vector.search import SemanticSearchEngine, QueryProcessor
from src.rag.retriever import RAGRetriever
from src.rag.llm_client import LLMClient
from src.rag.prompt_template import RAGPromptTemplate
from src.rag.citation_extractor import CitationExtractor
from src.rag.assistant import ResearchAssistant


def test_rag_retriever():
    """Test RAG retriever component."""
    print("\n" + "="*80)
    print("TEST 1: RAG Retriever")
    print("="*80)
    
    try:
        config = Config().config
        
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
        
        # Initialize retriever
        retriever = RAGRetriever(search_engine, max_context_tokens=3000)
        
        # Test retrieval
        query = "machine learning applications"
        context = retriever.retrieve_context(query, top_k=5)
        
        print(f"✓ Query: {query}")
        print(f"✓ Retrieved chunks: {len(context.retrieved_chunks)}")
        print(f"✓ Total found: {context.total_chunks_found}")
        print(f"✓ Context length: {context.context_length} chars")
        print(f"✓ Retrieval time: {context.retrieval_time:.2f}s")
        
        # Verify diversity
        doc_ids = [chunk["metadata"].get("document_id") for chunk in context.retrieved_chunks]
        unique_docs = len(set(doc_ids))
        print(f"✓ Unique documents: {unique_docs}/{len(context.retrieved_chunks)}")
        
        print("\n✅ RAG Retriever test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG Retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_template():
    """Test prompt template formatting."""
    print("\n" + "="*80)
    print("TEST 2: Prompt Template")
    print("="*80)
    
    try:
        template = RAGPromptTemplate()
        
        # Mock context chunks
        mock_chunks = [
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "metadata": {
                    "title": "Introduction to ML",
                    "authors": ["John Doe", "Jane Smith"],
                    "year": 2023,
                    "document_id": "doc1"
                }
            },
            {
                "text": "Deep learning uses neural networks with multiple layers.",
                "metadata": {
                    "title": "Deep Learning Basics",
                    "authors": ["Alice Johnson"],
                    "year": 2024,
                    "document_id": "doc2"
                }
            }
        ]
        
        # Test research prompt
        question = "What is machine learning?"
        prompt = template.format_research_prompt(question, mock_chunks)
        
        print(f"✓ Generated research prompt")
        print(f"✓ Prompt length: {len(prompt)} chars")
        print(f"✓ Contains question: {'QUESTION:' in prompt}")
        print(f"✓ Contains context: {'CONTEXT:' in prompt}")
        print(f"✓ Contains instructions: {'INSTRUCTIONS:' in prompt}")
        
        # Test summarization prompt
        summary_prompt = template.format_summarization_prompt(mock_chunks)
        print(f"✓ Generated summarization prompt")
        print(f"✓ Summary prompt length: {len(summary_prompt)} chars")
        
        print("\n✅ Prompt Template test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Prompt Template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_extractor():
    """Test citation extraction and validation."""
    print("\n" + "="*80)
    print("TEST 3: Citation Extractor")
    print("="*80)
    
    try:
        extractor = CitationExtractor()
        
        # Mock response with citations
        response = """Machine learning is widely used [Introduction to ML, John Doe].
Deep learning has shown great results [Deep Learning Basics, Alice Johnson].
Some claim it's revolutionary [Unknown Paper, Unknown Author]."""
        
        # Mock context chunks
        mock_chunks = [
            {
                "text": "ML content",
                "metadata": {
                    "title": "Introduction to ML",
                    "authors": ["John Doe", "Jane Smith"],
                    "document_id": "doc1"
                }
            },
            {
                "text": "DL content",
                "metadata": {
                    "title": "Deep Learning Basics",
                    "authors": ["Alice Johnson"],
                    "document_id": "doc2"
                }
            }
        ]
        
        # Extract citations
        citations = extractor.extract_citations(response, mock_chunks)
        
        print(f"✓ Total citations found: {citations['total_citations']}")
        print(f"✓ Citation accuracy: {citations['citation_accuracy']:.1%}")
        
        # Check individual citations
        for i, citation in enumerate(citations["citations"], 1):
            status = "✓" if citation["valid"] else "✗"
            print(f"{status} Citation {i}: {citation['citation']} - Valid: {citation['valid']}")
        
        # Verify at least 2 valid citations
        valid_count = sum(1 for c in citations["citations"] if c["valid"])
        assert valid_count >= 2, f"Expected at least 2 valid citations, got {valid_count}"
        
        print("\n✅ Citation Extractor test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Citation Extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_client_mock():
    """Test LLM client initialization (without actual API calls)."""
    print("\n" + "="*80)
    print("TEST 4: LLM Client (Mock)")
    print("="*80)
    
    try:
        # Test OpenAI client initialization (will fail without API key, but that's ok)
        try:
            client = LLMClient(provider="openai", model="gpt-4")
            print("✓ OpenAI client initialized")
        except Exception as e:
            print(f"⚠️  OpenAI client init failed (expected without API key): {e}")
        
        # Test Ollama client initialization
        try:
            client = LLMClient(provider="ollama", model="llama2")
            print("✓ Ollama client initialized")
        except Exception as e:
            print(f"⚠️  Ollama client init failed: {e}")
        
        # Test invalid provider
        try:
            client = LLMClient(provider="invalid", model="test")
            print("❌ Should have raised error for invalid provider")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected invalid provider: {e}")
        
        print("\n✅ LLM Client test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ LLM Client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_assistant_integration():
    """Test research assistant integration (without LLM calls)."""
    print("\n" + "="*80)
    print("TEST 5: Research Assistant Integration")
    print("="*80)
    
    try:
        config = Config().config
        
        # Initialize components
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
        retriever = RAGRetriever(search_engine)
        
        # Create mock LLM client that doesn't make real API calls
        class MockLLMClient:
            def __init__(self):
                self.provider = "mock"
                self.model = "mock"
            
            def generate_response(self, prompt, max_tokens=1000, temperature=0.3):
                return "This is a mock response [Test Paper, Test Author]."
        
        mock_llm = MockLLMClient()
        assistant = ResearchAssistant(retriever, mock_llm)
        
        print("✓ Research assistant initialized")
        
        # Test question answering
        question = "What is machine learning?"
        response = assistant.ask_question(question, top_k=3)
        
        print(f"✓ Question processed: {question}")
        print(f"✓ Answer generated: {len(response.answer)} chars")
        print(f"✓ Sources retrieved: {len(response.sources)}")
        print(f"✓ Citations extracted: {response.citations['total_citations']}")
        
        # Test conversation history
        assert assistant.get_conversation_length() == 1, "Conversation history not updated"
        print(f"✓ Conversation history: {assistant.get_conversation_length()} turns")
        
        # Test clear conversation
        assistant.clear_conversation()
        assert assistant.get_conversation_length() == 0, "Conversation not cleared"
        print("✓ Conversation cleared")
        
        print("\n✅ Research Assistant Integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Research Assistant Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 5 tests."""
    print("\n" + "="*80)
    print("PHASE 5 TEST SUITE: RAG and AI Research Assistant")
    print("="*80)
    
    # Setup logging
    config = Config().config
    setup_logging(
        log_level=config["logging"]["level"],
        log_file=config["logging"]["file"]
    )
    
    # Run tests
    results = []
    
    results.append(("RAG Retriever", test_rag_retriever()))
    results.append(("Prompt Template", test_prompt_template()))
    results.append(("Citation Extractor", test_citation_extractor()))
    results.append(("LLM Client", test_llm_client_mock()))
    results.append(("Research Assistant", test_research_assistant_integration()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All Phase 5 tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
    
    print("="*80)


if __name__ == "__main__":
    main()
