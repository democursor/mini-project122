"""
Quick test for agentic RAG pipeline
"""
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_classifier():
    """Test query classifier"""
    from src.rag.query_classifier import QueryClassifier
    
    classifier = QueryClassifier()
    
    # Test simple query
    simple_q = "What is COVID-19?"
    intent = classifier.classify(simple_q)
    print(f"Query: {simple_q}")
    print(f"Intent: {intent}")
    print()
    
    # Test complex query
    complex_q = "Give me a detailed report of COVID disease including symptoms, transmission, and treatment"
    intent = classifier.classify(complex_q)
    print(f"Query: {complex_q}")
    print(f"Intent: {intent}")
    print()
    
    # Test comparative query
    comp_q = "Compare COVID-19 and influenza"
    intent = classifier.classify(comp_q)
    print(f"Query: {comp_q}")
    print(f"Intent: {intent}")
    print()

def test_imports():
    """Test that all imports work"""
    try:
        from src.rag.query_classifier import QueryClassifier, QueryIntent
        from src.rag.query_decomposer import QueryDecomposer
        from src.rag.agentic_pipeline import AgenticRAGPipeline
        from src.services.chat_service import ChatService
        from src.utils.deduplication import deduplicate_chunks, deduplicate_sources
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Agentic RAG Pipeline")
    print("=" * 60)
    print()
    
    # Test imports
    print("1. Testing imports...")
    if not test_imports():
        sys.exit(1)
    print()
    
    # Test classifier
    print("2. Testing query classifier...")
    test_query_classifier()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
