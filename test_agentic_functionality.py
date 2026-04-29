"""
Test agentic RAG functionality with real queries
"""
import asyncio
import logging
from src.services.chat_service import ChatService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_simple_query():
    """Test with a simple query"""
    print("\n" + "="*80)
    print("TEST 1: Simple Query (Standard RAG)")
    print("="*80)
    
    service = ChatService()
    
    question = "What is COVID-19?"
    print(f"\nQuestion: {question}")
    print("\nProcessing with use_agentic=False...")
    
    response = await service.answer_question(question, use_agentic=False)
    
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources: {response.sources_count}")
    print(f"Citations: {len(response.citations)}")
    
    if response.citations:
        print("\nTop 3 Citations:")
        for i, citation in enumerate(response.citations[:3], 1):
            print(f"{i}. {citation.title}")
            print(f"   Excerpt: {citation.excerpt[:100]}...")

async def test_complex_query():
    """Test with a complex query using agentic pipeline"""
    print("\n" + "="*80)
    print("TEST 2: Complex Query (Agentic Multi-Step RAG)")
    print("="*80)
    
    service = ChatService()
    
    question = "Give me a detailed report of COVID disease including symptoms, transmission, and treatment"
    print(f"\nQuestion: {question}")
    print("\nProcessing with use_agentic=True...")
    
    response = await service.answer_question(question, use_agentic=True)
    
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources: {response.sources_count}")
    print(f"Citations: {len(response.citations)}")
    
    if response.citations:
        print("\nUnique Sources (Deduplicated):")
        for i, citation in enumerate(response.citations, 1):
            print(f"{i}. {citation.title}")

async def test_comparative_query():
    """Test with a comparative query"""
    print("\n" + "="*80)
    print("TEST 3: Comparative Query (Agentic Multi-Step RAG)")
    print("="*80)
    
    service = ChatService()
    
    question = "Compare COVID-19 and influenza"
    print(f"\nQuestion: {question}")
    print("\nProcessing with use_agentic=True...")
    
    response = await service.answer_question(question, use_agentic=True)
    
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources: {response.sources_count}")
    print(f"Citations: {len(response.citations)}")

async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("AGENTIC RAG FUNCTIONALITY TEST")
    print("="*80)
    
    try:
        # Test 1: Simple query
        await test_simple_query()
        
        # Test 2: Complex query with agentic pipeline
        await test_complex_query()
        
        # Test 3: Comparative query
        await test_comparative_query()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
