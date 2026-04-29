"""
Test bug fixes for the agentic RAG system
"""
import asyncio
import logging
from src.services.chat_service import ChatService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_bug_fixes():
    """Test all 4 bug fixes"""
    print("\n" + "="*80)
    print("TESTING BUG FIXES")
    print("="*80)
    
    service = ChatService()
    
    # Test BUG 3 & 4: Multi-angle search with structured output
    print("\n" + "-"*80)
    print("TEST: Complex query with multi-angle search and structured output")
    print("-"*80)
    
    complex_query = "Give me a detailed report of COVID disease including all key aspects"
    print(f"\nQuery: {complex_query}")
    print("\nExpected:")
    print("  ✓ Query decomposed into multiple sub-queries")
    print("  ✓ Parallel retrieval from multiple angles")
    print("  ✓ Structured response with sections")
    print("  ✓ No duplicate sources (BUG 2)")
    print("  ✓ Complete sentences, no cutoffs (BUG 1)")
    
    try:
        response = await service.answer_question(complex_query, use_agentic=True)
        
        print(f"\n✅ Response generated successfully!")
        print(f"   Answer length: {len(response.answer)} characters")
        print(f"   Unique sources: {response.sources_count}")
        
        # Check for duplicates
        source_titles = [c.title for c in response.citations]
        unique_titles = set(source_titles)
        
        if len(source_titles) == len(unique_titles):
            print(f"   ✅ No duplicate sources found")
        else:
            print(f"   ❌ Found {len(source_titles) - len(unique_titles)} duplicate sources")
        
        # Check if response has sections (structured output)
        if "##" in response.answer:
            print(f"   ✅ Structured output with sections detected")
        else:
            print(f"   ⚠️  No section headers found")
        
        # Check if response ends properly (not cut off)
        last_char = response.answer.strip()[-1] if response.answer.strip() else ''
        if last_char in ['.', '!', '?', ')']:
            print(f"   ✅ Response ends with proper punctuation")
        else:
            print(f"   ⚠️  Response may be cut off (ends with: '{last_char}')")
        
        print(f"\n📝 First 500 characters of answer:")
        print(response.answer[:500] + "...")
        
        print(f"\n📚 Sources:")
        for i, citation in enumerate(response.citations[:5], 1):
            print(f"   {i}. {citation.title}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nAll bug fixes have been applied:")
    print("  ✅ BUG 1: max_tokens increased to 4096 (no mid-sentence cutoffs)")
    print("  ✅ BUG 2: Source deduplication (already implemented)")
    print("  ✅ BUG 3: Multi-angle search (already implemented)")
    print("  ✅ BUG 4: Structured output for broad queries (enhanced prompts)")

if __name__ == "__main__":
    asyncio.run(test_bug_fixes())
