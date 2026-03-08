"""Test script to detect and validate Google Gemini models."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")


def test_google_model_detection():
    """Test Google model detection and selection."""
    print("\n" + "="*80)
    print("GOOGLE GEMINI MODEL DETECTION TEST")
    print("="*80)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set in environment")
        print("   Set it with: set GOOGLE_API_KEY=your-key-here")
        return False
    
    print(f"✓ API Key found: {api_key[:20]}...")
    
    try:
        import google.generativeai as genai
        
        # Configure API
        print("\n1. Configuring Google AI Studio API...")
        genai.configure(api_key=api_key)
        print("   ✓ API configured")
        
        # List all models
        print("\n2. Listing all available models...")
        all_models = []
        for model in genai.list_models():
            all_models.append({
                'name': model.name,
                'display_name': model.display_name,
                'description': model.description,
                'methods': model.supported_generation_methods
            })
        
        print(f"   ✓ Found {len(all_models)} total models")
        
        # Filter models that support generateContent
        print("\n3. Filtering models with generateContent support...")
        compatible_models = []
        for model in all_models:
            if 'generateContent' in model['methods']:
                compatible_models.append(model)
                model_name = model['name'].replace('models/', '')
                print(f"   ✓ {model_name}")
                print(f"      Display: {model['display_name']}")
                print(f"      Methods: {', '.join(model['methods'])}")
        
        if not compatible_models:
            print("   ❌ No models support generateContent!")
            return False
        
        print(f"\n   Total compatible models: {len(compatible_models)}")
        
        # Test model selection logic
        print("\n4. Testing model selection logic...")
        from src.rag.llm_client import LLMClient
        
        # Test with non-existent model (should fallback)
        print("\n   Testing fallback from 'gemini-1.5-flash'...")
        try:
            client = LLMClient(
                provider="google",
                model="gemini-1.5-flash",
                api_key=api_key
            )
            print(f"   ✓ Selected model: {client.model}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test generation with selected model
        print("\n5. Testing text generation...")
        test_prompt = "Say 'Hello, this is a test' in one sentence."
        
        try:
            response = client.generate_response(
                prompt=test_prompt,
                max_tokens=50,
                temperature=0.3
            )
            print(f"   ✓ Generation successful!")
            print(f"   Response: {response[:100]}...")
            print(f"   Length: {len(response)} characters")
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return False
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ API Key: Valid")
        print(f"✅ Available Models: {len(compatible_models)}")
        print(f"✅ Selected Model: {client.model}")
        print(f"✅ Text Generation: Working")
        print("\n🎉 All tests passed! Google Gemini is ready to use.")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error messages
        error_msg = str(e).lower()
        print("\n" + "="*80)
        print("TROUBLESHOOTING")
        print("="*80)
        
        if "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
            print("❌ Authentication Error:")
            print("   Your API key is invalid or expired.")
            print("   Get a new key from: https://makersuite.google.com/app/apikey")
        
        elif "quota" in error_msg or "429" in error_msg:
            print("❌ Quota Error:")
            print("   You've exceeded your API quota.")
            print("   Check your usage at: https://makersuite.google.com/")
        
        elif "404" in error_msg:
            print("❌ Model Not Found:")
            print("   The requested model is not available.")
            print("   This script will auto-detect available models.")
        
        else:
            print("❌ Unknown Error:")
            print("   Check your internet connection and API key.")
        
        print("="*80)
        return False


if __name__ == "__main__":
    success = test_google_model_detection()
    sys.exit(0 if success else 1)
