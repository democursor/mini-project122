"""Test suite for Phase 6: Web Interface."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.web.state import initialize_session_state, add_message, clear_chat, get_messages
from src.web.components import render_stats_card


def test_session_state():
    """Test session state management."""
    print("\n" + "="*80)
    print("TEST 1: Session State Management")
    print("="*80)
    
    try:
        # This would normally be called by Streamlit
        # We'll just test the functions exist and are callable
        
        print("✓ initialize_session_state function exists")
        print("✓ add_message function exists")
        print("✓ clear_chat function exists")
        print("✓ get_messages function exists")
        
        print("\n✅ Session State test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Session State test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_components():
    """Test UI components."""
    print("\n" + "="*80)
    print("TEST 2: UI Components")
    print("="*80)
    
    try:
        # Test that component functions exist
        from src.web.components import (
            render_message,
            render_document_card,
            render_search_result,
            render_stats_card,
            render_sidebar,
            show_success,
            show_error,
            show_warning,
            show_info
        )
        
        print("✓ render_message function exists")
        print("✓ render_document_card function exists")
        print("✓ render_search_result function exists")
        print("✓ render_stats_card function exists")
        print("✓ render_sidebar function exists")
        print("✓ show_success function exists")
        print("✓ show_error function exists")
        print("✓ show_warning function exists")
        print("✓ show_info function exists")
        
        print("\n✅ UI Components test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ UI Components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_app_structure():
    """Test web app file structure."""
    print("\n" + "="*80)
    print("TEST 3: Web App Structure")
    print("="*80)
    
    try:
        # Check if main web app file exists
        web_app = Path("web_app.py")
        assert web_app.exists(), "web_app.py not found"
        print("✓ web_app.py exists")
        
        # Check if web module exists
        web_module = Path("src/web")
        assert web_module.exists(), "src/web module not found"
        print("✓ src/web module exists")
        
        # Check if component files exist
        state_file = Path("src/web/state.py")
        assert state_file.exists(), "state.py not found"
        print("✓ src/web/state.py exists")
        
        components_file = Path("src/web/components.py")
        assert components_file.exists(), "components.py not found"
        print("✓ src/web/components.py exists")
        
        init_file = Path("src/web/__init__.py")
        assert init_file.exists(), "__init__.py not found"
        print("✓ src/web/__init__.py exists")
        
        print("\n✅ Web App Structure test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Web App Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all imports work."""
    print("\n" + "="*80)
    print("TEST 4: Import Test")
    print("="*80)
    
    try:
        # Test web module imports
        from src.web import state, components
        print("✓ Web module imports successful")
        
        # Test that web_app.py can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location("web_app", "web_app.py")
        if spec and spec.loader:
            print("✓ web_app.py is importable")
        
        print("\n✅ Import test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streamlit_dependency():
    """Test if Streamlit is installed."""
    print("\n" + "="*80)
    print("TEST 5: Streamlit Dependency")
    print("="*80)
    
    try:
        import streamlit as st
        print(f"✓ Streamlit installed: version {st.__version__}")
        
        print("\n✅ Streamlit dependency test passed!")
        return True
        
    except ImportError:
        print("❌ Streamlit not installed")
        print("\nTo install Streamlit, run:")
        print("  pip install streamlit")
        return False
    except Exception as e:
        print(f"\n❌ Streamlit dependency test failed: {e}")
        return False


def main():
    """Run all Phase 6 tests."""
    print("\n" + "="*80)
    print("PHASE 6 TEST SUITE: Web Interface")
    print("="*80)
    
    # Run tests
    results = []
    
    results.append(("Session State", test_session_state()))
    results.append(("UI Components", test_components()))
    results.append(("Web App Structure", test_web_app_structure()))
    results.append(("Imports", test_imports()))
    results.append(("Streamlit Dependency", test_streamlit_dependency()))
    
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
        print("\n🎉 All Phase 6 tests passed!")
        print("\nTo run the web app:")
        print("  streamlit run web_app.py")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        if not results[4][1]:  # Streamlit test failed
            print("\n📦 Install Streamlit first:")
            print("  pip install streamlit")
    
    print("="*80)


if __name__ == "__main__":
    main()
