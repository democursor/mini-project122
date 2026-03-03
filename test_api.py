"""
Test the FastAPI backend
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    print("\n2. Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_list_documents():
    """Test list documents"""
    print("\n3. Testing List Documents...")
    response = requests.get(f"{BASE_URL}/api/documents/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_graph_stats():
    """Test graph statistics"""
    print("\n4. Testing Graph Statistics...")
    response = requests.get(f"{BASE_URL}/api/graph/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_search():
    """Test semantic search"""
    print("\n5. Testing Semantic Search...")
    payload = {
        "query": "machine learning",
        "top_k": 5
    }
    response = requests.post(f"{BASE_URL}/api/search/", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 60)
    print("FastAPI Backend Tests")
    print("=" * 60)
    print("\nMake sure the API server is running:")
    print("  python run_api.py")
    print("\nStarting tests...")
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("List Documents", test_list_documents),
        ("Graph Statistics", test_graph_stats),
        ("Semantic Search", test_search),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASSED" if success else "❌ FAILED"))
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append((name, f"❌ ERROR: {str(e)}"))
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, result in results:
        print(f"{name}: {result}")
    print("=" * 60)

if __name__ == "__main__":
    main()
