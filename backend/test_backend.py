"""
Backend Testing Script
Comprehensive tests for normal and edge cases
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def print_test(test_name, success=True):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {test_name}")

def test_health_check():
    """Test 1: Health check endpoint"""
    print("\n=== Test 1: Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert 'service_ready' in data
        
        print_test("Health check endpoint", True)
        print(f"Response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print_test("Health check endpoint", False)
        print(f"Error: {e}")
        return False

def test_stats():
    """Test 2: Stats endpoint"""
    print("\n=== Test 2: Get Stats ===")
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        data = response.json()
        
        assert response.status_code == 200
        assert data['success'] == True
        assert 'stats' in data
        
        print_test("Stats endpoint", True)
        print(f"Stats: {json.dumps(data['stats'], indent=2)}")
        return True
    except Exception as e:
        print_test("Stats endpoint", False)
        print(f"Error: {e}")
        return False

def test_search_empty_query():
    """Test 3: Edge Case - Empty query"""
    print("\n=== Test 3: Edge Case - Empty Query ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/search",
            json={"query": ""}
        )
        data = response.json()
        
        assert response.status_code == 400
        assert data['success'] == False
        assert 'error' in data
        
        print_test("Empty query validation", True)
        print(f"Expected error response: {data['error']}")
        return True
    except Exception as e:
        print_test("Empty query validation", False)
        print(f"Error: {e}")
        return False

def test_search_missing_query():
    """Test 4: Edge Case - Missing query field"""
    print("\n=== Test 4: Edge Case - Missing Query Field ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/search",
            json={"top_k": 10}
        )
        data = response.json()
        
        assert response.status_code == 400
        assert data['success'] == False
        
        print_test("Missing query field validation", True)
        print(f"Expected error response: {data['error']}")
        return True
    except Exception as e:
        print_test("Missing query field validation", False)
        print(f"Error: {e}")
        return False

def test_search_invalid_top_k():
    """Test 5: Edge Case - Invalid top_k"""
    print("\n=== Test 5: Edge Case - Invalid top_k Value ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/search",
            json={"query": "test", "top_k": 1000}  # Too large
        )
        data = response.json()
        
        assert response.status_code == 400
        assert data['success'] == False
        
        print_test("Invalid top_k validation", True)
        print(f"Expected error response: {data['error']}")
        return True
    except Exception as e:
        print_test("Invalid top_k validation", False)
        print(f"Error: {e}")
        return False

def test_index_invalid_url():
    """Test 6: Edge Case - Invalid URL"""
    print("\n=== Test 6: Edge Case - Invalid URL ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/index",
            json={"url": "not-a-valid-url"}
        )
        data = response.json()
        
        # This should fail at the fetcher level
        assert data['success'] == False
        
        print_test("Invalid URL handling", True)
        print(f"Error caught: {data.get('error', 'Unknown')}")
        return True
    except Exception as e:
        print_test("Invalid URL handling", False)
        print(f"Error: {e}")
        return False

def test_index_missing_url():
    """Test 7: Edge Case - Missing URL"""
    print("\n=== Test 7: Edge Case - Missing URL Field ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/index",
            json={}
        )
        data = response.json()
        
        assert response.status_code == 400
        assert data['success'] == False
        
        print_test("Missing URL field validation", True)
        print(f"Expected error response: {data['error']}")
        return True
    except Exception as e:
        print_test("Missing URL field validation", False)
        print(f"Error: {e}")
        return False

def test_normal_index_and_search():
    """Test 8: Normal Case - Index and Search"""
    print("\n=== Test 8: Normal Case - Index and Search ===")
    
    # Use a reliable, small test page
    test_url = "https://smarter.codes/"
    test_query = "AI"
    
    try:
        print(f"\n📥 Indexing URL: {test_url}")
        response = requests.post(
            f"{BASE_URL}/api/index-and-search",
            json={
                "url": test_url,
                "query": test_query,
                "top_k": 5,
                "score_threshold": 0.1
            },
            timeout=120  # 2 minutes timeout for indexing
        )
        
        data = response.json()
        
        if data['success']:
            print_test("Index and search operation", True)
            print(f"\n✓ Processing stats:")
            print(f"  - HTML size: {data['processing']['html_size']} chars")
            print(f"  - Text extracted: {data['processing']['text_size']} chars")
            print(f"  - Chunks created: {data['processing']['chunk_count']}")
            print(f"  - Vectors indexed: {data['processing']['indexed_count']}")
            print(f"\n🔍 Search results for '{test_query}':")
            print(f"  - Total results: {data['total_results']}")
            
            if data['results']:
                for result in data['results'][:3]:
                    print(f"\n  Rank #{result['rank']} (Score: {result['score']:.4f})")
                    print(f"  Text: {result['text'][:100]}...")
                    print(f"  Chunk ID: {result['chunk_id']}")
            
            return True
        else:
            print_test("Index and search operation", False)
            print(f"Error: {data.get('error', 'Unknown')}")
            return False
            
    except requests.Timeout:
        print_test("Index and search operation", False)
        print("Error: Request timed out (>120s)")
        return False
    except Exception as e:
        print_test("Index and search operation", False)
        print(f"Error: {e}")
        return False

def test_search_only():
    """Test 9: Normal Case - Search existing content"""
    print("\n=== Test 9: Normal Case - Search Indexed Content ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/search",
            json={
                "query": "domain",
                "top_k": 3,
                "score_threshold": 0.1
            }
        )
        
        data = response.json()
        
        if data['success']:
            print_test("Search indexed content", True)
            print(f"Found {data['total_results']} results")
            
            if data['results']:
                for result in data['results']:
                    print(f"  - Rank #{result['rank']}: {result['text'][:60]}... (Score: {result['score']:.4f})")
            else:
                print("  No results found above threshold")
            
            return True
        else:
            print_test("Search indexed content", False)
            print(f"Error: {data.get('error')}")
            return False
            
    except Exception as e:
        print_test("Search indexed content", False)
        print(f"Error: {e}")
        return False

def test_invalid_endpoint():
    """Test 10: Edge Case - Invalid endpoint"""
    print("\n=== Test 10: Edge Case - Invalid Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/api/invalid")
        data = response.json()
        
        assert response.status_code == 404
        assert data['success'] == False
        
        print_test("404 handling", True)
        print(f"Expected 404 response: {data['error']}")
        return True
    except Exception as e:
        print_test("404 handling", False)
        print(f"Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 BACKEND COMPREHENSIVE TESTING")
    print("="*60)
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/api/health", timeout=2)
    except:
        print("\n❌ ERROR: Flask server is not running!")
        print("Please start the server first with: python app.py")
        return
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    time.sleep(0.5)
    
    results.append(("Stats Endpoint", test_stats()))
    time.sleep(0.5)
    
    results.append(("Empty Query Validation", test_search_empty_query()))
    time.sleep(0.5)
    
    results.append(("Missing Query Validation", test_search_missing_query()))
    time.sleep(0.5)
    
    results.append(("Invalid top_k Validation", test_search_invalid_top_k()))
    time.sleep(0.5)
    
    results.append(("Invalid URL Handling", test_index_invalid_url()))
    time.sleep(0.5)
    
    results.append(("Missing URL Validation", test_index_missing_url()))
    time.sleep(0.5)
    
    results.append(("404 Handling", test_invalid_endpoint()))
    time.sleep(0.5)
    
    # Heavy tests
    print("\n⚠️  Running heavy tests (may take 1-2 minutes)...")
    results.append(("Index and Search", test_normal_index_and_search()))
    time.sleep(1)
    
    results.append(("Search Only", test_search_only()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend is production-ready!")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review errors above.")

if __name__ == "__main__":
    run_all_tests()
