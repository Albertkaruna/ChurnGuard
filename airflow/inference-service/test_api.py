import requests
import json

# Configuration
API_URL = "http://localhost:8000"  # Change to your service URL
# For Kubernetes NodePort: http://<node-ip>:30080

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info Endpoint ===")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction"""
    print("\n=== Testing Single Prediction ===")
    
    # Sample data - 20 numeric features
    data = {
        "features": [
            0.5, -1.2, 0.8, -0.3, 1.5, 0.2, -0.7, 0.9, -0.4, 1.1,
            0.6, -0.9, 0.3, -1.0, 0.7, 0.1, -0.5, 0.4, -0.8, 1.3
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction"""
    print("\n=== Testing Batch Prediction ===")
    
    # Sample batch data - each instance has 20 features
    data = {
        "instances": [
            [0.5, -1.2, 0.8, -0.3, 1.5, 0.2, -0.7, 0.9, -0.4, 1.1,
             0.6, -0.9, 0.3, -1.0, 0.7, 0.1, -0.5, 0.4, -0.8, 1.3],
            [-0.5, 1.2, -0.8, 0.3, -1.5, -0.2, 0.7, -0.9, 0.4, -1.1,
             -0.6, 0.9, -0.3, 1.0, -0.7, -0.1, 0.5, -0.4, 0.8, -1.3],
            [0.3, -0.7, 0.5, -0.1, 1.0, 0.4, -0.9, 0.6, -0.2, 0.8,
             0.7, -0.4, 0.2, -0.8, 0.9, 0.3, -0.6, 0.1, -0.5, 1.1]
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print(f"Testing API at: {API_URL}")
    print("=" * 50)
    
    tests = [
        ("Root", test_root),
        ("Health", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("=== Test Summary ===")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
