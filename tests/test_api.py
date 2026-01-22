import requests

BASE_URL = "http://localhost:8000"

def test_index():
    documents = [
        {
            "id": "test_1",
            "text": "FastAPI is a modern Python web framework",
            "metadata": {"type": "technology"}
        }
    ]
    
    response = requests.post(f"{BASE_URL}/index", json=documents)
    print("Index Response:", response.json())

def test_search():
    query = {
        "text": "web framework",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/search", json=query)
    print("Search Response:", response.json())

if __name__ == "__main__":
    test_index()
    test_search()