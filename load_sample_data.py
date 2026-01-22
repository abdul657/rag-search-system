from models import Document
import requests

def load_sample_documents():
    with open('data/sample_documents.txt', 'r') as f:
        lines = f.readlines()
    
    documents = []
    for i, line in enumerate(lines):
        if line.strip():
            documents.append(Document(
                id=f"doc_{i}",
                text=line.strip(),
                metadata={"source": "sample_data"}
            ))
    
    return documents

if __name__ == "__main__":
    docs = load_sample_documents()
    
    # Send to API
    response = requests.post(
        "http://localhost:8000/index",
        json=[doc.dict() for doc in docs]
    )
    
    print(response.json())