from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from models import Document, Query, SearchResponse
from retrieval import RetrievalPipeline
import uvicorn

app = FastAPI(
    title="RAG Search System",
    description="Retrieval-Augmented Generation Search API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrieval pipeline
pipeline = RetrievalPipeline()

@app.get("/")
def read_root():
    return {
        "message": "RAG Search System API",
        "version": "1.0.0",
        "endpoints": {
            "/index": "POST - Index documents",
            "/search": "POST - Search documents",
            "/health": "GET - Health check"
        }
    }

@app.post("/index", status_code=201)
def index_documents(documents: List[Document]):
    """Index documents into the vector store"""
    try:
        result = pipeline.index_documents(documents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
def search_documents(query: Query):
    """Search for relevant documents"""
    try:
        results = pipeline.search(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)# Start the server

# run "uvicorn main:app --reload" to start the server.

