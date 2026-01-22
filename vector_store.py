from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from config import get_settings
from models import Document, SearchResult
import time

class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.index_name
        self.dimension = settings.embedding_dimension
        self.index = None
        
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        return self.index
    
    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Upload documents with embeddings to Pinecone"""
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vectors.append({
                "id": doc.id,
                "values": embedding,
                "metadata": {
                    "text": doc.text,
                    **doc.metadata
                }
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        search_results = []
        for match in results['matches']:
            search_results.append(SearchResult(
                id=match['id'],
                text=match['metadata'].get('text', ''),
                score=match['score'],
                metadata={k: v for k, v in match['metadata'].items() if k != 'text'}
            ))
        
        return search_results
    
    def delete_all(self):
        """Delete all vectors from index"""
        self.index.delete(delete_all=True)