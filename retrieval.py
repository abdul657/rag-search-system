from embeddings import EmbeddingService
from vector_store import VectorStore
from models import Document, Query, SearchResponse
from typing import List

class RetrievalPipeline:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.vector_store.create_index()
    
    def index_documents(self, documents: List[Document]):
        """Index documents into vector store"""
        # Extract text from documents
        texts = [doc.text for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_service.encode_batch(texts)
        
        # Upload to vector store
        self.vector_store.upsert_documents(documents, embeddings)
        
        return {"status": "success", "indexed": len(documents)}
    
    def search(self, query: Query) -> SearchResponse:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embedding_service.encode_text(query.text)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, query.top_k)
        
        return SearchResponse(
            query=query.text,
            results=results
        )