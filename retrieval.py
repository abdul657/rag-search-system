from embeddings import EmbeddingService
from models import Document, Query, SearchResponse
from vector_store import VectorStore


class RetrievalPipeline:
    """Coordinates document embeddings and vector search."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        if self.embedding_service.get_dimension() != self.vector_store.dimension:
            raise ValueError(
                "Embedding model dimension does not match the configured Pinecone index dimension."
            )
        self.vector_store.create_index()

    def index_documents(self, documents: list[Document]) -> dict:
        if not documents:
            return {"status": "success", "indexed": 0}

        texts = [document.text for document in documents]
        embeddings = self.embedding_service.encode_batch(texts)
        self.vector_store.upsert_documents(documents, embeddings)
        return {"status": "success", "indexed": len(documents)}

    def search(self, query: Query) -> SearchResponse:
        query_embedding = self.embedding_service.encode_text(query.text)
        results = self.vector_store.search(query_embedding, query.top_k)
        return SearchResponse(query=query.text, results=results)
