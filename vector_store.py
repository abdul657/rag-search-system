import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import get_settings
from models import Document, SearchResult


class VectorStore:
    """Pinecone adapter for batched document upserts and cosine search."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY must be configured before creating VectorStore.")

        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.index_name
        self.dimension = settings.embedding_dimension
        self.index = None

    def create_index(self):
        """Create the configured serverless index when absent, then connect."""
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                description = self.pc.describe_index(self.index_name)
                status = description.status
                ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
                if ready:
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"Pinecone index {self.index_name!r} did not become ready within 90 seconds.")

        description = self.pc.describe_index(self.index_name)
        actual_dimension = getattr(description, "dimension", None)
        if actual_dimension is not None and actual_dimension != self.dimension:
            raise ValueError(
                f"Pinecone index {self.index_name!r} has dimension {actual_dimension}; "
                f"configured embedding dimension is {self.dimension}."
            )

        self.index = self.pc.Index(self.index_name)
        return self.index

    def _require_index(self):
        if self.index is None:
            raise RuntimeError("Vector index is not initialized. Call create_index() first.")
        return self.index

    def upsert_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Upsert documents in batches; refuse mismatched input lengths."""
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have exactly one embedding.")
        if not documents:
            return

        vectors = []
        for doc, embedding in zip(documents, embeddings):
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Embedding for document {doc.id!r} has dimension {len(embedding)}; "
                    f"expected {self.dimension}."
                )
            metadata = dict(doc.metadata)
            metadata["text"] = doc.text
            vectors.append({"id": doc.id, "values": embedding, "metadata": metadata})

        index = self._require_index()
        for offset in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[offset : offset + 100])

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return nearest documents from Pinecone."""
        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding has dimension {len(query_embedding)}; expected {self.dimension}."
            )

        response = self._require_index().query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])

        results: list[SearchResult] = []
        for match in matches:
            if isinstance(match, dict):
                match_id = match.get("id", "")
                score = match.get("score", 0.0)
                metadata: dict[str, Any] = match.get("metadata") or {}
            else:
                match_id = getattr(match, "id", "")
                score = getattr(match, "score", 0.0)
                metadata = getattr(match, "metadata", None) or {}

            metadata = dict(metadata)
            text = metadata.pop("text", "")
            results.append(
                SearchResult(
                    id=str(match_id),
                    text=str(text),
                    score=float(score or 0.0),
                    metadata=metadata,
                )
            )
        return results

    def delete_all(self) -> None:
        """Delete every vector in this index. Use deliberately; this is destructive."""
        self._require_index().delete(delete_all=True)
