from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate normalized-size sentence embeddings with Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text to embed cannot be empty.")
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(float).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Batch contains empty text.")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(float).tolist()

    def get_dimension(self) -> int:
        return self.dimension
