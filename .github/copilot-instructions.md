# GitHub Copilot / AI Agent Instructions

This repository implements a small Retrieval-Augmented-Generation (RAG) search API using FastAPI, a sentence-transformers embedding service, and Pinecone as the vector store. Use these instructions to be productive quickly when making code changes or adding features.

## Big picture
- **API layer:** [main.py](main.py) exposes endpoints `POST /index` and `POST /search` and a `/health` check.
- **Retrieval pipeline:** [retrieval.py](retrieval.py) wires together the `EmbeddingService` and `VectorStore` and defines `index_documents()` and `search()` flows.
- **Embeddings:** [embeddings.py](embeddings.py) uses `sentence-transformers` (`SentenceTransformer`) — change the model name here to swap embedder.
- **Vector store:** [vector_store.py](vector_store.py) wraps Pinecone operations (create index, upsert, query). Index name and dimension come from config.
- **Schemas:** [models.py](models.py) contains Pydantic models `Document`, `Query`, `SearchResult`, and `SearchResponse` used by FastAPI request/response validation.

## How data flows (quick)
1. Client POSTs list of `Document` to `/index` (see [main.py](main.py)).
2. `RetrievalPipeline.index_documents()` extracts texts, computes embeddings via `EmbeddingService.encode_batch()`, and calls `VectorStore.upsert_documents()`.
3. Client POSTs a `Query` to `/search` → pipeline computes query embedding and calls `VectorStore.search()` returning `SearchResponse`.

## Running & developer workflows
- Start local server (dev): `uvicorn main:app --reload` (or run `python main.py`).
- Before running, ensure Python 3.10/3.11 and dependencies from `requirements.txt` are installed and that heavy deps (PyTorch) are available for `sentence-transformers`.
- Environment: the project reads settings via `get_settings()` in [config.py](config.py). Provide `pinecone_api_key` and `pinecone_environment` via a `.env` file or environment variables.
- To load sample data: run `python load_sample_data.py` after the server is running (it POSTs to `/index`).
- Tests are lightweight integration scripts in [tests/test_api.py](tests/test_api.py) — they assume a running server at `http://localhost:8000`.

## Project-specific conventions & notes
- Pydantic models in `models.py` are the single source of truth for API payloads — always update them first when changing shape.
- Embedding dimension is read from the embedding model or set in config (`embedding_dimension` in [config.py](config.py)). Keep model and Pinecone index dimension in sync.
- `VectorStore.upsert_documents()` batches upserts (batch size = 100). Follow this pattern for other bulk operations to avoid rate issues.
- `vector_store.create_index()` checks existence and waits for readiness — tests and local runs expect the index to be present or created automatically.

## Integration points & things to watch
- Pinecone client usage lives in [vector_store.py](vector_store.py). API keys must be provided; test environments commonly mock this layer.
- The embedder (`sentence-transformers`) may require GPU/appropriate torch; for CI use a lightweight mock or a small CPU model.
- Error handling is minimal and surfaced as 500 HTTP responses in [main.py](main.py). When adding retries or more advanced behavior, prefer adding logic in `RetrievalPipeline` and `VectorStore`.

## Quick examples
- Change embedder model: edit `model_name` in `EmbeddingService.__init__` ([embeddings.py](embeddings.py)).
- Index documents via HTTP: POST JSON array of `Document` objects to `/index` (see `load_sample_data.py`).

If anything here is unclear or you want this file expanded with run/debug commands or CI guidance, tell me which sections to elaborate and I'll iterate.
