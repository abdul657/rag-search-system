# RAG Search System

A small, modular semantic-search API built with **Python, FastAPI, Sentence Transformers, and Pinecone**. It demonstrates the retrieval part of a retrieval-augmented generation (RAG) workflow: ingest text, create embeddings, store vectors, and retrieve the most similar documents for a query.

> This repository currently implements semantic retrieval. It does **not** call a generative LLM to compose answers. The API returns matched source documents and similarity scores.

## Architecture

```text
Client
  |
  | POST /index                 POST /search
  v                                  v
FastAPI ----------------------> RetrievalPipeline
                                  |          |
                           EmbeddingService  VectorStore
                           Sentence          Pinecone
                           Transformers      cosine search
```

- **API:** FastAPI request validation and interactive OpenAPI docs.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors by default).
- **Vector database:** Pinecone serverless index with cosine similarity.
- **Schemas/config:** Pydantic models and environment-based settings.

## Requirements

- Python 3.10 or 3.11
- A Pinecone API key and permission to create/use a serverless index
- Internet access on the first run so Sentence Transformers can download its model

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/abdul657/rag-search-system.git
cd rag-search-system
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Pinecone

Copy `.env.example` to `.env` and set your key:

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env`:

```dotenv
PINECONE_API_KEY=replace_with_your_key
INDEX_NAME=rag-search
EMBEDDING_DIMENSION=384
CORS_ORIGINS=http://localhost:5173
```

Never commit your real `.env` file or API key. The configured embedding model must produce the same dimension as the Pinecone index. The default model produces 384 dimensions.

### 4. Start the API

```bash
uvicorn main:app --reload
```

Open:
- API overview: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- Liveness check: http://127.0.0.1:8000/health

The embedding model and Pinecone connection are initialized lazily when an indexing or search request first needs them. The health endpoint can therefore respond before external services are configured.

### 5. Index sample documents

In another terminal, with the virtual environment active:

```bash
python load_sample_data.py
```

Then query the API:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"text":"What is semantic search?","top_k":3}'
```

Example indexing payload:

```json
[
  {
    "id": "doc-1",
    "text": "FastAPI is a Python framework for building APIs.",
    "metadata": {"source": "example"}
  }
]
```

### 6. Run tests

```bash
pytest -q
```

The API tests use a fake pipeline and do not require Pinecone credentials or a running vector database.

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information and endpoint list |
| GET | `/health` | Process liveness check |
| POST | `/index` | Embed and upsert a list of documents |
| POST | `/search` | Embed a query and return nearest documents |

### Search request

```json
{"text":"distributed systems observability","top_k":5}
```

`top_k` must be between 1 and 50. Search results include the document ID, stored text, similarity score, and metadata.

## Project layout

```text
.
├── config.py             # Environment-based settings
├── embeddings.py         # Sentence Transformers wrapper
├── load_sample_data.py   # Sample document ingestion client
├── main.py               # FastAPI application
├── models.py             # Pydantic request/response schemas
├── retrieval.py          # Embedding + vector-search orchestration
├── vector_store.py       # Pinecone integration
├── data/
│   └── sample_documents.txt
├── tests/
│   └── test_api.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Current scope and limitations

- This is a semantic retrieval API, not a complete chatbot or LLM answer-generation service.
- Pinecone is the configured vector backend; a valid API key is required for indexing and search.
- The default embedding model is downloaded by Sentence Transformers on first use.
- The sample API is intended for local development. Before production use, add authentication, request rate limits, stricter CORS origins, structured tracing/metrics, and a deployment-specific secrets manager.

## License

No license is currently declared. Contact the repository owner before reusing this code in another project.
