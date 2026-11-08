import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from models import Document, Query, SearchResponse
from retrieval import RetrievalPipeline

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="RAG Search System",
    description="Semantic retrieval API using Sentence Transformers and Pinecone.",
    version="1.1.0",
)

allowed_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


def get_pipeline() -> RetrievalPipeline:
    """Build the embedding and vector-store clients only when required."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        return pipeline

    if not settings.pinecone_api_key:
        raise HTTPException(
            status_code=503,
            detail="Search backend is not configured. Set PINECONE_API_KEY.",
        )

    try:
        pipeline = RetrievalPipeline()
        app.state.pipeline = pipeline
        return pipeline
    except Exception as exc:
        logger.exception("Unable to initialize retrieval pipeline")
        raise HTTPException(
            status_code=503,
            detail="Search backend could not be initialized. Check configuration and service availability.",
        ) from exc


@app.get("/")
def read_root() -> dict:
    return {
        "message": "RAG Search System API",
        "version": app.version,
        "endpoints": {
            "/index": "POST - Index documents",
            "/search": "POST - Search documents",
            "/health": "GET - Liveness check",
        },
        "docs": "/docs",
    }


@app.post("/index", status_code=201)
def index_documents(documents: list[Document]) -> dict:
    if not documents:
        raise HTTPException(status_code=400, detail="Provide at least one document.")

    try:
        return get_pipeline().index_documents(documents)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc


@app.post("/search", response_model=SearchResponse)
def search_documents(query: Query) -> SearchResponse:
    try:
        return get_pipeline().search(query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document search failed")
        raise HTTPException(status_code=500, detail="Document search failed.") from exc


@app.get("/health")
def health_check() -> dict:
    # Liveness only: external services are initialized lazily by /index and /search.
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
