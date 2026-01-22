RAG Search System
A lightweight Retrieval‑Augmented Generation (RAG) backend built with FastAPI, HuggingFace embeddings, and Pinecone vector search.
This project demonstrates a clean, modular architecture for building intelligent search systems that combine semantic embeddings with fast vector retrieval.

PLS read the requirement.txt file to not run into any errors

Features

Semantic search using transformer embeddings

FastAPI backend with clean, modular endpoints

Vector store integration (Pinecone or local)

 Embeddings pipeline for document ingestion

 Unit tests for API reliability

Environment variable management with .env.example

 Production‑ready project structure

Add frontend UI

git clone https://github.com/abdul657/rag-search-system.git
cd rag-search-system


python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


pip install -r requirements.txt


cp .env.example .env


PINECONE_API_KEY=your_key_here
PINECONE_ENV=your_env_here

uvicorn app.main:app --reload

http://localhost:8000

pytest

