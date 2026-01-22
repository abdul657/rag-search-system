from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = {}

class Query(BaseModel):
    text: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]