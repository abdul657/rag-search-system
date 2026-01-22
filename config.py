
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_environment: str
    index_name: str = "rag-search"
    embedding_dimension: int = 384
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
