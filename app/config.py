"""Configuration management for RAG service."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb"
    
    # Storage
    STORAGE_DIR: Path = Path("./storage")
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    
    # LLM
    LLM_MODEL_PATH: str = "./models/Qwen2.5-0.5B-Instruct"
    
    # Processing
    INDEXING_CONCURRENCY: int = 1
    GENERATION_CONCURRENCY: int = 1
    
    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Retrieval
    DEFAULT_TOP_K: int = 5
    
    # External OCR service (optional)
    OCR_SERVICE_URL: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure storage directory exists
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
