"""Configuration management for RAG service."""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb"
    DATABASE_URL_SYNC: str = ""
    
    # Storage
    STORAGE_DIR: Path = Path("./storage")
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # Vector store
    VECTOR_COLLECTION: str = "rag_documents"

    # LangChain retrieval features
    USE_SELF_QUERY: bool = False
    ENABLE_RERANKER: bool = False
    RERANKER_MODEL_PATH: str = ""
    
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
    
    # LLM generation
    LLM_MAX_INPUT_LENGTH: int = 2000
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    
    # External OCR service (optional)
    OCR_SERVICE_URL: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()

# Ensure storage directory exists
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
