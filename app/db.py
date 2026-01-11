"""Database connection and schema management."""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection, AsyncEngine
from app.config import settings


# Global engine instance
_engine: Optional[AsyncEngine] = None
_embedding_dimension: Optional[int] = None


def get_engine() -> AsyncEngine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
        )
    return _engine


async def get_embedding_dimension() -> int:
    """Get the cached embedding dimension."""
    global _embedding_dimension
    if _embedding_dimension is None:
        raise RuntimeError("Embedding dimension not initialized")
    return _embedding_dimension


async def set_embedding_dimension(dimension: int):
    """Set the embedding dimension."""
    global _embedding_dimension
    _embedding_dimension = dimension


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[AsyncConnection, None]:
    """Get a database connection from the pool."""
    engine = get_engine()
    async with engine.begin() as conn:
        yield conn


async def initialize_database(embedding_model: str, dimension: int):
    """Initialize database schema and verify embedding model consistency."""
    engine = get_engine()
    
    async with engine.begin() as conn:
        # Create pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create metadata table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Check existing metadata
        result = await conn.execute(text(
            "SELECT model_name, dimension FROM embedding_metadata ORDER BY id DESC LIMIT 1"
        ))
        row = result.fetchone()
        
        if row:
            existing_model, existing_dim = row
            if existing_model != embedding_model or existing_dim != dimension:
                raise RuntimeError(
                    f"Embedding model mismatch! Database has {existing_model} "
                    f"(dim={existing_dim}), but trying to use {embedding_model} "
                    f"(dim={dimension})"
                )
        else:
            # Insert metadata for first time
            await conn.execute(
                text("""
                INSERT INTO embedding_metadata (model_name, dimension)
                VALUES (:model, :dim)
                """),
                {"model": embedding_model, "dim": dimension}
            )
        
        # Create documents table
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create chunks table with dynamic vector dimension
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dimension}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create index for vector similarity search
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        # Create index for document_id lookup
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chunks_document_id_idx 
            ON chunks(document_id)
        """))
    
    # Cache the embedding dimension
    await set_embedding_dimension(dimension)


async def close_database():
    """Close database connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
