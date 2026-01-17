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
        
        # Note: vector storage now uses LangChain PGVector tables
    
    # Cache the embedding dimension
    await set_embedding_dimension(dimension)


async def close_database():
    """Close database connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_langchain_chunk_counts(collection_name: str) -> dict[int, int]:
    """Return a mapping of document_id to chunk count from LangChain PGVector tables."""
    async with get_db_connection() as conn:
        result = await conn.execute(
            text("""
            SELECT
                (e.cmetadata->>'document_id')::int AS document_id,
                COUNT(*) AS chunk_count
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
              AND e.cmetadata ? 'document_id'
            GROUP BY (e.cmetadata->>'document_id')::int
            """),
            {"collection": collection_name}
        )
        rows = result.fetchall()

    return {row.document_id: row.chunk_count for row in rows}


async def get_langchain_chunk_count(document_id: int, collection_name: str) -> int:
    """Return chunk count for a single document_id."""
    async with get_db_connection() as conn:
        result = await conn.execute(
            text("""
            SELECT COUNT(*) AS chunk_count
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
              AND e.cmetadata->>'document_id' = :doc_id
            """),
            {"collection": collection_name, "doc_id": str(document_id)}
        )
        row = result.fetchone()
        return int(row.chunk_count) if row else 0


async def delete_langchain_embeddings(document_id: int, collection_name: str):
    """Delete embeddings for a document_id from LangChain PGVector tables."""
    async with get_db_connection() as conn:
        # Resolve collection id
        result = await conn.execute(
            text("""
            SELECT uuid FROM langchain_pg_collection WHERE name = :collection
            """),
            {"collection": collection_name}
        )
        row = result.fetchone()
        if not row:
            return

        collection_id = row.uuid
        await conn.execute(
            text("""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = :collection_id
              AND cmetadata->>'document_id' = :doc_id
            """),
            {"collection_id": collection_id, "doc_id": str(document_id)}
        )
