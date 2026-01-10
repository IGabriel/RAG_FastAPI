"""FastAPI main application."""
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import initialize_database, close_database, get_db_connection
from app.ingest import probe_embedding_dimension, enqueue_indexing
from app.rag import retrieve_chunks, chat
from app.schemas import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    RetrievalRequest,
    RetrievalResponse,
    ChatRequest,
    ChatResponse,
    ReindexResponse,
    DeleteResponse,
    DocumentStatus
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Initializing embedding model...")
    dimension = probe_embedding_dimension()
    print(f"Detected embedding dimension: {dimension}")
    
    print("Initializing database...")
    await initialize_database(settings.EMBEDDING_MODEL, dimension)
    print("Database initialized successfully")
    
    yield
    
    # Shutdown
    print("Closing database connections...")
    await close_database()


app = FastAPI(
    title="RAG FastAPI Service",
    description="Async RAG service with pgvector and document management",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG FastAPI Service",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/documents",
            "list": "/documents",
            "detail": "/documents/{doc_id}",
            "delete": "/documents/{doc_id}",
            "reindex": "/documents/{doc_id}/reindex",
            "retrieve": "/retrieve",
            "chat": "/chat"
        }
    }


@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing."""
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to start
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f}MB"
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ['.pdf', '.txt', '.text', '.md', '.markdown']:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: .pdf, .txt, .md"
        )
    
    try:
        # Create document record in database
        async with get_db_connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO documents (filename, file_path, status)
                VALUES (:filename, :filepath, 'pending')
                RETURNING id
                """,
                {"filename": file.filename, "filepath": ""}
            )
            row = result.fetchone()
            document_id = row[0]
        
        # Save file to disk
        doc_dir = settings.STORAGE_DIR / str(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / file.filename
        
        # Stream file to disk
        with open(file_path, 'wb') as f:
            while chunk := await file.read(8192):
                f.write(chunk)
        
        # Update file path in database
        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE documents 
                SET file_path = :filepath
                WHERE id = :doc_id
                """,
                {"filepath": str(file_path), "doc_id": document_id}
            )
        
        # Enqueue indexing
        await enqueue_indexing(document_id, file_path)
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status=DocumentStatus.PENDING,
            message="Document uploaded successfully. Processing started."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100)
):
    """List all documents with pagination."""
    offset = (page - 1) * per_page
    
    async with get_db_connection() as conn:
        # Get total count
        result = await conn.execute("SELECT COUNT(*) FROM documents")
        total = result.fetchone()[0]
        
        # Get documents with chunk counts
        result = await conn.execute(
            """
            SELECT 
                d.id,
                d.filename,
                d.status,
                COUNT(c.id) as chunk_count,
                d.created_at,
                d.updated_at
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            LIMIT :limit OFFSET :offset
            """,
            {"limit": per_page, "offset": offset}
        )
        rows = result.fetchall()
        
        documents = [
            DocumentInfo(
                document_id=row.id,
                filename=row.filename,
                status=DocumentStatus(row.status),
                chunk_count=row.chunk_count,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in rows
        ]
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            per_page=per_page
        )


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: int):
    """Get document details."""
    async with get_db_connection() as conn:
        result = await conn.execute(
            """
            SELECT 
                d.id,
                d.filename,
                d.status,
                COUNT(c.id) as chunk_count,
                d.created_at,
                d.updated_at
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            WHERE d.id = :doc_id
            GROUP BY d.id
            """,
            {"doc_id": doc_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(
            document_id=row.id,
            filename=row.filename,
            status=DocumentStatus(row.status),
            chunk_count=row.chunk_count,
            created_at=row.created_at,
            updated_at=row.updated_at
        )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: int):
    """Delete a document and its chunks."""
    async with get_db_connection() as conn:
        # Get file path
        result = await conn.execute(
            "SELECT file_path FROM documents WHERE id = :doc_id",
            {"doc_id": doc_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_path = Path(row[0])
        doc_dir = file_path.parent
        
        # Delete from database (cascades to chunks)
        await conn.execute(
            "DELETE FROM documents WHERE id = :doc_id",
            {"doc_id": doc_id}
        )
    
    # Delete from disk
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    
    return DeleteResponse(
        document_id=doc_id,
        message="Document deleted successfully"
    )


@app.post("/documents/{doc_id}/reindex", response_model=ReindexResponse)
async def reindex_document(doc_id: int):
    """Reindex an existing document."""
    async with get_db_connection() as conn:
        result = await conn.execute(
            "SELECT file_path FROM documents WHERE id = :doc_id",
            {"doc_id": doc_id}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_path = Path(row[0])
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found on disk")
    
    # Enqueue reindexing
    await enqueue_indexing(doc_id, file_path)
    
    return ReindexResponse(
        document_id=doc_id,
        message="Document reindexing started"
    )


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """Retrieve relevant chunks for debugging."""
    chunks = await retrieve_chunks(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id
    )
    
    return RetrievalResponse(
        query=request.query,
        chunks=chunks
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with RAG."""
    answer, chunks = await chat(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id
    )
    
    return ChatResponse(
        query=request.query,
        answer=answer,
        sources=chunks
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
