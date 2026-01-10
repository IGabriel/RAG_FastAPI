"""Pydantic schemas for API request/response models."""
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    document_id: int
    filename: str
    status: DocumentStatus
    message: str


class DocumentInfo(BaseModel):
    """Document information."""
    document_id: int
    filename: str
    status: DocumentStatus
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: List[DocumentInfo]
    total: int
    page: int
    per_page: int


class RetrievalRequest(BaseModel):
    """Request for document retrieval."""
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[int] = None


class ChunkResult(BaseModel):
    """Retrieved chunk with metadata."""
    chunk_id: int
    document_id: int
    filename: str
    content: str
    distance: float
    chunk_index: int


class RetrievalResponse(BaseModel):
    """Response from retrieval endpoint."""
    query: str
    chunks: List[ChunkResult]


class ChatRequest(BaseModel):
    """Request for chat/RAG endpoint."""
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[int] = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    query: str
    answer: str
    sources: List[ChunkResult]


class ReindexResponse(BaseModel):
    """Response after reindex request."""
    document_id: int
    message: str


class DeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: int
    message: str
