"""Document ingestion, chunking, and embedding."""
import asyncio
import os
import re
from pathlib import Path
from typing import List, Optional
import aiohttp
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np

from app.config import settings
from app.db import get_db_connection


# Global embedding model and processing queue
_embedding_model: Optional[SentenceTransformer] = None
_indexing_semaphore: Optional[asyncio.Semaphore] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        # If the host cannot access huggingface.co, run in offline/local mode.
        # Libraries also respect env vars like HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE,
        # but SentenceTransformer benefits from an explicit local_files_only flag.
        model_ref = settings.EMBEDDING_MODEL
        model_path_exists = Path(model_ref).exists()
        local_only = (
            model_path_exists
            or os.getenv("HF_HUB_OFFLINE") == "1"
            or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        )
        cache_folder = (
            os.getenv("SENTENCE_TRANSFORMERS_HOME")
            or os.getenv("HF_HOME")
            or os.getenv("TRANSFORMERS_CACHE")
        )

        _embedding_model = SentenceTransformer(
            model_ref,
            cache_folder=cache_folder,
            local_files_only=local_only,
        )
    return _embedding_model


def get_indexing_semaphore() -> asyncio.Semaphore:
    """Get the indexing concurrency semaphore."""
    global _indexing_semaphore
    if _indexing_semaphore is None:
        _indexing_semaphore = asyncio.Semaphore(settings.INDEXING_CONCURRENCY)
    return _indexing_semaphore


def probe_embedding_dimension() -> int:
    """Probe the embedding dimension of the model."""
    model = get_embedding_model()
    test_embedding = model.encode(["test"], convert_to_numpy=True)
    return test_embedding.shape[1]


async def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF, using OCR service if needed."""
    
    def _extract_with_pypdf():
        """Synchronous PDF text extraction."""
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    
    # Try pypdf first
    text = await asyncio.to_thread(_extract_with_pypdf)
    
    # If no text extracted and OCR service is configured, use OCR
    if not text and settings.OCR_SERVICE_URL:
        text = await _ocr_fallback(pdf_path)
    
    return text


async def _ocr_fallback(pdf_path: Path) -> str:
    """Use external OCR service for scanned PDFs."""
    try:
        async with aiohttp.ClientSession() as session:
            with open(pdf_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=pdf_path.name)
                
                async with session.post(
                    settings.OCR_SERVICE_URL,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Expected format: {"pages": [{"page": 1, "text": "..."}]}
                        pages = result.get("pages", [])
                        text = "\n".join(page.get("text", "") for page in pages)
                        return text.strip()
                    else:
                        print(f"OCR service returned status {response.status}")
                        return ""
    except Exception as e:
        print(f"OCR service error: {e}")
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    """Extract text from text file."""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_text_from_markdown(md_path: Path) -> str:
    """Extract text from markdown file."""
    with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


async def extract_text_from_document(file_path: Path) -> str:
    """Extract text from document based on file type."""
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return await extract_text_from_pdf(file_path)
    elif suffix in ['.txt', '.text']:
        return await asyncio.to_thread(extract_text_from_txt, file_path)
    elif suffix in ['.md', '.markdown']:
        return await asyncio.to_thread(extract_text_from_markdown, file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def split_by_markdown_headings(text: str) -> List[str]:
    """Split text by markdown headings first, then by size."""
    # Split by markdown headings (# ## ### etc)
    heading_pattern = r'\n#{1,6}\s+.+\n'
    sections = re.split(heading_pattern, text)
    
    # Keep the headings with their content
    headings = re.findall(heading_pattern, text)
    
    result = []
    for i, section in enumerate(sections):
        if section.strip():
            # Add heading back if exists
            if i > 0 and i - 1 < len(headings):
                result.append(headings[i - 1].strip() + "\n" + section.strip())
            else:
                result.append(section.strip())
    
    return result if result else [text]


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into chunks with overlap, respecting markdown structure."""
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP
    
    # First split by markdown headings
    sections = split_by_markdown_headings(text)
    
    chunks = []
    for section in sections:
        # If section is small enough, keep it as one chunk
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        
        # Otherwise, split into overlapping chunks
        words = section.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap words (calculate based on actual character length)
                overlap_length = 0
                overlap_words = []
                for w in reversed(current_chunk):
                    word_len = len(w) + 1
                    if overlap_length + word_len <= overlap:
                        overlap_words.insert(0, w)
                        overlap_length += word_len
                    else:
                        break
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    return [c.strip() for c in chunks if c.strip()]


async def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model()
    
    def _encode():
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    embeddings = await asyncio.to_thread(_encode)
    return embeddings


async def index_document(document_id: int, file_path: Path):
    """Index a document: extract text, chunk, embed, and store in database."""
    semaphore = get_indexing_semaphore()
    
    async with semaphore:
        try:
            # Update status to processing
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE documents 
                    SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                    WHERE id = :doc_id
                    """,
                    {"doc_id": document_id}
                )
            
            # Extract text
            text = await extract_text_from_document(file_path)
            
            if not text:
                raise ValueError("No text extracted from document")
            
            # Chunk text
            chunks = chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Generate embeddings
            embeddings = await embed_texts(chunks)
            
            # Store chunks in database
            async with get_db_connection() as conn:
                # Delete existing chunks (for reindexing)
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id = :doc_id",
                    {"doc_id": document_id}
                )
                
                # Insert new chunks
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    embedding_list = embedding.tolist()
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content, embedding)
                        VALUES (:doc_id, :idx, :content, :embedding)
                        """,
                        {
                            "doc_id": document_id,
                            "idx": idx,
                            "content": chunk,
                            "embedding": str(embedding_list)
                        }
                    )
                
                # Update document status
                await conn.execute(
                    """
                    UPDATE documents 
                    SET status = 'completed', error_message = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :doc_id
                    """,
                    {"doc_id": document_id}
                )
        
        except Exception as e:
            # Update status to failed
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE documents 
                    SET status = 'failed', error_message = :error, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :doc_id
                    """,
                    {"doc_id": document_id, "error": str(e)}
                )
            raise


async def enqueue_indexing(document_id: int, file_path: Path):
    """Enqueue document indexing as a background task."""
    asyncio.create_task(index_document(document_id, file_path))
