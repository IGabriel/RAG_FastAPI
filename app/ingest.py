"""Document ingestion, chunking, and embedding."""
import asyncio
import re
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks
from sqlalchemy import text
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

from app.config import settings
from app.db import get_db_connection, delete_langchain_embeddings
from app.langchain_utils import get_embeddings, get_vectorstore
from app.langchain_loaders import OCRPdfLoader
import thulac  # type: ignore


# Global processing queue
_indexing_semaphore: Optional[asyncio.Semaphore] = None
_thulac_instance = None



def _sanitize_text(text: str) -> str:
    """Remove characters that PostgreSQL UTF-8 rejects (e.g., NUL bytes)."""
    if not text:
        return text
    return text.replace("\x00", "")


def _is_cjk_heavy(text: str) -> bool:
    """Heuristic: treat text as CJK-heavy if CJK chars dominate letters."""
    if not text:
        return False
    cjk_count = 0
    latin_count = 0
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x20000 <= code <= 0x2A6DF  # CJK Extension B
            or 0x2A700 <= code <= 0x2B73F  # CJK Extension C
            or 0x2B740 <= code <= 0x2B81F  # CJK Extension D
            or 0x2B820 <= code <= 0x2CEAF  # CJK Extension E
            or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
        ):
            cjk_count += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin_count += 1

    total = cjk_count + latin_count
    if total == 0:
        return False
    return cjk_count / total >= 0.3


def _split_by_cjk_punctuation(text: str) -> List[str]:
    """Split Chinese text into sentences by common punctuation."""
    pattern = r"(?<=[。！？；!?;])\s*"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p and p.strip()]


def _get_thulac():
    """Lazy initializer for THULAC tokenizer."""
    global _thulac_instance
    if _thulac_instance is None and thulac is not None:
        _thulac_instance = thulac.thulac(seg_only=True)
    return _thulac_instance


def _tokenize_cjk_words(text: str) -> List[str]:
    """Tokenize Chinese text into words using THULAC.

    Returns empty list if tokenizer is unavailable.
    """
    if not text:
        return []

    tokenizer = _get_thulac()
    if tokenizer is not None:
        # thulac.cut returns a space-separated string when text=True
        cut_text = tokenizer.cut(text, text=True)
        if isinstance(cut_text, str):
            return [t for t in cut_text.split() if t]
        return [t for t, _ in cut_text if t]

    return []


def _chunk_text_cjk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk CJK text by characters with sentence-aware grouping."""
    sentences = _split_by_cjk_punctuation(text)
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) <= chunk_size:
            current = sent if not current else current + sent
            continue

        if current:
            chunks.append(current)
            if overlap > 0:
                current = current[-overlap:] + sent
            else:
                current = sent
        else:
            # Single long sentence: hard-split by chars
            for i in range(0, len(sent), chunk_size):
                piece = sent[i : i + chunk_size]
                if piece:
                    chunks.append(piece)
            current = ""

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def _chunk_text_cjk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk CJK text by word tokens with overlap."""
    sentences = _split_by_cjk_punctuation(text)
    if not sentences:
        sentences = [text]

    tokens: List[str] = []
    for sent in sentences:
        sent_tokens = _tokenize_cjk_words(sent)
        if not sent_tokens:
            # Fallback to character-level if tokenizer yields nothing
            sent_tokens = list(sent)
        tokens.extend(sent_tokens)

    if not tokens:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for token in tokens:
        token_length = len(token)
        if token_length == 0:
            continue

        if current_length + token_length > chunk_size and current_chunk:
            chunks.append("".join(current_chunk))

            # overlap by token lengths
            overlap_length = 0
            overlap_tokens: List[str] = []
            for t in reversed(current_chunk):
                t_len = len(t)
                if overlap_length + t_len <= overlap:
                    overlap_tokens.insert(0, t)
                    overlap_length += t_len
                else:
                    break

            current_chunk = overlap_tokens
            current_length = sum(len(t) for t in current_chunk)

        current_chunk.append(token)
        current_length += token_length

    if current_chunk:
        chunks.append("".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def get_indexing_semaphore() -> asyncio.Semaphore:
    """Get the indexing concurrency semaphore."""
    global _indexing_semaphore
    if _indexing_semaphore is None:
        _indexing_semaphore = asyncio.Semaphore(settings.INDEXING_CONCURRENCY)
    return _indexing_semaphore


def probe_embedding_dimension() -> int:
    """Probe the embedding dimension of the model."""
    embeddings = get_embeddings()
    test_embedding = embeddings.embed_query("test")
    return len(test_embedding)


def _load_with_langchain(file_path: Path) -> str:
    """Load document text using LangChain loaders (sync)."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = OCRPdfLoader(str(file_path))
    elif suffix in [".txt", ".text", ".md", ".markdown"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    return "\n".join(d.page_content for d in docs if d.page_content).strip()


async def extract_text_from_document(file_path: Path) -> str:
    """Extract text from document based on file type."""
    return await asyncio.to_thread(_load_with_langchain, file_path)


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


def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
    """Split text into chunks with overlap, respecting markdown structure."""
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP

    if _is_cjk_heavy(text):
        word_chunks = _chunk_text_cjk_words(text, chunk_size, overlap)
        return word_chunks if word_chunks else _chunk_text_cjk(text, chunk_size, overlap)
    
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


async def index_document(document_id: int, file_path: Path):
    """Index a document: extract text, chunk, embed, and store in database."""
    semaphore = get_indexing_semaphore()
    
    async with semaphore:
        try:
            print(f"[index] start document_id={document_id} file={file_path}")
            # Update status to processing
            async with get_db_connection() as conn:
                await conn.execute(
                    text("""
                    UPDATE documents 
                    SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                    WHERE id = :doc_id
                    """),
                    {"doc_id": document_id}
                )
            
            # Extract text
            document_text = await extract_text_from_document(file_path)
            document_text = _sanitize_text(document_text)
            
            if not document_text:
                raise ValueError(
                    "No text extracted from document. "
                    "If this is a scanned PDF, configure OCR_SERVICE_URL for OCR, "
                    "or upload a text-based PDF/TXT/MD."
                )
            
            # Chunk text
            chunks = [_sanitize_text(chunk) for chunk in chunk_text(document_text)]
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Delete existing embeddings (for reindexing)
            await delete_langchain_embeddings(document_id, settings.VECTOR_COLLECTION)

            # Store chunks in LangChain PGVector
            vectorstore = get_vectorstore()
            metadatas = [
                {
                    "document_id": document_id,
                    "filename": file_path.name,
                    "chunk_index": idx,
                    "chunk_id": idx,
                }
                for idx, _ in enumerate(chunks)
            ]
            ids = [f"{document_id}-{idx}" for idx in range(len(chunks))]
            await asyncio.to_thread(vectorstore.add_texts, texts=chunks, metadatas=metadatas, ids=ids)

            # Update document status
            async with get_db_connection() as conn:
                await conn.execute(
                    text("""
                    UPDATE documents 
                    SET status = 'completed', error_message = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :doc_id
                    """),
                    {"doc_id": document_id}
                )

            print(f"[index] done document_id={document_id}")
        
        except Exception as e:
            # Update status to failed
            async with get_db_connection() as conn:
                await conn.execute(
                    text("""
                    UPDATE documents 
                    SET status = 'failed', error_message = :error, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :doc_id
                    """),
                    {"doc_id": document_id, "error": str(e)}
                )
            print(f"[index] failed document_id={document_id} error={e}")
            raise


def _log_background_exception(task: asyncio.Task):
    try:
        task.result()
    except Exception as e:
        print(f"[index] background task error: {e}")


async def enqueue_indexing(
    document_id: int,
    file_path: Path,
    background_tasks: Optional[BackgroundTasks] = None,
):
    """Enqueue document indexing as a background task.

    Prefer FastAPI/Starlette BackgroundTasks when available (more reliable under
    debug/reload). Falls back to asyncio.create_task otherwise.
    """
    if background_tasks is not None:
        background_tasks.add_task(index_document, document_id, file_path)
        return

    task = asyncio.create_task(index_document(document_id, file_path))
    task.add_done_callback(_log_background_exception)
