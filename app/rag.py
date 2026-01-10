"""RAG retrieval and generation."""
import asyncio
from typing import List, Optional, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from app.config import settings
from app.db import get_db_connection
from app.ingest import embed_texts, get_embedding_model
from app.schemas import ChunkResult


# Global LLM model and tokenizer
_llm_model: Optional[AutoModelForCausalLM] = None
_llm_tokenizer: Optional[AutoTokenizer] = None
_generation_semaphore: Optional[asyncio.Semaphore] = None


def get_llm_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Get or load the LLM model and tokenizer."""
    global _llm_model, _llm_tokenizer
    
    if _llm_model is None or _llm_tokenizer is None:
        _llm_tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_PATH,
            trust_remote_code=True
        )
        _llm_model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        _llm_model.eval()
    
    return _llm_model, _llm_tokenizer


def get_generation_semaphore() -> asyncio.Semaphore:
    """Get the generation concurrency semaphore."""
    global _generation_semaphore
    if _generation_semaphore is None:
        _generation_semaphore = asyncio.Semaphore(settings.GENERATION_CONCURRENCY)
    return _generation_semaphore


async def retrieve_chunks(
    query: str,
    top_k: int = 5,
    document_id: Optional[int] = None
) -> List[ChunkResult]:
    """Retrieve relevant chunks using vector similarity search."""
    # Generate query embedding
    query_embeddings = await embed_texts([query])
    query_embedding = query_embeddings[0].tolist()
    
    # Search in database
    async with get_db_connection() as conn:
        if document_id is not None:
            # Filter by document_id
            result = await conn.execute(
                """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    d.filename,
                    c.content,
                    c.chunk_index,
                    (c.embedding <=> :query_embedding::vector) as distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = :doc_id
                ORDER BY distance
                LIMIT :limit
                """,
                {
                    "query_embedding": str(query_embedding),
                    "doc_id": document_id,
                    "limit": top_k
                }
            )
        else:
            # Search across all documents
            result = await conn.execute(
                """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    d.filename,
                    c.content,
                    c.chunk_index,
                    (c.embedding <=> :query_embedding::vector) as distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY distance
                LIMIT :limit
                """,
                {
                    "query_embedding": str(query_embedding),
                    "limit": top_k
                }
            )
        
        rows = result.fetchall()
        
        chunks = [
            ChunkResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                filename=row.filename,
                content=row.content,
                distance=float(row.distance),
                chunk_index=row.chunk_index
            )
            for row in rows
        ]
        
        return chunks


async def generate_answer(query: str, chunks: List[ChunkResult]) -> str:
    """Generate answer using LLM with retrieved chunks."""
    semaphore = get_generation_semaphore()
    
    async with semaphore:
        # Build prompt with citations
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] From {chunk.filename} (chunk {chunk.chunk_index}):\n{chunk.content}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following context, answer the question. Include citations using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        model, tokenizer = get_llm_model()
        
        def _generate():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=settings.LLM_MAX_INPUT_LENGTH)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    do_sample=True,
                    top_p=settings.LLM_TOP_P,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part (after "Answer:")
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()
            
            return answer
        
        answer = await asyncio.to_thread(_generate)
        return answer


async def chat(query: str, top_k: int = 5, document_id: Optional[int] = None) -> Tuple[str, List[ChunkResult]]:
    """Perform RAG: retrieve chunks and generate answer."""
    # Retrieve relevant chunks
    chunks = await retrieve_chunks(query, top_k, document_id)
    
    if not chunks:
        return "No relevant information found in the documents.", []
    
    # Generate answer
    answer = await generate_answer(query, chunks)
    
    return answer, chunks
