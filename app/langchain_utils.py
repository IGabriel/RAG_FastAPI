"""LangChain helpers for embeddings, vector store, and LLM."""
from __future__ import annotations

import asyncio
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import PGVector
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from app.config import settings


_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[PGVector] = None
_llm: Optional[HuggingFacePipeline] = None


def _get_sync_database_url() -> str:
    """Return a psycopg2-compatible connection string for LangChain PGVector."""
    if settings.DATABASE_URL_SYNC:
        return settings.DATABASE_URL_SYNC
    if "+asyncpg" in settings.DATABASE_URL:
        return settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
    return settings.DATABASE_URL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the LangChain embeddings object."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> PGVector:
    """Get or create the PGVector vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PGVector(
            connection_string=_get_sync_database_url(),
            embedding_function=get_embeddings(),
            collection_name=settings.VECTOR_COLLECTION,
        )
    return _vectorstore


def get_llm() -> HuggingFacePipeline:
    """Get or create the local LLM pipeline."""
    global _llm
    if _llm is None:
        tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        _llm = HuggingFacePipeline(pipeline=text_gen)
    return _llm


async def similarity_search_with_score(query: str, top_k: int, document_id: Optional[int] = None):
    """Run similarity search in a thread to avoid blocking the event loop."""
    vectorstore = get_vectorstore()
    filter_meta = {"document_id": document_id} if document_id is not None else None
    return await asyncio.to_thread(
        vectorstore.similarity_search_with_score,
        query,
        k=top_k,
        filter=filter_meta,
    )
