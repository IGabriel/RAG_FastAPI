"""LangChain helpers for embeddings, vector store, and LLM."""
from __future__ import annotations

import asyncio
from typing import Optional, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import PGVector
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from app.config import settings


_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[PGVector] = None
_llm: Optional[HuggingFacePipeline] = None
_reranker: Optional[HuggingFaceCrossEncoder] = None

try:  # Optional, depends on langchain version
    from langchain.retrievers.self_query.base import SelfQueryRetriever as _SelfQueryRetriever
except Exception:  # pragma: no cover - optional dependency
    _SelfQueryRetriever = None

class _CrossEncoderRerankRetriever(BaseRetriever):
    """Retriever wrapper that reranks results with a cross-encoder."""

    def __init__(self, base_retriever: BaseRetriever, cross_encoder: HuggingFaceCrossEncoder, top_k: int):
        super().__init__()
        self.base_retriever = base_retriever
        self.cross_encoder = cross_encoder
        self.top_k = top_k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        if not docs:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


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


def get_reranker() -> Optional[HuggingFaceCrossEncoder]:
    """Get or create the cross-encoder model."""
    global _reranker
    if not settings.ENABLE_RERANKER:
        return None
    if _reranker is None:
        model_path = settings.RERANKER_MODEL_PATH or ""
        if not model_path:
            return None
        _reranker = HuggingFaceCrossEncoder(model_name=model_path)
    return _reranker


def _get_metadata_field_info() -> List[AttributeInfo]:
    return [
        AttributeInfo(
            name="document_id",
            description="Document id in the database",
            type="integer",
        ),
        AttributeInfo(
            name="filename",
            description="Original filename",
            type="string",
        ),
        AttributeInfo(
            name="chunk_index",
            description="Chunk index within a document",
            type="integer",
        ),
    ]


def get_retriever(top_k: int, document_id: Optional[int] = None):
    """Build a retriever with optional self-query and reranking."""
    vectorstore = get_vectorstore()
    search_kwargs = {"k": top_k}
    if document_id is not None:
        search_kwargs["filter"] = {"document_id": document_id}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if settings.USE_SELF_QUERY:
        if settings.USE_SELF_QUERY and _SelfQueryRetriever is not None:
            retriever = _SelfQueryRetriever.from_llm(
            llm=get_llm(),
            vectorstore=vectorstore,
            document_contents="Document chunk",
            metadata_field_info=_get_metadata_field_info(),
            search_kwargs=search_kwargs,
        )

    reranker = get_reranker()
    if reranker is not None:
        retriever = _CrossEncoderRerankRetriever(retriever, reranker, top_k)

    return retriever


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
