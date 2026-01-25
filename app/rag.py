"""RAG retrieval and generation using LangChain."""
import asyncio
from typing import List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA

from app.config import settings
from app.langchain_utils import get_llm, get_retriever, similarity_search_with_score
from app.schemas import ChunkResult


_generation_semaphore: Optional[asyncio.Semaphore] = None


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
    """Retrieve relevant chunks using LangChain PGVector similarity search."""
    results = await similarity_search_with_score(query, top_k, document_id)

    chunks: List[ChunkResult] = []
    for idx, (doc, score) in enumerate(results):
        metadata = doc.metadata or {}
        chunks.append(
            ChunkResult(
                chunk_id=int(metadata.get("chunk_id", idx)),
                document_id=int(metadata.get("document_id", -1)),
                filename=str(metadata.get("filename", "")),
                content=doc.page_content,
                distance=float(score),
                chunk_index=int(metadata.get("chunk_index", idx)),
            )
        )

    return chunks


async def generate_answer(query: str, chunks: List[ChunkResult]) -> str:
    """Generate answer using local LLM with retrieved chunks."""
    semaphore = get_generation_semaphore()

    async with semaphore:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] From {chunk.filename} (chunk {chunk.chunk_index}):\n{chunk.content}\n"
            )
        context = "\n".join(context_parts)

        prompt = PromptTemplate.from_template(
            """
You are a helpful assistant. Use only the provided context to answer.
Include citations like [1], [2] after statements that use the context.

Context:
{context}

Question: {question}

Answer:
""".strip()
        )

        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        answer = await asyncio.to_thread(chain.invoke, {"context": context, "question": query})
        return answer.strip()


async def chat(query: str, top_k: int = 5, document_id: Optional[int] = None) -> Tuple[str, List[ChunkResult]]:
    """Perform RAG: retrieve chunks and generate answer."""
    retriever = get_retriever(top_k, document_id)
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = await asyncio.to_thread(qa_chain.invoke, {"query": query})
    docs = result.get("source_documents", [])

    # Build ChunkResult list from docs (score not available here)
    chunks: List[ChunkResult] = []
    for idx, doc in enumerate(docs):
        metadata = doc.metadata or {}
        chunks.append(
            ChunkResult(
                chunk_id=int(metadata.get("chunk_id", idx)),
                document_id=int(metadata.get("document_id", -1)),
                filename=str(metadata.get("filename", "")),
                content=doc.page_content,
                distance=0.0,
                chunk_index=int(metadata.get("chunk_index", idx)),
            )
        )

    if not chunks:
        prompt = PromptTemplate.from_template(
            """
You are a helpful assistant. Answer the user's question directly.

Question: {question}

Answer:
""".strip()
        )
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        answer = await asyncio.to_thread(chain.invoke, {"question": query})
        return answer.strip(), []

    answer = result.get("result", "").strip()
    return answer, chunks
