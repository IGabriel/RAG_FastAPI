"""LangChain document loaders with OCR fallback."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List

import aiohttp
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import PyPDFLoader

from app.config import settings


@dataclass
class OCRPdfLoader(BaseLoader):
    """PDF loader with OCR fallback using external service (sync interface)."""

    file_path: str

    def load(self) -> List[Document]:
        pdf_loader = PyPDFLoader(self.file_path)
        docs = pdf_loader.load()
        text = "\n".join(d.page_content for d in docs if d.page_content).strip()
        if text:
            return docs

        if settings.OCR_SERVICE_URL:
            ocr_text = asyncio.run(_ocr_fallback(Path(self.file_path)))
            if ocr_text:
                return [Document(page_content=ocr_text, metadata={"source": self.file_path})]

        return docs


async def _ocr_fallback(pdf_path: Path) -> str:
    """Use external OCR service for scanned PDFs."""
    try:
        async with aiohttp.ClientSession() as session:
            with open(pdf_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=pdf_path.name)

                async with session.post(
                    settings.OCR_SERVICE_URL,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        pages = result.get("pages", [])
                        text = "\n".join(page.get("text", "") for page in pages)
                        return text.strip()
                    print(f"OCR service returned status {response.status}")
                    return ""
    except Exception as e:
        print(f"OCR service error: {e}")
        return ""
