from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


app = FastAPI(title="RAG OCR Service", version="0.1.0")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")
OCR_DPI = _env_int("OCR_DPI", 200)
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 50)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.TemporaryDirectory(prefix="ocr-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / file.filename
        data = await file.read()
        input_path.write_bytes(data)

        pages: list[dict[str, Any]] = []

        if suffix == ".pdf":
            images = convert_from_path(str(input_path), dpi=OCR_DPI)
            if len(images) > OCR_MAX_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Too many pages for OCR: {len(images)} > {OCR_MAX_PAGES}",
                )

            for i, image in enumerate(images, start=1):
                text = pytesseract.image_to_string(image, lang=OCR_LANG)
                pages.append({"page": i, "text": (text or "").strip()})
        else:
            image = Image.open(input_path)
            text = pytesseract.image_to_string(image, lang=OCR_LANG)
            pages.append({"page": 1, "text": (text or "").strip()})

        return JSONResponse({"pages": pages, "lang": OCR_LANG, "dpi": OCR_DPI})
