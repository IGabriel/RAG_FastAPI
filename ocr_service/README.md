# OCR Service

Minimal OCR microservice used by the main RAG service when a PDF has no text layer.

## API

- `GET /health` → `{"status":"ok"}`
- `POST /ocr` (multipart form field `file`) → `{"pages": [{"page": 1, "text": "..."}]}`

## Environment

- `OCR_LANG` (default: `chi_sim+eng`)
- `OCR_DPI` (default: `200`)
- `OCR_MAX_PAGES` (default: `50`)
