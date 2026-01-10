# RAG_FastAPI

A production-ready Async RAG (Retrieval-Augmented Generation) service built with FastAPI, PostgreSQL with pgvector, and local LLM inference.

## Features

- 🚀 **Async FastAPI** - High-performance async API compatible with Windows/Linux/macOS
- 📦 **Document Management** - Upload, list, delete, and reindex documents (PDF, TXT, MD)
- 🔍 **Vector Search** - Fast similarity search using pgvector with cosine distance
- 🤖 **Local LLM** - Generate answers using Qwen2.5-0.5B-Instruct model
- 📝 **Smart Chunking** - Markdown heading-aware text splitting
- 🖼️ **OCR Support** - External OCR service integration for scanned PDFs
- 🎯 **Background Processing** - Async document indexing with configurable concurrency
- 💾 **Metadata Persistence** - Embedding model and dimension validation

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│   FastAPI    │─────▶│  PostgreSQL │
│             │      │              │      │  + pgvector │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ├─────▶ Document Storage (local disk)
                            ├─────▶ Embedding Model (sentence-transformers)
                            └─────▶ LLM (Qwen2.5-0.5B-Instruct)
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- At least 4GB RAM (for models)
- 10GB disk space (for models and storage)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/IGabriel/RAG_FastAPI.git
cd RAG_FastAPI
```

### 2. Start PostgreSQL with pgvector

```bash
docker-compose up -d
```

This starts a PostgreSQL 16 container with pgvector extension.

### 3. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download the LLM Model

Download Qwen2.5-0.5B-Instruct model:

```bash
mkdir -p models
# Option 1: Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# Option 2: Manual download from https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
# Download all files and place them in models/Qwen2.5-0.5B-Instruct/
```

### 5. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env if needed (defaults should work for local development)
```

### 6. Run the Service

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m app.main
```

The API will be available at `http://localhost:8000`

### 7. Access API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Document Management

#### Upload Document
```http
POST /documents
Content-Type: multipart/form-data

file: <file>
```

**Supported formats:** PDF, TXT, MD (up to 100MB)

**Response:**
```json
{
  "document_id": 1,
  "filename": "example.pdf",
  "status": "pending",
  "message": "Document uploaded successfully. Processing started."
}
```

#### List Documents
```http
GET /documents?page=1&per_page=10
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": 1,
      "filename": "example.pdf",
      "status": "completed",
      "chunk_count": 42,
      "created_at": "2024-01-10T12:00:00",
      "updated_at": "2024-01-10T12:01:00"
    }
  ],
  "total": 100,
  "page": 1,
  "per_page": 10
}
```

**Status values:**
- `pending` - Document uploaded, waiting for processing
- `processing` - Currently being indexed
- `completed` - Successfully indexed and ready for queries
- `failed` - Processing failed (check logs)

#### Get Document Details
```http
GET /documents/{doc_id}
```

#### Delete Document
```http
DELETE /documents/{doc_id}
```

Deletes the document record, all associated chunks from the database, and removes the document directory from disk.

#### Reindex Document
```http
POST /documents/{doc_id}/reindex
```

Re-processes an existing document (useful after changing chunking parameters).

### Retrieval and Chat

#### Retrieve Chunks (Debug Endpoint)
```http
POST /retrieve
Content-Type: application/json

{
  "query": "What is RAG?",
  "top_k": 5,
  "document_id": null  // Optional: filter by document
}
```

**Response:**
```json
{
  "query": "What is RAG?",
  "chunks": [
    {
      "chunk_id": 123,
      "document_id": 1,
      "filename": "example.pdf",
      "content": "RAG stands for Retrieval-Augmented Generation...",
      "distance": 0.234,
      "chunk_index": 5
    }
  ]
}
```

#### Chat with Documents
```http
POST /chat
Content-Type: application/json

{
  "query": "Explain the main concepts",
  "top_k": 5,
  "document_id": null  // Optional: filter by document
}
```

**Response:**
```json
{
  "query": "Explain the main concepts",
  "answer": "Based on the documents, the main concepts are... [1][2]",
  "sources": [
    {
      "chunk_id": 123,
      "document_id": 1,
      "filename": "example.pdf",
      "content": "...",
      "distance": 0.234,
      "chunk_index": 5
    }
  ]
}
```

The answer includes inline citations like [1], [2] that correspond to the sources array.

### Health Check
```http
GET /health
```

## Configuration

All configuration is done via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection URL |
| `STORAGE_DIR` | `./storage` | Directory for uploaded documents |
| `MAX_UPLOAD_SIZE` | `104857600` | Max upload size in bytes (100MB) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Sentence-transformers model |
| `LLM_MODEL_PATH` | `./models/Qwen2.5-0.5B-Instruct` | Path to LLM model |
| `INDEXING_CONCURRENCY` | `1` | Max concurrent indexing tasks |
| `GENERATION_CONCURRENCY` | `1` | Max concurrent generation tasks |
| `CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `DEFAULT_TOP_K` | `5` | Default number of chunks to retrieve |
| `OCR_SERVICE_URL` | ` ` | External OCR service URL (optional) |

## External OCR Service Integration

For scanned PDFs without extractable text, the service can call an external OCR service.

### OCR API Contract

**Endpoint:** Configured via `OCR_SERVICE_URL` environment variable

**Request:**
```http
POST {OCR_SERVICE_URL}
Content-Type: multipart/form-data

file: <pdf-file>
```

**Expected Response:**
```json
{
  "pages": [
    {
      "page": 1,
      "text": "Text content from page 1..."
    },
    {
      "page": 2,
      "text": "Text content from page 2..."
    }
  ]
}
```

**Behavior:**
1. When a PDF is uploaded, the service first tries to extract text using `pypdf`
2. If no text is extracted AND `OCR_SERVICE_URL` is configured, it calls the OCR service
3. The OCR service response is parsed and text from all pages is concatenated
4. If OCR fails or is not configured, the document will fail with "No text extracted"

### Example OCR Service Implementation

You can use services like:
- Tesseract OCR via REST API
- Google Cloud Vision API
- Azure Computer Vision
- AWS Textract
- Custom OCR service

Example using a simple OCR service wrapper:
```python
# Simple Flask OCR service example
from flask import Flask, request, jsonify
import pytesseract
from pdf2image import convert_from_bytes

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr():
    pdf_file = request.files['file']
    images = convert_from_bytes(pdf_file.read())
    
    pages = []
    for i, image in enumerate(images, 1):
        text = pytesseract.image_to_string(image)
        pages.append({"page": i, "text": text})
    
    return jsonify({"pages": pages})
```

## Technical Details

### Database Schema

The service automatically creates three tables at startup:

1. **embedding_metadata** - Stores embedding model info and dimension
   - Validates model consistency across restarts
   - Prevents dimension mismatches

2. **documents** - Stores document metadata
   - `id`, `filename`, `file_path`, `status`, `error_message`
   - Timestamps: `created_at`, `updated_at`

3. **chunks** - Stores text chunks with embeddings
   - `id`, `document_id`, `chunk_index`, `content`
   - `embedding` - pgvector column with dynamic dimension
   - Foreign key to documents with CASCADE delete

### Indexing Pipeline

1. **Upload** - Stream file to disk (protects against memory issues)
2. **Extract** - Text extraction (pypdf or OCR fallback)
3. **Chunk** - Markdown heading-aware splitting with overlap
4. **Embed** - Generate embeddings using sentence-transformers
5. **Store** - Save chunks and embeddings to PostgreSQL
6. **Update** - Mark document as completed/failed

All steps run asynchronously in background with concurrency control.

### Chunking Strategy

The service uses a hybrid chunking approach:

1. **Markdown-aware** - First splits by headings (# ## ### etc.)
2. **Size-based** - Splits large sections into smaller chunks
3. **Overlapping** - Maintains context between chunks
4. **Word-boundary** - Splits on word boundaries, not mid-word

This preserves document structure while ensuring chunks fit model context windows.

### Vector Search

Uses pgvector's `ivfflat` index with cosine distance (`<=>` operator):
- Fast approximate nearest neighbor search
- Configurable list parameter (default: 100)
- Good trade-off between speed and accuracy

### Async Architecture

- **Non-blocking I/O** - All database and file operations are async
- **Thread offloading** - CPU-bound tasks (embedding, generation) use `asyncio.to_thread`
- **Concurrency limits** - Semaphores protect CPU and memory
- **Background tasks** - Document indexing runs in background

## Development

### Project Structure

```
RAG_FastAPI/
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI app and routes
│   ├── config.py       # Configuration management
│   ├── schemas.py      # Pydantic models
│   ├── db.py           # Database connection and schema
│   ├── ingest.py       # Document ingestion and chunking
│   └── rag.py          # Retrieval and generation
├── storage/            # Uploaded documents (created at runtime)
├── models/             # LLM models (not in git)
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # PostgreSQL with pgvector
├── .env.example        # Environment variables template
└── README.md          # This file
```

### Adding New File Types

To support additional file types, add extraction logic to `app/ingest.py`:

```python
def extract_text_from_docx(docx_path: Path) -> str:
    # Your extraction logic
    pass

async def extract_text_from_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    
    if suffix == '.docx':
        return await asyncio.to_thread(extract_text_from_docx, file_path)
    # ... existing cases
```

Then update the validation in `app/main.py`:

```python
if suffix not in ['.pdf', '.txt', '.md', '.docx']:
    raise HTTPException(...)
```

### Testing

Basic testing with curl:

```bash
# Upload document
curl -X POST "http://localhost:8000/documents" \
  -F "file=@test.pdf"

# List documents
curl "http://localhost:8000/documents"

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document", "top_k": 5}'
```

## Troubleshooting

### Model Download Issues

If model download fails:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Use mirror (if needed)
export HF_ENDPOINT=https://hf-mirror.com
```

### Database Connection Issues

Check PostgreSQL is running:
```bash
docker-compose ps
docker-compose logs postgres
```

### Out of Memory

Reduce concurrency in `.env`:
```env
INDEXING_CONCURRENCY=1
GENERATION_CONCURRENCY=1
```

Or use a smaller embedding model:
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Slow Processing

- Enable GPU support for transformers (install `torch` with CUDA)
- Increase `INDEXING_CONCURRENCY` if you have multiple CPU cores
- Use a smaller LLM model
- Reduce `CHUNK_SIZE` to create fewer chunks

## Performance Notes

- **First request is slow** - Models are loaded on first use
- **Embedding batch size** - Automatically handled by sentence-transformers
- **LLM inference** - ~1-2s per query on CPU (faster with GPU)
- **Vector search** - Sub-second even with thousands of chunks

## Security Considerations

- **File uploads** - Limited to 100MB to prevent DoS
- **Input validation** - All inputs validated by Pydantic
- **SQL injection** - Protected by parameterized queries
- **Path traversal** - Uses pathlib for safe path handling
- **OCR service** - Use HTTPS and authentication in production

## Production Deployment

For production:

1. **Use environment variables** for all secrets
2. **Enable HTTPS** - Use reverse proxy (nginx/traefik)
3. **Set up monitoring** - Track API latency and error rates
4. **Use proper PostgreSQL** - Not the Docker Compose setup
5. **Scale horizontally** - Run multiple API instances
6. **Add authentication** - Implement OAuth2/JWT
7. **Set up logging** - Use structured logging
8. **Enable CORS** - Configure allowed origins
9. **Use GPU** - For faster inference
10. **Backup database** - Regular PostgreSQL backups

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions:
- GitHub Issues: https://github.com/IGabriel/RAG_FastAPI/issues
- Documentation: This README

## Acknowledgments

- FastAPI - https://fastapi.tiangolo.com/
- pgvector - https://github.com/pgvector/pgvector
- sentence-transformers - https://www.sbert.net/
- Qwen - https://github.com/QwenLM/Qwen
