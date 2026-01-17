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

### System Packages (Ubuntu/Debian)

Install the baseline system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  curl \
  ca-certificates
```

If `pip install -r requirements.txt` needs to build native wheels (varies by platform/Python), install build tools too:

```bash
sudo apt-get install -y build-essential python3-dev
```

Docker (choose one):

```bash
# Option A (Ubuntu packages)
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER

# Option B (official Docker Engine) - see: https://docs.docker.com/engine/install/
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/IGabriel/RAG_FastAPI.git
cd RAG_FastAPI
```

### 2. Start PostgreSQL with pgvector

```bash
docker compose up -d
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

# Option 3: from modelscope
# modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir models/Qwen2.5-0.5B-Instruct
```

If your host cannot access Hugging Face, prefer the ModelScope option above, and keep `LLM_MODEL_PATH` pointing to the local folder:

```bash
LLM_MODEL_PATH=./models/Qwen2.5-0.5B-Instruct
```

### 4.1 Download the Embedding Model (Recommended)

This service uses `sentence-transformers` and will try to download the embedding model on first startup. If your server cannot access Hugging Face (common on some cloud hosts), download it to a local folder and point `EMBEDDING_MODEL` to that path.

```bash
mkdir -p models

# If you are in mainland China, this often helps:
export HF_ENDPOINT=https://hf-mirror.com

# Download the embedding model to a local directory
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5

# Option 2: ModelScope (often works better on mainland China servers)
# modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir models/bge-small-zh-v1.5
```

Then set in `.env`:

```bash
EMBEDDING_MODEL=./models/bge-small-zh-v1.5

# If the server cannot access huggingface.co at runtime, force offline mode:
HF_HUB_OFFLINE=1
# (Optional)
TRANSFORMERS_OFFLINE=1
```

If you still cannot download from the server, do the download on a machine with internet access, then copy `models/bge-small-zh-v1.5/` to the server.

### 5. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env if needed (defaults should work for local development)
```

#### LangChain + PGVector (Offline)

- The service now uses LangChain's PGVector store. It requires a **sync** PostgreSQL URL for LangChain:

```bash
DATABASE_URL_SYNC=postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb
VECTOR_COLLECTION=rag_documents
```

- Keep models fully local (ModelScope download) and enable offline mode if needed:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### 6. Run the Service

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m app.main
```

The API will be available at `http://localhost:8000`

## VS Code Remote Development (Remote-SSH / Remote Tunnels)

This project works well with VS Code remote development. The recommended workflow is:

- Use **Remote-SSH** to edit and run the code on the server.
- Use **port forwarding** to access the FastAPI UI (e.g. `/docs`) from your local browser.
- Optionally, use **Remote Tunnels** when you cannot (or do not want to) SSH directly.

### Option A: Remote-SSH (Recommended)

#### 1) Configure SSH host alias (Windows example)

Edit your local SSH config (Windows): `C:\Users\<you>\.ssh\config` and add a stable alias.

```ssh-config
Host rag-fastapi
    HostName <YOUR_SERVER_IP>
    User <YOUR_SSH_USER>
    Port 22
    IdentityFile C:/Users/<you>/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

If the server IP changes later, you only need to update `HostName`.

#### 2) Connect from VS Code

- Install the VS Code extension: **Remote - SSH**
- Open the Command Palette and run: `Remote-SSH: Connect to Host...`
- Select your host alias (e.g. `rag-fastapi`)

After connecting, the VS Code window should indicate `SSH: <host>` in the bottom-left corner.

#### 3) Forward port 8000 to your local machine

If your FastAPI server listens on `127.0.0.1:8000` on the remote machine (common for dev), your local browser cannot reach it directly. Use port forwarding:

- Open the **Ports** panel in VS Code (Remote Explorer → Ports)
- Click **Forward a Port** and enter `8000`
- Open the forwarded **Local Address** (e.g. `http://127.0.0.1:8000/docs`)

If forwarding to local `8000` fails, map to another local port (example `18000`):

- Create the forwarding for remote `8000`
- Right-click the entry → **Change Local Port** → `18000`
- Then open `http://127.0.0.1:18000/docs`

##### Alternative: manual SSH port forward

You can also forward ports using an SSH command from your local machine:

```bash
ssh -N -L 18000:127.0.0.1:8000 rag-fastapi
```

Then open `http://127.0.0.1:18000/health` or `http://127.0.0.1:18000/docs`.

#### 4) Trigger endpoints without a browser

If you prefer not to use a browser (or you are only debugging), run `curl` on the remote machine:

```bash
curl -v http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"hello","top_k":3}'
```

### Option B: VS Code Remote Tunnels

Remote Tunnels let you connect to a machine without opening inbound SSH to the Internet. This is useful when:

- You are behind NAT/firewall and cannot SSH into the machine directly.
- The public IP changes frequently.
- You want an easy “sign-in to connect” experience.

#### 1) Start a tunnel on the remote machine

On the remote machine, start VS Code (or use the VS Code CLI) and run:

- Command Palette: `Remote Tunnels: Turn on Remote Tunnel...`
- Sign in (GitHub or Microsoft account)
- Give the machine a name

Keep the tunnel running while you work.

#### 2) Connect from your local VS Code

On your local machine:

- Command Palette: `Remote Tunnels: Connect to Remote Tunnel...`
- Choose the machine name you created

#### 3) Access FastAPI via forwarded ports

Once connected via a tunnel, use the same **Ports** panel workflow:

- Forward remote `8000` to a local port
- Open `/docs` or `/health` in your local browser

### Troubleshooting

- **Browser shows `ERR_CONNECTION_REFUSED` for `http://127.0.0.1:8000`**
  - `127.0.0.1` always means “this machine”. If FastAPI runs on the remote server, you must use VS Code port forwarding (or SSH `-L`) to access it from your local browser.

- **VS Code shows: “Unable to forward localhost:8000…”**
  - Try forwarding to a different local port (e.g. `18000`).
  - Check whether another VS Code window already forwarded the same port.
  - Use manual SSH forwarding as a fallback: `ssh -N -L 18000:127.0.0.1:8000 <host>`.

- **Uvicorn shows “Application startup complete” but the terminal does not return**
  - This is normal: the server is running and waiting for requests.

- **`--reload` + debugger breakpoints not hitting reliably**
  - Use the VS Code launch configuration that enables subprocess debugging (see `.vscode/launch.json`).

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

**Supported formats:** PDF, TXT, MD, DOCX (up to 100MB)

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

## OCR (Scanned PDFs)

If a PDF has no selectable text (image-only/scanned PDF), `pypdf` may extract an empty string and indexing will fail unless OCR is enabled.

This repo includes a minimal OCR service (FastAPI + Tesseract) that implements the expected `OCR_SERVICE_URL` contract.

- Start OCR service:
  - `docker compose up -d ocr`
- Set `.env`:
  - `OCR_SERVICE_URL=http://127.0.0.1:9001/ocr`
- Reindex a document:
  - `curl -sS -X POST http://127.0.0.1:8000/documents/<DOC_ID>/reindex`

Notes:
- OCR quality depends on PDF quality and language pack. Default is `chi_sim+eng`.
- You can change OCR settings via docker-compose environment: `OCR_LANG`, `OCR_DPI`, `OCR_MAX_PAGES`.

### If `docker compose build ocr` fails (apt timeouts)

On some servers (especially with restricted outbound network), the OCR image build may fail while running `apt-get`.
In that case, use a closer Debian mirror for the OCR build:

```bash
export APT_MIRROR_URL=http://mirrors.aliyun.com/debian
export APT_SECURITY_URL=http://mirrors.aliyun.com/debian-security
docker compose build ocr
docker compose up -d ocr
```
Common symptoms include:

- `Could not connect to debian.map.fastlydns.net:80 ... connection timed out`
- `E: Unable to locate package tesseract-ocr` (usually because `apt-get update` could not fetch indexes)

If you prefer not to export variables each time, you can also put these in the local `.env` file in the repo root
(Docker Compose reads it automatically) and then run `docker compose build ocr`.

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
