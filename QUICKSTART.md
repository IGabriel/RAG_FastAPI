# Quick Start Guide

## Setup (5 minutes)

1. **Start PostgreSQL with pgvector:**
   ```bash
   docker-compose up -d
   ```

2. **Install Python dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the LLM model:**
   ```bash
   # Using huggingface-cli (recommended)
   pip install huggingface-hub
   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
   
   # Or download manually from:
   # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for local dev)
   ```

5. **Start the service:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Quick API Test

### 1. Upload a document
```bash
# Create a test document
echo "# Machine Learning
Machine learning is a subset of artificial intelligence.

## Deep Learning
Deep learning uses neural networks with multiple layers." > test.md

# Upload it
curl -X POST "http://localhost:8000/documents" \
  -F "file=@test.md"

# Response: {"document_id": 1, "filename": "test.md", "status": "pending", ...}
```

### 2. Check document status
```bash
# Wait a few seconds for processing, then check
curl "http://localhost:8000/documents/1"

# Response: {"document_id": 1, "status": "completed", "chunk_count": 2, ...}
```

### 3. List all documents
```bash
curl "http://localhost:8000/documents?page=1&per_page=10"
```

### 4. Retrieve chunks (debug)
```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "top_k": 3
  }'

# Response: {"query": "...", "chunks": [{"content": "...", "distance": 0.23, ...}]}
```

### 5. Chat with documents
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain machine learning and deep learning",
    "top_k": 5
  }'

# Response: {"query": "...", "answer": "Based on the documents... [1][2]", "sources": [...]}
```

### 6. Reindex a document
```bash
curl -X POST "http://localhost:8000/documents/1/reindex"
```

### 7. Delete a document
```bash
curl -X DELETE "http://localhost:8000/documents/1"
```

## Common Issues

### "No module named 'pydantic_settings'"
```bash
pip install pydantic-settings
```

### "Connection refused" to PostgreSQL
```bash
docker-compose ps  # Check if postgres is running
docker-compose logs postgres  # Check logs
```

### "Model not found" error
Make sure you've downloaded the Qwen model to the correct directory:
```bash
ls -la models/Qwen2.5-0.5B-Instruct/
# Should see: config.json, model.safetensors, tokenizer files, etc.
```

### Out of memory
Reduce concurrency in .env:
```
INDEXING_CONCURRENCY=1
GENERATION_CONCURRENCY=1
```

## Development Tips

### Enable debug logging
```python
# In app/db.py, change:
_engine = create_async_engine(settings.DATABASE_URL, echo=True)
```

### Watch logs
```bash
# Run with uvicorn logs visible
uvicorn app.main:app --reload --log-level debug
```

### Test OCR integration
Set `OCR_SERVICE_URL` in .env to your OCR service endpoint, then upload a scanned PDF.

### Adjust chunking
Modify in .env:
```
CHUNK_SIZE=1000  # Larger chunks
CHUNK_OVERLAP=100  # More overlap
```

## Next Steps

- Add authentication (OAuth2/JWT)
- Set up monitoring and logging
- Deploy with Docker
- Scale with multiple workers
- Add more file types
- Implement caching for embeddings
- Add rate limiting
- Set up CI/CD

Enjoy your RAG service! 🚀
