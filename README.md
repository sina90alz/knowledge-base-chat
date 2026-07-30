# Knowledge Base Chat

FastAPI RAG service for local PDF and TXT documents.

## Current Features

- Load documents from `data/raw/`
- Split documents into overlapping chunks
- Generate embeddings with `sentence-transformers`
- Store vectors in FAISS or Chroma
- Query documents through `POST /api/query`
- Return answer, context, sources, distances, metadata, and retrieval status
- Reject empty retrievals
- Verify answers against retrieved context when enabled
- Log every query attempt to SQLite audit storage
- Search and inspect audit records through `/api/audit`
- Run threshold-based evaluation and generate a markdown report
- Build and run as a Docker image

## Flow

```text
PDF/TXT files
  -> load
  -> chunk
  -> embed
  -> FAISS or Chroma
  -> retrieve
  -> prompt LLM
  -> verify answer
  -> audit result
```

## Main Modules

- `app/api/routes.py` - query and health endpoints
- `app/api/audit.py` - audit endpoints
- `app/services/retrieval.py` - retrieval, filtering, context, prompt creation
- `app/services/verification.py` - answer support check
- `app/services/audit_service.py` - SQLite audit persistence
- `app/ingestion/` - loading, chunking, embedding
- `app/adapters/vectorstores/` - FAISS and Chroma implementations
- `app/infrastructure/` - factories and application wiring
- `scripts/ingest_documents.py` - ingestion pipeline
- `scripts/evaluate.py` - evaluation pipeline
- `scripts/evaluation/` - evaluation cases, metrics, threshold sweep, report output

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set the LLM provider in `.env`:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=tinyllama
OLLAMA_BASE_URL=http://localhost:11434/api/generate
```

or:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## Key Settings

```text
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ENABLE_ANSWER_VERIFICATION=true
SIMILARITY_THRESHOLD=1.2
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
VECTOR_STORE_PROVIDER=faiss
VECTOR_STORE_PATH=data/vector_store
CHROMA_COLLECTION_NAME=documents
AUDIT_DB_PATH=data/audit/audit.db
```

Use Chroma instead of FAISS:

```text
VECTOR_STORE_PROVIDER=chroma
```

## Ingest Documents

Add files:

```text
data/raw/
```

Run ingestion:

```bash
python scripts/ingest_documents.py
```

## Run API

```bash
python app/main.py
```

Open:

```text
http://localhost:8000/docs
```

Health check:

```text
GET /api/health
```

## Query API

Endpoint:

```text
POST /api/query
```

Request:

```json
{
  "query": "What information is available in the knowledge base?",
  "k": 5
}
```

Response fields:

- `answer`
- `context`
- `retrieved_docs`
- `distances`
- `metadata`
- `sources`
- `retrieval_status`

Retrieval status values:

- `GOOD` - usable context
- `WEAK` - context returned, but best distance is above threshold
- `REJECTED` - no usable context

## Audit API

List recent records:

```text
GET /api/audit?limit=20&offset=0
```

Search records:

```text
GET /api/audit/search?status=SUCCESS&retrieval_status=GOOD&limit=20
```

Get one record:

```text
GET /api/audit/{id}
```

Audit records include:

- query
- answer
- model
- retrieval status
- top distance
- retrieved chunk count
- response time
- verification status
- error message

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

Output:

```text
reports/evaluation_report.md
```

Evaluation covers:

- retrieval quality
- answer support
- weak and rejected retrieval behavior
- threshold comparison for `0.8`, `1.0`, `1.2`, `1.5`, `1.8`, `2.0`

## Tests

Run all tests:

```bash
pytest
```

Run unit tests:

```bash
pytest tests/unit -v
```

Run integration tests:

```bash
pytest tests/integration -v
```

## Docker

Build:

```bash
docker build -t knowledge-base-chat .
```

Run:

```bash
docker run --rm -p 8000:8000 -v "%cd%/data:/app/data" knowledge-base-chat
```

Default Docker settings:

- `LLM_PROVIDER=ollama`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434/api/generate`
- `VECTOR_STORE_PATH=/app/data/vector_store`
- `AUDIT_DB_PATH=/app/data/audit/audit.db`

## CI

- Unit tests run on push and pull request
- Docker image builds on push and pull request
- Integration tests run weekly and on manual dispatch
