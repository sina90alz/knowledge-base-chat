# Knowledge Base Chat

A compact Retrieval-Augmented Generation (RAG) app for chatting with local documents. It uses FastAPI, FAISS, sentence-transformer embeddings, OpenAI or Ollama for generation, and an evaluation script to check retrieval quality and answer grounding.

## Highlights

- Ingests PDF/TXT documents from `data/raw/`
- Chunks, embeds, and stores documents in a local FAISS vector store
- Exposes a `/api/query` endpoint for document-grounded Q&A
- Filters weak retrieval results using a similarity threshold
- Verifies whether generated answers are supported by retrieved context
- Runs threshold sweeps and writes an evaluation report

## Architecture

```text
Documents -> Chunking -> Embeddings -> FAISS
                                      |
User query -> Retrieval -> Prompt -> LLM -> Verification -> Answer
```

Key modules:

- `app/ingestion/` - document loading, chunking, embeddings
- `app/vectorstore/` - FAISS persistence and search
- `app/services/retrieval.py` - retrieval filtering and context formatting
- `app/services/verification.py` - grounding verification
- `app/api/routes.py` - FastAPI endpoints
- `scripts/evaluate.py` - retrieval and generation evaluation

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

For macOS/Linux, activate with:

```bash
source venv/bin/activate
```

Set the LLM provider in `.env`:

```text
LLM_PROVIDER=ollama
# or
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

## Ingest Documents

Add PDF or TXT files to:

```text
data/raw/
```

Then build the vector store:

```bash
python scripts/ingest_documents.py
```

## Run The API

```bash
python app/main.py
```

Open:

```text
http://localhost:8000/docs
```

Example request:

```json
{
  "query": "What information is available in the knowledge base?",
  "k": 5
}
```

The response includes the answer, retrieved context, source metadata, distances, and retrieval status.

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

The evaluation workflow checks:

- retrieval distances
- retrieved document counts
- `GOOD`, `WEAK`, or `REJECTED` retrieval status
- supported vs unsupported answers
- threshold performance across multiple similarity cutoffs

It writes a report to:

```text
reports/evaluation_report.md
```

## Why This Project Matters

This is not only a basic RAG demo. It includes production-minded pieces that matter in real AI systems:

- retrieval quality checks before generation
- fallback behavior when context is missing or weak
- answer verification to reduce unsupported responses
- source metadata preserved through the pipeline
- repeatable evaluation for tuning retrieval thresholds

## Tech Stack

- Python
- FastAPI
- FAISS
- sentence-transformers
- OpenAI SDK
- Ollama
- pypdf
