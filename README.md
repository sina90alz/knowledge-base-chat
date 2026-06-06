# Knowledge Base Chat

A compact Retrieval-Augmented Generation (RAG) app for querying local documents. It uses FastAPI, FAISS, sentence-transformer embeddings, OpenAI or Ollama for generation, and evaluation tooling for retrieval and grounding.

## Highlights

- Ingests PDF/TXT documents from `data/raw/`
- Chunks documents, generates embeddings, and stores them in FAISS
- Serves a `/api/query` endpoint for document-grounded Q&A
- Uses a similarity threshold to reject weak retrievals
- Optionally verifies answers against the retrieved context
- Includes an evaluation script and report generation

## Architecture

```text
Documents -> Chunking -> Embeddings -> FAISS
                                      |
User query -> Retrieval -> Prompt -> LLM -> Verification -> Answer
```

Key modules:

- `app/ingestion/` - loading, chunking, embeddings
- `app/vectorstore/` - FAISS storage and search
- `app/services/retrieval.py` - retrieval and context preparation
- `app/services/verification.py` - answer grounding checks
- `app/api/routes.py` - FastAPI chat routes
- `scripts/evaluate.py` - evaluation workflow

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` and configure your provider:

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

Then run:

```bash
python scripts/ingest_documents.py
```

## Run the API

```bash
python app/main.py
```

Open the Swagger UI at:

```text
http://localhost:8000/docs
```

Query payload example:

```json
{
  "query": "What information is available in the knowledge base?",
  "k": 5
}
```

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

The evaluation generates a markdown report in:

```text
reports/evaluation_report.md
```

## Notes

- Default query batch size is `k=5`
- `ENABLE_ANSWER_VERIFICATION` controls whether answers are verified
- `SIMILARITY_THRESHOLD` controls when retrieval is marked weak or rejected

## Tech Stack

- Python
- FastAPI
- FAISS
- sentence-transformers
- OpenAI
- Ollama
- pypdf
