# Knowledge Base Chat - RAG Learning Project

A clean architecture Python project for building a Retrieval-Augmented Generation (RAG) application using FastAPI, FAISS vector store, and sentence transformers.

## Project Structure

```
knowledge-base-chat/
├── app/
│   ├── api/                 # API routes and endpoints
│   ├── core/                # Configuration and prompts
│   ├── ingestion/           # Document loading and processing
│   ├── vectorstore/         # Vector store implementations
│   ├── services/            # Business logic
│   └── main.py              # FastAPI application
├── data/
│   ├── raw/                 # Source documents
│   └── vector_store/        # Vector store indexes
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Add Documents
Place your documents in `data/raw/` directory.

### 5. Ingest Documents
```bash
python scripts/ingest_documents.py
```

## Running the Application

```bash
python app/main.py
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Components

- **Config**: Environment configuration management
- **Ingestion**: PDF loading and text extraction
- **Chunker**: Text splitting for embeddings
- **Embedder**: Sentence transformer integration
- **Vector Store**: FAISS-based similarity search
- **Retrieval Service**: RAG logic and context retrieval
- **API Routes**: FastAPI endpoints

## Development

This project follows clean architecture principles with:
- Separation of concerns
- Dependency injection
- Minimal coupling between modules
- OOP design patterns
