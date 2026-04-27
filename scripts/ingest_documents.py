"""Script to ingest documents into the vector store."""

import logging
from pathlib import Path

from app.core.config import settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_documents():
    """Load documents, chunk them, embed them, and store in vector DB."""
    logger.info("Starting document ingestion...")

    # Initialize services
    logger.info(f"Initializing embedding service with model: {settings.EMBEDDING_MODEL}")
    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)

    logger.info(f"Initializing vector store: {settings.VECTOR_STORE_PATH}")
    vector_store = FAISSVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        store_path=settings.VECTOR_STORE_PATH,
    )

    chunker = TextChunker(
        chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP
    )

    # Load documents
    logger.info(f"Loading documents from: {settings.RAW_DATA_DIR}")
    pdf_files = DocumentLoader.load_directory(settings.RAW_DATA_DIR, pattern="*.pdf")
    txt_files = DocumentLoader.load_directory(settings.RAW_DATA_DIR, pattern="*.txt")

    all_files = pdf_files + txt_files
    if not all_files:
        logger.warning(f"No documents found in {settings.RAW_DATA_DIR}")
        return

    logger.info(f"Found {len(all_files)} documents")

    documents = []
    for file_path in all_files:
        try:
            logger.info(f"Processing: {file_path}")
            if file_path.endswith(".pdf"):
                text = DocumentLoader.load_pdf(file_path)
            else:
                text = DocumentLoader.load_txt(file_path)
            documents.append(text)
            logger.info(f"  - Loaded {len(text)} characters")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    if not documents:
        logger.error("No documents loaded successfully")
        return

    # Chunk documents
    logger.info("Chunking documents...")
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Embed chunks
    logger.info("Generating embeddings...")
    embeddings = embedding_service.embed_texts(chunks)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    # Store in vector DB
    logger.info("Storing embeddings in vector store...")
    vector_store.add_texts(chunks, embeddings)
    logger.info("Document ingestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()
