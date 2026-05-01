"""Script to ingest documents into the vector store.

This module implements the complete Phase 1 pipeline:
1. Load documents from data/raw directory
2. Chunk documents with metadata preservation
3. Generate embeddings for chunks
4. Store vectors in FAISS index
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_documents() -> None:
    """Load documents, chunk them, embed them, and store in vector DB.

    Pipeline steps:
    1. Initialize services (embedder, vector store, chunker)
    2. Load documents from data/raw directory
    3. Chunk documents into overlapping segments
    4. Generate embeddings for all chunks
    5. Store vectors and metadata in FAISS index
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Document Ingestion Pipeline Started")
    logger.info("=" * 80)

    # Step 1: Initialize services
    logger.info("\n[STEP 1] Initializing services...")
    try:
        logger.info(f"  Embedding model: {settings.EMBEDDING_MODEL}")
        embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
        embedding_dim = embedding_service.get_embedding_dimension()
        logger.info(f"  Embedding dimension: {embedding_dim}")

        logger.info(f"  Vector store path: {settings.VECTOR_STORE_PATH}")
        vector_store = FAISSVectorStore(
            dimension=embedding_dim,
            store_path=settings.VECTOR_STORE_PATH,
        )
        store_stats = vector_store.get_stats()
        logger.info(f"  Vector store stats: {store_stats}")

        chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP
        )
        logger.info(
            f"  Chunker configured: chunk_size={settings.CHUNK_SIZE} words, overlap={settings.CHUNK_OVERLAP} words"
        )
        logger.info("Services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    # Step 2: Load documents
    logger.info("\n[STEP 2] Loading documents...")
    try:
        logger.info(f"  Source directory: {settings.RAW_DATA_DIR}")
        documents = DocumentLoader.load_directory(settings.RAW_DATA_DIR, pattern="*.*")

        if not documents:
            logger.warning(f"No documents found in {settings.RAW_DATA_DIR}")
            logger.info("Please add PDF or TXT files to data/raw/ directory")
            return

        # Calculate total text size
        total_chars = sum(len(doc) for doc in documents)
        logger.info(f"Loaded {len(documents)} documents ({total_chars:,} characters total)")

        for doc in documents:
            logger.debug(
                f"  - {doc.metadata.get('filename')}: "
                f"{len(doc):,} chars, "
                f"{doc.metadata.get('page_count', 'N/A')} pages"
            )

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

    # Step 3: Chunk documents
    logger.info("\n[STEP 3] Chunking documents...")
    try:
        chunks = chunker.chunk_documents(documents)

        if not chunks:
            logger.error(" No chunks created from documents")
            return

        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        total_chunk_words = sum(chunk.word_count() for chunk in chunks)

        logger.info(
            f"Created {len(chunks)} chunks "
            f"({total_chunk_chars:,} characters, {total_chunk_words:,} words)"
        )

        # Log chunk statistics
        chunk_sizes = [chunk.word_count() for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        logger.debug(f"  Average chunk size: {avg_chunk_size:.1f} words")
        logger.debug(f"  Min chunk size: {min(chunk_sizes)} words")
        logger.debug(f"  Max chunk size: {max(chunk_sizes)} words")

    except Exception as e:
        logger.error(f"✗ Error chunking documents: {e}")
        raise

    # Step 4: Generate embeddings
    logger.info("\n[STEP 4] Generating embeddings...")
    try:
        logger.info(f"  Embedding {len(chunks)} chunks...")
        embeddings = embedding_service.embed_chunks(chunks)

        logger.info(f" Generated embeddings with shape: {embeddings.shape}")
        logger.debug(f"  Embedding dtype: {embeddings.dtype}")
        logger.debug(f"  Embedding min/max: {embeddings.min():.4f}/{embeddings.max():.4f}")

    except Exception as e:
        logger.error(f" Error generating embeddings: {e}")
        raise

    # Step 5: Store in vector DB
    logger.info("\n[STEP 5] Storing vectors in FAISS index...")
    try:
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_metadata = [chunk.metadata for chunk in chunks]

        logger.info(f"  Adding {len(chunks)} vectors to FAISS index...")
        vector_store.add_texts(chunk_texts, embeddings, chunk_metadata)

        logger.info(" Vectors stored successfully")

        # Final stats
        stats = vector_store.get_stats()
        logger.info(f"\nFinal Vector Store Stats:")
        logger.info(f"  Total vectors: {stats['total_vectors']}")
        logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"  Store path: {stats['store_path']}")

    except Exception as e:
        logger.error(f" Error storing vectors: {e}")
        raise

    # Success summary
    logger.info("\n" + "=" * 80)
    logger.info(" PHASE 1: Document Ingestion Pipeline Completed Successfully!")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Documents loaded: {len(documents)}")
    logger.info(f"  Chunks created: {len(chunks)}")
    logger.info(f"  Vectors stored: {len(embeddings)}")
    logger.info(f"  Storage location: {settings.VECTOR_STORE_PATH}")
    logger.info("\nNext steps:")
    logger.info("  1. Start the API: python app/main.py")
    logger.info("  2. Query the RAG system via http://localhost:8000/docs")


if __name__ == "__main__":
    try:
        ingest_documents()
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}")
        sys.exit(1)
