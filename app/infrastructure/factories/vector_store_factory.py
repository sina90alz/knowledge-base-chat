"""Factory for vector store infrastructure."""

from app.adapters.vectorstores import ChromaVectorStore, FaissVectorStore
from app.core.config import settings
from app.core.ports import VectorStore
from app.ingestion.embedder import EmbeddingService


class VectorStoreFactory:
    """Create vector store implementations for application wiring."""

    def create_vector_store(self, embedding_dimension: int | None = None) -> VectorStore:
        """Create the configured vector store implementation."""
        provider = settings.VECTOR_STORE_PROVIDER.strip().lower()

        dimension = embedding_dimension
        if dimension is None and provider in {"faiss", "chroma"}:
            embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
            dimension = embedding_service.get_embedding_dimension()

        if provider == "faiss":
            return FaissVectorStore(
                dimension=dimension,
                store_path=settings.VECTOR_STORE_PATH,
            )

        if provider == "chroma":
            return ChromaVectorStore(
                dimension=dimension,
                store_path=settings.VECTOR_STORE_PATH,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )

        raise ValueError(f"Unsupported vector store provider: {settings.VECTOR_STORE_PROVIDER}")


def create_vector_store(embedding_dimension: int | None = None) -> VectorStore:
    """Create the configured vector store implementation."""
    return VectorStoreFactory().create_vector_store(embedding_dimension)
