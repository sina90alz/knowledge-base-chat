"""Factory for vector store infrastructure."""

from app.adapters.vectorstores import FaissVectorStore
from app.core.config import settings
from app.core.ports import VectorStore
from app.ingestion.embedder import EmbeddingService


class VectorStoreFactory:
    """Create vector store implementations for application wiring."""

    def create_vector_store(self, embedding_dimension: int | None = None) -> VectorStore:
        """Create the configured vector store implementation."""
        provider = settings.VECTOR_STORE_PROVIDER.strip().lower()

        if provider == "faiss":
            dimension = embedding_dimension
            if dimension is None:
                embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
                dimension = embedding_service.get_embedding_dimension()

            return FaissVectorStore(
                dimension=dimension,
                store_path=settings.VECTOR_STORE_PATH,
            )

        raise ValueError(f"Unsupported vector store provider: {settings.VECTOR_STORE_PROVIDER}")


def create_vector_store(embedding_dimension: int | None = None) -> VectorStore:
    """Create the configured vector store implementation."""
    return VectorStoreFactory().create_vector_store(embedding_dimension)
