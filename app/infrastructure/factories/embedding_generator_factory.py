"""Factory for embedding generator infrastructure."""

from app.adapters.embeddings import SentenceTransformerEmbeddingGenerator
from app.core.config import settings
from app.core.ports.embedder import Embedder


class EmbeddingGeneratorFactory:
    """Create embedding generator implementations for application wiring."""

    def create_embedding_generator(self) -> Embedder:
        """Instantiate the configured embedding generator.

        The provider is read from ``settings.EMBEDDING_PROVIDER``.  Only
        ``sentence-transformers`` is supported in this iteration; additional
        providers (openai, cohere, voyage-ai, …) can be added here by
        implementing a new adapter and registering it below.

        Returns:
            An :class:`~app.core.ports.embedder.Embedder`
            ready to use.

        Raises:
            ValueError: If the configured provider is not recognised.
        """
        provider = settings.EMBEDDING_PROVIDER.strip().lower()

        if provider == "sentence-transformers":
            return SentenceTransformerEmbeddingGenerator(
                model_name=settings.EMBEDDING_MODEL,
            )

        raise ValueError(
            f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER!r}. "
            "Supported providers: sentence-transformers"
        )


def create_embedding_generator() -> Embedder:
    """Convenience function: create the configured embedding generator."""
    return EmbeddingGeneratorFactory().create_embedding_generator()
