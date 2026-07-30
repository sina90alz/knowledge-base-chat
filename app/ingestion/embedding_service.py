"""Embedding generation service."""

import logging
from typing import List

import numpy as np

from app.core.ports.embedder import Embedder
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Orchestrate embedding generation for the application.

    Delegates all embedding computation to the injected
    :class:`~app.core.ports.embedder.Embedder` so the
    service remains decoupled from any specific provider (SentenceTransformer,
    OpenAI, Cohere, etc.).

    EmbeddingService is responsible for:

    * input validation that reflects business rules
    * chunk-aware helpers (embed_chunks)
    * logging at the workflow level

    It is *not* responsible for model loading, caching, or provider-specific APIs.
    """

    def __init__(self, generator: Embedder) -> None:
        """Initialize the embedding service.

        Args:
            generator: Provider-specific embedding generator that implements
                the Embedder port.
        """

        self._generator = generator

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Configured model identifier."""
        return self._generator.get_model_name()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a float32 numpy array with shape (embedding_dim,).

        Raises:
            ValueError: If *text* is empty or not a string.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        return self._generator.embed_text(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently in a single batch.

        Args:
            texts: List of texts to embed.

        Returns:
            Embedding matrix with shape (num_texts, embedding_dim).

        Raises:
            ValueError: If *texts* is empty or contains non-string items.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All items in texts list must be strings")
        logger.debug("Embedding %d texts", len(texts))
        return self._generator.embed_texts(texts)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Embed a list of document chunks.

        Args:
            chunks: List of :class:`~app.ingestion.chunker.Chunk` objects.

        Returns:
            Embedding matrix with shape (len(chunks), embedding_dim).

        Raises:
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        chunk_texts = [chunk.content for chunk in chunks]
        logger.info("Embedding %d chunks", len(chunks))
        embeddings = self._generator.embed_texts(chunk_texts)
        logger.info("Successfully embedded %d chunks", len(chunks))
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Integer embedding dimension.
        """
        return self._generator.get_embedding_dimension()

    def get_model_name(self) -> str:
        """Return the configured model name.

        Returns:
            Model identifier string.
        """
        return self._generator.get_model_name()
