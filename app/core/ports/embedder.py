"""Port for embedding generation."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Embedder(ABC):
    """Provider-independent interface for generating text embeddings.

    The application depends only on this abstraction.  Concrete adapters
    (SentenceTransformer, OpenAI, Cohere, …) implement this interface so that
    the embedding provider can be swapped without touching any business logic.
    """

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding vector for a single text string.

        Args:
            text: Non-empty text to embed.

        Returns:
            Embedding vector as a float32 numpy array with shape (embedding_dim,).
        """
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embedding vectors for multiple texts in a single batch.

        Args:
            texts: Non-empty list of text strings to embed.

        Returns:
            Embedding matrix with shape (len(texts), embedding_dim) and
            dtype float32.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of the generated embedding vectors."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the configured model identifier."""
        raise NotImplementedError
