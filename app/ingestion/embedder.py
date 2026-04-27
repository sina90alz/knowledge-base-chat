"""Embedding generation service."""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate embeddings using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding service.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Matrix of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        return self.embedding_dim
