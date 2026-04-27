"""Embedding generation service."""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize embedding service.

        Args:
            model_name: Name of the sentence transformer model

        Raises:
            ValueError: If model cannot be loaded
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise ValueError(f"Cannot load model {model_name}: {e}") from e

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array with shape (embedding_dim,)

        Raises:
            ValueError: If text is empty
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently in batch.

        Args:
            texts: List of texts to embed

        Returns:
            Matrix of embedding vectors with shape (num_texts, embedding_dim)

        Raises:
            ValueError: If texts list is empty or contains invalid items
        """
        if not texts or len(texts) == 0:
            raise ValueError("Texts list cannot be empty")

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All items in texts list must be strings")

        try:
            logger.debug(f"Embedding {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Embed multiple chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Matrix of embedding vectors

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks or len(chunks) == 0:
            raise ValueError("Chunks list cannot be empty")

        try:
            chunk_texts = [chunk.content for chunk in chunks]
            logger.info(f"Embedding {len(chunks)} chunks")
            embeddings = self.embed_texts(chunk_texts)
            logger.info(f"Successfully embedded {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        return self.embedding_dim

    def get_model_name(self) -> str:
        """Get loaded model name.

        Returns:
            Name of the embedding model
        """
        return self.model_name
