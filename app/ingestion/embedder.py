"""Embedding generation service."""

import logging
from typing import Any, List, Protocol

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Minimal interface required from an embedding model."""

    def encode(self, sentences: str | List[str], **kwargs: Any) -> Any:
        """Return embeddings for one string or a batch of strings."""

    def get_sentence_embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""


class ModelProvider(Protocol):
    """Minimal interface required from an embedding model provider."""

    def get_model(self) -> EmbeddingModel:
        """Return a loaded embedding model."""


class EmbeddingModelProvider:
    """Acquire and load sentence-transformer embedding models.

    This provider keeps download/cache management separate from embedding
    generation so tests can bypass HuggingFace and SentenceTransformer by
    injecting a fake model or provider into EmbeddingService.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the provider with the configured model name.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self._model: EmbeddingModel | None = None

    def get_model(self) -> EmbeddingModel:
        """Return a loaded model, loading it on first access."""
        if self._model is None:
            self._model = self.load_model()
        return self._model

    def load_model(self) -> EmbeddingModel:
        """Load the configured model from cache when possible."""
        logger.info(f"Loading embedding model: {self.model_name}")
        model_path = self.get_cached_model_path(self.model_name)
        return SentenceTransformer(model_path or self.model_name)

    @staticmethod
    def get_cached_model_path(model_name: str) -> str | None:
        """Return a local cached model path when available."""
        repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"

        try:
            return snapshot_download(repo_id=repo_id, local_files_only=True)
        except Exception:
            logger.info("Model not found in local cache; attempting normal model load")
            return None


class EmbeddingService:
    """Generate embeddings using sentence transformers.

    Dependency injection is supported through the ``model`` argument so unit
    tests can provide a fake object with ``encode`` and
    ``get_sentence_embedding_dimension`` methods. That allows embedding
    generation to be tested without internet access, HuggingFace downloads, or
    local model cache dependencies.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: EmbeddingModel | None = None,
        model_provider: ModelProvider | None = None,
    ) -> None:
        """Initialize embedding service.

        The constructor stores dependencies only. The default model is loaded
        lazily on first use, preserving production behavior while avoiding
        expensive constructor side effects. Tests can inject ``model`` to avoid
        loading SentenceTransformer entirely.

        Args:
            model_name: Name of the sentence transformer model
            model: Optional preloaded model or fake model for tests
            model_provider: Optional provider for custom model loading

        Raises:
            ValueError: If both model and model_provider are provided
        """
        if model is not None and model_provider is not None:
            raise ValueError("Provide either model or model_provider, not both")

        self.model_name = model_name
        self._model = model
        self._model_provider = model_provider or EmbeddingModelProvider(model_name)
        self._embedding_dim: int | None = None

    @property
    def model(self) -> EmbeddingModel:
        """Return the model, loading it lazily when needed."""
        return self._get_model()

    @model.setter
    def model(self, value: EmbeddingModel) -> None:
        """Set a preloaded model instance."""
        self._model = value
        self._embedding_dim = None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension, resolving the model if necessary."""
        return self.get_embedding_dimension()

    @embedding_dim.setter
    def embedding_dim(self, value: int) -> None:
        """Set a known embedding dimension."""
        self._embedding_dim = value

    @staticmethod
    def _get_cached_model_path(model_name: str) -> str | None:
        """Return a local cached model path when available."""
        return EmbeddingModelProvider.get_cached_model_path(model_name)

    def _get_model(self) -> EmbeddingModel:
        """Return the injected model or lazily load one from the provider."""
        if self._model is None:
            try:
                self._model = self._model_provider.get_model()
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Model loaded successfully. Embedding dimension: {self._embedding_dim}"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise ValueError(f"Cannot load model {self.model_name}: {e}") from e
        return self._model

    def _get_embedding_dimension(self) -> int:
        """Resolve and cache the embedding dimension."""
        if self._embedding_dim is None:
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        return self._embedding_dim

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
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
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
        return self._get_embedding_dimension()

    def get_model_name(self) -> str:
        """Get loaded model name.

        Returns:
            Name of the embedding model
        """
        return self.model_name
