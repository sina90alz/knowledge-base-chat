"""SentenceTransformer implementation of the Embedder port."""

import logging
from typing import Any, List, Protocol

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from app.core.ports.embedder import Embedder

logger = logging.getLogger(__name__)


class ModelProvider(Protocol):
    """Minimal interface required from an embedding model provider."""

    def get_model(self) -> Any:
        """Return a loaded embedding model."""


class SentenceTransformerEmbeddingGenerator(Embedder):
    """Generate embeddings using the sentence-transformers library.

    Supports lazy model loading and local HuggingFace cache resolution to
    avoid unnecessary network requests.

    For unit tests a pre-built fake model (or a fake ModelProvider) can be
    injected so that no SentenceTransformer download or import side-effects
    are triggered.  This mirrors the ``model`` / ``model_provider`` approach
    used in the previous EmbeddingService design.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: Any | None = None,
        model_provider: ModelProvider | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model_name: Name of the sentence-transformer model.
            model: Optional pre-loaded or fake model.  When supplied the model
                is used directly and no loading is attempted.
            model_provider: Optional provider used to load the model on first
                use.  Useful for testing lazy-loading behaviour.

        Raises:
            ValueError: If both ``model`` and ``model_provider`` are supplied.
        """
        if model is not None and model_provider is not None:
            raise ValueError("Provide either model or model_provider, not both")

        self._model_name = model_name
        self._model = model
        self._model_provider = model_provider
        self._embedding_dim: int | None = None

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        embedding = self._get_model().encode(text, convert_to_numpy=True)
        return np.asarray(embedding, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts in batch."""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All items in texts list must be strings")
        logger.debug("Embedding %d texts", len(texts))
        embeddings = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        """Return embedding vector dimension."""
        if self._embedding_dim is None:
            self._embedding_dim = self._get_model().get_sentence_embedding_dimension()
        return self._embedding_dim

    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Return the loaded model, loading it lazily on first access."""
        if self._model is None:
            provider = self._model_provider or self._create_default_provider()
            try:
                self._model = provider.get_model()
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(
                    "Embedding model loaded. Dimension: %d",
                    self._embedding_dim,
                )
            except Exception as exc:
                logger.error("Failed to load embedding model %s: %s", self._model_name, exc)
                raise ValueError(
                    f"Cannot load model {self._model_name!r}: {exc}"
                ) from exc
        return self._model

    def _create_default_provider(self) -> "_DefaultModelProvider":
        """Create the default SentenceTransformer-backed provider."""
        return _DefaultModelProvider(self._model_name)


class _DefaultModelProvider:
    """Load a SentenceTransformer model, preferring the local HuggingFace cache."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def get_model(self) -> SentenceTransformer:
        logger.info("Loading embedding model: %s", self._model_name)
        model_path = self._get_cached_model_path(self._model_name)
        return SentenceTransformer(model_path or self._model_name)

    @staticmethod
    def _get_cached_model_path(model_name: str) -> str | None:
        repo_id = (
            model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        )
        try:
            return snapshot_download(repo_id=repo_id, local_files_only=True)
        except Exception:
            logger.info("Model not found in local cache; attempting normal model load")
            return None
