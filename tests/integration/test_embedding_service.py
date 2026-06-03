"""Integration tests for EmbeddingService with the real SentenceTransformer model.

These tests exercise production components end-to-end — no mocks, no fakes.
They verify that the configured model loads, generates correctly-shaped
float32 embeddings, and handles batch input.  Exact embedding values are
intentionally not asserted; only structure and dimensions are checked.
"""

import numpy as np
import pytest

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService


@pytest.fixture(scope="module")
def real_service() -> EmbeddingService:
    """EmbeddingService backed by the real configured SentenceTransformer model.

    Module-scoped so the model is loaded once for the entire module, keeping
    total runtime reasonable.
    """
    return EmbeddingService(model_name=settings.EMBEDDING_MODEL)


@pytest.mark.integration
def test_real_model_loads_successfully(real_service: EmbeddingService) -> None:
    """EmbeddingService loads the configured model and reports a positive dimension."""
    dim = real_service.get_embedding_dimension()

    assert dim > 0, "Embedding dimension must be greater than zero"


@pytest.mark.integration
def test_real_model_generates_embedding(real_service: EmbeddingService) -> None:
    """embed_text returns a correctly-typed, correctly-sized vector."""
    sentence = "Integration tests verify that components work together."

    result = real_service.embed_text(sentence)
    expected_dim = real_service.get_embedding_dimension()

    assert isinstance(result, np.ndarray), "Result must be a numpy.ndarray"
    assert result.dtype == np.float32, "dtype must be float32"
    assert result.shape == (expected_dim,), (
        f"Vector length {result.shape[0]} must equal embedding dimension {expected_dim}"
    )


@pytest.mark.integration
def test_real_model_batch_embedding(real_service: EmbeddingService) -> None:
    """embed_texts returns one vector per input text with the correct shape."""
    texts = [
        "The first document.",
        "A second, different document.",
        "Yet another document for batch testing.",
    ]

    result = real_service.embed_texts(texts)
    expected_dim = real_service.get_embedding_dimension()

    assert isinstance(result, np.ndarray), "Result must be a numpy.ndarray"
    assert result.shape == (len(texts), expected_dim), (
        f"Expected shape ({len(texts)}, {expected_dim}), got {result.shape}"
    )
