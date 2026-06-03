"""Unit tests for EmbeddingService.

All tests use injected fakes so no SentenceTransformer, HuggingFace model
download, or internet access is required.
"""

import numpy as np
import pytest

from app.ingestion.embedder import EmbeddingService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
SAMPLE_TEXTS = ["first sentence", "second sentence", "third sentence"]


def _service(fake_embedding_model):
    """Return an EmbeddingService pre-loaded with the fake model."""
    return EmbeddingService(model=fake_embedding_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_text_returns_embedding(fake_embedding_model) -> None:
    """embed_text returns a non-empty float32 numpy array."""
    service = _service(fake_embedding_model)

    result = service.embed_text(SAMPLE_TEXT)

    assert isinstance(result, np.ndarray), "Result must be a numpy.ndarray"
    assert result.dtype == np.float32, "dtype must be float32"
    assert result.size > 0, "Embedding must not be empty"


def test_embed_texts_returns_multiple_embeddings(fake_embedding_model) -> None:
    """embed_texts returns exactly one vector per input text."""
    service = _service(fake_embedding_model)

    result = service.embed_texts(SAMPLE_TEXTS)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(SAMPLE_TEXTS), (
        "Number of returned vectors must equal number of input texts"
    )


def test_embedding_dimension_matches_model(fake_embedding_model) -> None:
    """get_embedding_dimension reflects the dimension reported by the model."""
    service = _service(fake_embedding_model)

    dim = service.get_embedding_dimension()

    assert dim == fake_embedding_model.get_sentence_embedding_dimension()


def test_get_model_name(fake_embedding_model) -> None:
    """get_model_name returns the name supplied at construction time."""
    model_name = "my-custom-model"
    service = EmbeddingService(model_name=model_name, model=fake_embedding_model)

    assert service.get_model_name() == model_name


def test_injected_model_is_used(fake_embedding_model) -> None:
    """EmbeddingService forwards embed_text calls to the injected model."""
    service = _service(fake_embedding_model)

    service.embed_text("hello")

    assert fake_embedding_model.calls == [
        ("hello", {"convert_to_numpy": True})
    ], "encode() must be called exactly once with the supplied text"


def test_lazy_loading_occurs_only_on_first_use(
    fake_model_provider,
    fake_embedding_model,
) -> None:
    """Model provider must not be invoked during __init__; only on first use."""
    service = EmbeddingService(model_provider=fake_model_provider)

    # Provider must not have been called yet
    assert fake_model_provider.load_count == 0, (
        "Model provider must not be called during __init__"
    )

    # First embedding request triggers a single load
    service.embed_text("first call")
    assert fake_model_provider.load_count == 1, (
        "Model provider must be called exactly once on first use"
    )

    # Subsequent requests must reuse the already-loaded model
    service.embed_text("second call")
    service.embed_text("third call")
    assert fake_model_provider.load_count == 1, (
        "Model provider must not be called again after the first load"
    )


def test_empty_text_raises_value_error(fake_embedding_model) -> None:
    """embed_text raises ValueError when given an empty string."""
    service = _service(fake_embedding_model)

    with pytest.raises(ValueError):
        service.embed_text("")
