"""Unit tests for vector store factory wiring."""

from app.adapters.vectorstores import FaissVectorStore
from app.core.config import settings
from app.infrastructure.factories import create_vector_store


def test_create_vector_store_returns_faiss_for_configured_provider(monkeypatch, tmp_path):
    """The factory should create the FAISS adapter for the faiss provider."""
    monkeypatch.setattr(settings, "VECTOR_STORE_PROVIDER", "faiss")
    monkeypatch.setattr(settings, "VECTOR_STORE_PATH", tmp_path / "vector_store")

    vector_store = create_vector_store(embedding_dimension=3)

    assert isinstance(vector_store, FaissVectorStore)
    assert vector_store.get_stats()["embedding_dimension"] == 3


def test_create_vector_store_accepts_case_and_whitespace(monkeypatch, tmp_path):
    """Provider normalization should avoid fragile environment formatting."""
    monkeypatch.setattr(settings, "VECTOR_STORE_PROVIDER", " FAISS ")
    monkeypatch.setattr(settings, "VECTOR_STORE_PATH", tmp_path / "vector_store")

    vector_store = create_vector_store(embedding_dimension=3)

    assert isinstance(vector_store, FaissVectorStore)


def test_create_vector_store_rejects_unknown_provider(monkeypatch, tmp_path):
    """Unknown providers should fail with a descriptive error."""
    monkeypatch.setattr(settings, "VECTOR_STORE_PROVIDER", "unknown")
    monkeypatch.setattr(settings, "VECTOR_STORE_PATH", tmp_path / "vector_store")

    try:
        create_vector_store(embedding_dimension=3)
    except ValueError as exc:
        assert str(exc) == "Unsupported vector store provider: unknown"
    else:
        raise AssertionError("Expected ValueError for unsupported vector store provider")


def test_unknown_provider_fails_before_loading_embedding_model(monkeypatch):
    """Unsupported providers should not trigger unrelated model loading."""
    monkeypatch.setattr(settings, "VECTOR_STORE_PROVIDER", "unknown")

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("EmbeddingService should not be created")

    monkeypatch.setattr(
        "app.infrastructure.factories.vector_store_factory.EmbeddingService",
        fail_if_loaded,
    )

    try:
        create_vector_store()
    except ValueError as exc:
        assert str(exc) == "Unsupported vector store provider: unknown"
    else:
        raise AssertionError("Expected ValueError for unsupported vector store provider")
