from typing import Any

import numpy as np
import pytest


class FakeEmbeddingModel:
    """Fake embedding model for tests that should not load SentenceTransformer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | list[str], dict[str, Any]]] = []

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append((sentences, kwargs))
        if isinstance(sentences, str):
            return np.array([1.0, 2.0, 3.0], dtype=np.float64)
        return np.array(
            [
                [float(index), float(index + 1), float(index + 2)]
                for index, _ in enumerate(sentences)
            ],
            dtype=np.float64,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 3


class FakeModelProvider:
    """Fake model provider for tests that need to inspect lazy loading."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.load_count = 0

    def get_model(self) -> Any:
        self.load_count += 1
        return self.model


class FakeVectorStore:
    """Fake vector store for testing RetrievalService without FAISS."""

    def __init__(
        self,
        search_results: tuple[list[str], list[float], list[dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize fake vector store.
        
        Args:
            search_results: Tuple of (documents, distances, metadata) to return from search.
                If None, returns empty results.
        """
        self.search_results = search_results or ([], [], [])
        self.search_calls: list[tuple[np.ndarray, int]] = []

    def find_relevant_documents(
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> tuple[list[str], list[float], list[dict[str, Any]]]:
        """Record retrieval call and return predefined results."""
        self.search_calls.append((query_embedding, limit))
        return self.search_results

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> tuple[list[str], list[float], list[dict[str, Any]]]:
        """Compatibility wrapper for older tests and scripts."""
        return self.find_relevant_documents(query_embedding, limit=k)

    def get_stats(self) -> dict[str, Any]:
        """Return fake statistics."""
        return {
            "total_vectors": len(self.search_results[0]),
            "embedding_dimension": 3,
            "store_path": "fake/path",
            "index_file_exists": True,
        }


@pytest.fixture
def fake_embedding_model() -> FakeEmbeddingModel:
    """Provide an injected embedding model with deterministic vectors."""
    return FakeEmbeddingModel()


@pytest.fixture
def fake_model_provider(fake_embedding_model: FakeEmbeddingModel) -> FakeModelProvider:
    """Provide a provider that returns the fake model and records load calls."""
    return FakeModelProvider(fake_embedding_model)


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """Provide a fake vector store for testing retrieval service."""
    return FakeVectorStore()
