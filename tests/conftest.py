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


@pytest.fixture
def fake_embedding_model() -> FakeEmbeddingModel:
    """Provide an injected embedding model with deterministic vectors."""
    return FakeEmbeddingModel()


@pytest.fixture
def fake_model_provider(fake_embedding_model: FakeEmbeddingModel) -> FakeModelProvider:
    """Provide a provider that returns the fake model and records load calls."""
    return FakeModelProvider(fake_embedding_model)
