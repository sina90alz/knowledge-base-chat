"""Unit tests for the ChromaDB vector store adapter."""

import numpy as np

from app.adapters.vectorstores import ChromaVectorStore


def test_add_documents_and_find_relevant_documents(tmp_path):
    """Chroma adapter should satisfy the VectorStore port."""
    vector_store = ChromaVectorStore(
        dimension=3,
        store_path=tmp_path / "chroma",
        collection_name="test_documents",
    )

    vector_store.add_documents(
        ["alpha document", "beta document"],
        [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        [{"filename": "alpha.txt", "page_number": 2}, {"filename": "beta.txt"}],
    )

    documents, distances, metadata = vector_store.find_relevant_documents(
        [0.0, 0.0, 0.0],
        limit=1,
    )

    assert documents == ["alpha document"]
    assert distances == [0.0]
    assert metadata == [{"filename": "alpha.txt", "page": 2}]


def test_search_empty_store_returns_empty_results(tmp_path):
    """Empty Chroma stores should match FAISS empty-search behavior."""
    vector_store = ChromaVectorStore(
        dimension=3,
        store_path=tmp_path / "chroma",
        collection_name="test_documents",
    )

    assert vector_store.search(np.array([0.0, 0.0, 0.0]), k=5) == ([], [], [])
    assert len(vector_store) == 0


def test_persists_documents_between_instances(tmp_path):
    """The adapter should use Chroma's persistent client."""
    store_path = tmp_path / "chroma"
    first_store = ChromaVectorStore(
        dimension=3,
        store_path=store_path,
        collection_name="test_documents",
    )
    first_store.add_texts(
        ["persistent document"],
        np.array([[1.0, 2.0, 3.0]]),
        [{"source": "fixture"}],
    )

    second_store = ChromaVectorStore(
        dimension=3,
        store_path=store_path,
        collection_name="test_documents",
    )

    documents, distances, metadata = second_store.search(
        np.array([1.0, 2.0, 3.0]),
        k=1,
    )

    assert len(second_store) == 1
    assert documents == ["persistent document"]
    assert distances == [0.0]
    assert metadata == [{"source": "fixture"}]


def test_clear_removes_documents(tmp_path):
    """Clear should reset the collection while keeping the adapter usable."""
    vector_store = ChromaVectorStore(
        dimension=3,
        store_path=tmp_path / "chroma",
        collection_name="test_documents",
    )
    vector_store.add_texts(
        ["document"],
        np.array([[1.0, 2.0, 3.0]]),
        [{"source": "fixture"}],
    )

    vector_store.clear()

    assert len(vector_store) == 0
    assert vector_store.find_relevant_documents([1.0, 2.0, 3.0]) == ([], [], [])


def test_rejects_embedding_dimension_mismatch(tmp_path):
    """Input validation should fail before writing invalid Chroma records."""
    vector_store = ChromaVectorStore(
        dimension=3,
        store_path=tmp_path / "chroma",
        collection_name="test_documents",
    )

    try:
        vector_store.add_texts(["document"], np.array([[1.0, 2.0]]))
    except ValueError as exc:
        assert str(exc) == "Embedding dimension 2 does not match store dimension 3"
    else:
        raise AssertionError("Expected ValueError for invalid embedding dimension")
