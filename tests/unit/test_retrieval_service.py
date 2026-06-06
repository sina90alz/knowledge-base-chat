"""Unit tests for RetrievalService.

These tests demonstrate the improved testability after refactoring:
- Injected threshold parameters enable testing boundary conditions
- Fake vector store avoids filesystem dependencies
- Custom prompt formatters enable isolated testing
"""

import numpy as np
import pytest

from app.ingestion.embedder import EmbeddingService
from app.services.retrieval import RetrievalService


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _create_service(
    fake_embedding_model,
    fake_vector_store,
    similarity_threshold=1.2,
    max_chunks=5,
    prompt_formatter=None,
):
    """Create RetrievalService with test doubles."""
    embedding_service = EmbeddingService(model=fake_embedding_model)
    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=fake_vector_store,
        similarity_threshold=similarity_threshold,
        max_chunks=max_chunks,
        prompt_formatter=prompt_formatter,
    )


# ---------------------------------------------------------------------------
# Tests: Threshold Filtering
# ---------------------------------------------------------------------------


def test_filters_documents_above_threshold(fake_embedding_model, fake_vector_store):
    """Documents with distance above threshold should be filtered out."""
    # Setup: 3 documents with varying distances
    fake_vector_store.search_results = (
        ["doc1", "doc2", "doc3"],
        [0.5, 1.5, 2.0],  # Only first passes threshold of 1.2
        [{"source": "a"}, {"source": "b"}, {"source": "c"}],
    )
    service = _create_service(
        fake_embedding_model, fake_vector_store, similarity_threshold=1.2
    )

    # Execute
    docs, distances, metadata = service.retrieve_context("test query")

    # Verify: only document within threshold is returned
    assert len(docs) == 1
    assert docs[0] == "doc1"
    assert distances[0] == 0.5


def test_returns_all_documents_below_threshold(fake_embedding_model, fake_vector_store):
    """All documents below threshold should be included."""
    # Setup: all documents below threshold
    fake_vector_store.search_results = (
        ["doc1", "doc2", "doc3"],
        [0.3, 0.6, 0.9],
        [{}, {}, {}],
    )
    service = _create_service(
        fake_embedding_model, fake_vector_store, similarity_threshold=1.0
    )

    # Execute
    docs, distances, _ = service.retrieve_context("test query")

    # Verify: all documents returned
    assert len(docs) == 3
    assert distances == [0.3, 0.6, 0.9]


def test_empty_results_when_all_above_threshold(fake_embedding_model, fake_vector_store):
    """Should return empty results when all documents above threshold."""
    # Setup: all documents above threshold
    fake_vector_store.search_results = (
        ["doc1", "doc2"],
        [1.5, 2.0],
        [{}, {}],
    )
    service = _create_service(
        fake_embedding_model, fake_vector_store, similarity_threshold=1.0
    )

    # Execute
    docs, distances, metadata = service.retrieve_context("test query")

    # Verify: empty results
    assert len(docs) == 0
    assert len(distances) == 0
    assert len(metadata) == 0


# ---------------------------------------------------------------------------
# Tests: Max Chunks Limiting
# ---------------------------------------------------------------------------


def test_respects_max_chunks_parameter(fake_embedding_model, fake_vector_store):
    """Should limit results to max_chunks even when more are available."""
    # Setup: 5 documents all below threshold
    fake_vector_store.search_results = (
        ["doc1", "doc2", "doc3", "doc4", "doc5"],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [{}, {}, {}, {}, {}],
    )
    service = _create_service(
        fake_embedding_model,
        fake_vector_store,
        similarity_threshold=1.0,
        max_chunks=3,  # Limit to 3
    )

    # Execute
    docs, distances, _ = service.retrieve_context("test query")

    # Verify: only 3 returned
    assert len(docs) == 3
    assert docs == ["doc1", "doc2", "doc3"]


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


def test_deduplicates_identical_documents(fake_embedding_model, fake_vector_store):
    """Identical documents should be deduplicated."""
    # Setup: duplicate documents
    fake_vector_store.search_results = (
        ["same doc", "same doc", "different doc"],
        [0.1, 0.2, 0.3],
        [{}, {}, {}],
    )
    service = _create_service(fake_embedding_model, fake_vector_store)

    # Execute
    docs, _, _ = service.retrieve_context("test query")

    # Verify: duplicates removed
    assert len(docs) == 2
    assert "same doc" in docs
    assert "different doc" in docs


def test_deduplicates_by_source_and_chunk_position(
    fake_embedding_model, fake_vector_store
):
    """Documents with same source and chunk position should be deduplicated."""
    # Setup: same source and chunk position
    fake_vector_store.search_results = (
        ["doc1", "doc2"],
        [0.1, 0.2],
        [
            {"source": "file.pdf", "chunk_start_word": 100, "chunk_end_word": 200},
            {"source": "file.pdf", "chunk_start_word": 100, "chunk_end_word": 200},
        ],
    )
    service = _create_service(fake_embedding_model, fake_vector_store)

    # Execute
    docs, _, _ = service.retrieve_context("test query")

    # Verify: duplicate removed based on metadata
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# Tests: Quality Assessment
# ---------------------------------------------------------------------------


def test_quality_good_when_best_distance_below_threshold(
    fake_embedding_model, fake_vector_store
):
    """Quality should be GOOD when best distance is below threshold."""
    service = _create_service(
        fake_embedding_model, fake_vector_store, similarity_threshold=1.0
    )

    # Best distance below threshold
    quality = service.get_retrieval_quality(raw_distances=[0.5, 0.8], filtered_count=2)

    assert quality == "GOOD"

def test_quality_rejected_when_no_results(fake_embedding_model, fake_vector_store):
    """Quality should be REJECTED when no documents returned."""
    service = _create_service(fake_embedding_model, fake_vector_store)

    quality = service.get_retrieval_quality(raw_distances=[1.5], filtered_count=0)

    assert quality == "REJECTED"


# ---------------------------------------------------------------------------
# Tests: Prompt Generation
# ---------------------------------------------------------------------------


def test_uses_custom_prompt_formatter(fake_embedding_model, fake_vector_store):
    """Should use injected prompt formatter when provided."""
    # Setup: custom formatter
    def custom_formatter(context, query):
        return f"CUSTOM: {query} | {context}"

    service = _create_service(
        fake_embedding_model,
        fake_vector_store,
        prompt_formatter=custom_formatter,
    )

    # Execute
    prompt = service.generate_prompt("test query", "test context")

    # Verify: custom format used
    assert prompt == "CUSTOM: test query | test context"

# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


def test_raises_error_for_empty_query(fake_embedding_model, fake_vector_store):
    """Should raise ValueError for empty query string."""
    service = _create_service(fake_embedding_model, fake_vector_store)

    with pytest.raises(ValueError, match="non-empty string"):
        service.retrieve_context("")


def test_handles_empty_vector_store(fake_embedding_model, fake_vector_store):
    """Should return empty results when vector store has no documents."""
    # Setup: empty store
    fake_vector_store.search_results = ([], [], [])
    service = _create_service(fake_embedding_model, fake_vector_store)

    # Execute
    docs, distances, metadata = service.retrieve_context("test query")

    # Verify: empty results
    assert docs == []
    assert distances == []
    assert metadata == []


# ---------------------------------------------------------------------------
# Tests: Context Formatting
# ---------------------------------------------------------------------------


def test_format_context_includes_all_metadata(fake_embedding_model, fake_vector_store):
    """Formatted context should include source and page metadata."""
    service = _create_service(fake_embedding_model, fake_vector_store)

    docs = ["First document", "Second document"]
    metadata = [
        {"filename": "doc1.pdf", "page": 5},
        {"filename": "doc2.pdf", "page": 10},
    ]

    context = service.format_context(docs, metadata)

    assert "doc1.pdf" in context
    assert "Page: 5" in context
    assert "doc2.pdf" in context
    assert "Page: 10" in context
    assert "First document" in context
    assert "Second document" in context


def test_format_context_truncates_when_too_long(fake_embedding_model, fake_vector_store):
    """Long context should be truncated to max_length."""
    service = _create_service(fake_embedding_model, fake_vector_store)

    # Create very long document
    long_doc = "x" * 5000
    context = service.format_context([long_doc], max_length=1000)

    assert len(context) <= 1003  # 1000 + "..."
    assert context.endswith("...")
