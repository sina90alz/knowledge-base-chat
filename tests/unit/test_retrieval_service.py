"""Unit tests for RetrievalService.

These tests demonstrate the improved testability after refactoring:
- Injected threshold parameters enable testing boundary conditions
- Fake vector store avoids filesystem dependencies
- Custom prompt formatters enable isolated testing
"""

import numpy as np
import pytest

from app.ingestion.embedding_service import EmbeddingService
from app.models import RetrievalResult
from app.services.retrieval import RetrievalService


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _create_service(
    fake_embedding_generator,
    fake_vector_store,
    similarity_threshold=1.2,
    max_chunks=5,
    prompt_formatter=None,
):
    """Create RetrievalService with test doubles."""
    embedding_service = EmbeddingService(generator=fake_embedding_generator)
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


def test_filters_documents_above_threshold(fake_embedding_generator, fake_vector_store):
    """Documents with distance above threshold should be filtered out."""
    # Setup: 3 documents with varying distances
    fake_vector_store.search_results = (
        ["doc1", "doc2", "doc3"],
        [0.5, 1.5, 2.0],  # Only first passes threshold of 1.2
        [{"source": "a"}, {"source": "b"}, {"source": "c"}],
    )
    service = _create_service(
        fake_embedding_generator, fake_vector_store, similarity_threshold=1.2
    )

    # Execute
    result = service.retrieve_context("test query")

    # Verify: only document within threshold is returned
    assert isinstance(result, RetrievalResult)
    assert len(result.documents) == 1
    assert result.documents[0] == "doc1"
    assert result.distances[0] == 0.5
    assert result.metadata == [{"source": "a"}]
    assert result.diagnostics.best_distance == 0.5
    assert result.diagnostics.retrieved_chunks == 1
    assert result.diagnostics.rejected_chunks == 2
    assert result.diagnostics.raw_distances == [0.5, 1.5, 2.0]
    assert result.diagnostics.filtered_distances == [0.5]


def test_returns_all_documents_below_threshold(fake_embedding_generator, fake_vector_store):
    """All documents below threshold should be included."""
    # Setup: all documents below threshold
    fake_vector_store.search_results = (
        ["doc1", "doc2", "doc3"],
        [0.3, 0.6, 0.9],
        [{}, {}, {}],
    )
    service = _create_service(
        fake_embedding_generator, fake_vector_store, similarity_threshold=1.0
    )

    # Execute
    result = service.retrieve_context("test query")

    # Verify: all documents returned
    assert len(result.documents) == 3
    assert result.distances == [0.3, 0.6, 0.9]
    assert result.diagnostics.best_distance == 0.3
    assert result.diagnostics.retrieved_chunks == 3
    assert result.diagnostics.rejected_chunks == 0
    assert result.diagnostics.raw_distances == [0.3, 0.6, 0.9]
    assert result.diagnostics.filtered_distances == [0.3, 0.6, 0.9]


def test_empty_results_when_all_above_threshold(fake_embedding_generator, fake_vector_store):
    """Should return empty results when all documents above threshold."""
    # Setup: all documents above threshold
    fake_vector_store.search_results = (
        ["doc1", "doc2"],
        [1.5, 2.0],
        [{}, {}],
    )
    service = _create_service(
        fake_embedding_generator, fake_vector_store, similarity_threshold=1.0
    )

    # Execute
    result = service.retrieve_context("test query")

    # Verify: empty results
    assert len(result.documents) == 0
    assert len(result.distances) == 0
    assert len(result.metadata) == 0
    assert result.diagnostics.best_distance == 1.5
    assert result.diagnostics.retrieved_chunks == 0
    assert result.diagnostics.rejected_chunks == 2
    assert result.diagnostics.raw_distances == [1.5, 2.0]
    assert result.diagnostics.filtered_distances == []


# ---------------------------------------------------------------------------
# Tests: Max Chunks Limiting
# ---------------------------------------------------------------------------


def test_respects_max_chunks_parameter(fake_embedding_generator, fake_vector_store):
    """Should limit results to max_chunks even when more are available."""
    # Setup: 5 documents all below threshold
    fake_vector_store.search_results = (
        ["doc1", "doc2", "doc3", "doc4", "doc5"],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [{}, {}, {}, {}, {}],
    )
    service = _create_service(
        fake_embedding_generator,
        fake_vector_store,
        similarity_threshold=1.0,
        max_chunks=3,  # Limit to 3
    )

    # Execute
    result = service.retrieve_context("test query")

    # Verify: only 3 returned
    assert len(result.documents) == 3
    assert result.documents == ["doc1", "doc2", "doc3"]
    assert result.diagnostics.best_distance == 0.1
    assert result.diagnostics.retrieved_chunks == 3
    assert result.diagnostics.rejected_chunks == 2
    assert result.diagnostics.raw_distances == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert result.diagnostics.filtered_distances == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


def test_deduplicates_identical_documents(fake_embedding_generator, fake_vector_store):
    """Identical documents should be deduplicated."""
    # Setup: duplicate documents
    fake_vector_store.search_results = (
        ["same doc", "same doc", "different doc"],
        [0.1, 0.2, 0.3],
        [{}, {}, {}],
    )
    service = _create_service(fake_embedding_generator, fake_vector_store)

    # Execute
    result = service.retrieve_context("test query")

    # Verify: duplicates removed
    assert len(result.documents) == 2
    assert "same doc" in result.documents
    assert "different doc" in result.documents
    assert result.diagnostics.best_distance == 0.1
    assert result.diagnostics.retrieved_chunks == 2
    assert result.diagnostics.rejected_chunks == 1
    assert result.diagnostics.raw_distances == [0.1, 0.2, 0.3]
    assert result.diagnostics.filtered_distances == [0.1, 0.3]


def test_deduplicates_by_source_and_chunk_position(
    fake_embedding_generator, fake_vector_store
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
    service = _create_service(fake_embedding_generator, fake_vector_store)

    # Execute
    result = service.retrieve_context("test query")

    # Verify: duplicate removed based on metadata
    assert len(result.documents) == 1
    assert result.diagnostics.best_distance == 0.1
    assert result.diagnostics.retrieved_chunks == 1
    assert result.diagnostics.rejected_chunks == 1
    assert result.diagnostics.raw_distances == [0.1, 0.2]
    assert result.diagnostics.filtered_distances == [0.1]


# ---------------------------------------------------------------------------
# Tests: Retrieval Result
# ---------------------------------------------------------------------------


def test_retrieve_context_returns_result_and_single_retrieval_operation(
    fake_embedding_generator, fake_embedding_model, fake_vector_store
):
    """Retrieval should embed and search once, then return a RetrievalResult."""
    fake_vector_store.search_results = (
        ["doc2", "doc1"],
        [0.8, 0.2],
        [{"source": "b"}, {"source": "a"}],
    )
    service = _create_service(
        fake_embedding_generator, fake_vector_store, similarity_threshold=1.0
    )

    result = service.retrieve_context("test query", k=2)

    assert isinstance(result, RetrievalResult)
    assert result.documents == ["doc1", "doc2"]
    assert result.distances == [0.2, 0.8]
    assert result.metadata == [{"source": "a"}, {"source": "b"}]
    assert result.diagnostics.best_distance == 0.2
    assert result.diagnostics.retrieved_chunks == 2
    assert result.diagnostics.rejected_chunks == 0
    assert result.diagnostics.raw_distances == [0.8, 0.2]
    assert result.diagnostics.filtered_distances == [0.2, 0.8]
    assert len(fake_embedding_model.calls) == 1
    assert fake_embedding_model.calls[0][0] == "test query"
    assert len(fake_vector_store.search_calls) == 1
    assert fake_vector_store.search_calls[0][1] == 5


# ---------------------------------------------------------------------------
# Tests: Prompt Generation
# ---------------------------------------------------------------------------


def test_uses_custom_prompt_formatter(fake_embedding_generator, fake_vector_store):
    """Should use injected prompt formatter when provided."""
    # Setup: custom formatter
    def custom_formatter(context, query):
        return f"CUSTOM: {query} | {context}"

    service = _create_service(
        fake_embedding_generator,
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


def test_raises_error_for_empty_query(fake_embedding_generator, fake_vector_store):
    """Should raise ValueError for empty query string."""
    service = _create_service(fake_embedding_generator, fake_vector_store)

    with pytest.raises(ValueError, match="non-empty string"):
        service.retrieve_context("")


def test_handles_empty_vector_store(fake_embedding_generator, fake_vector_store):
    """Should return empty results when vector store has no documents."""
    # Setup: empty store
    fake_vector_store.search_results = ([], [], [])
    service = _create_service(fake_embedding_generator, fake_vector_store)

    # Execute
    result = service.retrieve_context("test query")

    # Verify: empty results
    assert isinstance(result, RetrievalResult)
    assert result.documents == []
    assert result.distances == []
    assert result.metadata == []
    assert result.diagnostics.best_distance is None
    assert result.diagnostics.retrieved_chunks == 0
    assert result.diagnostics.rejected_chunks == 0
    assert result.diagnostics.raw_distances == []
    assert result.diagnostics.filtered_distances == []


# ---------------------------------------------------------------------------
# Tests: Context Formatting
# ---------------------------------------------------------------------------


def test_format_context_includes_all_metadata(fake_embedding_generator, fake_vector_store):
    """Formatted context should include source and page metadata."""
    service = _create_service(fake_embedding_generator, fake_vector_store)

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


def test_format_context_truncates_when_too_long(fake_embedding_generator, fake_vector_store):
    """Long context should be truncated to max_length."""
    service = _create_service(fake_embedding_generator, fake_vector_store)

    # Create very long document
    long_doc = "x" * 5000
    context = service.format_context([long_doc], max_length=1000)

    assert len(context) <= 1003  # 1000 + "..."
    assert context.endswith("...")
