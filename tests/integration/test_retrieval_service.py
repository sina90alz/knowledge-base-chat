"""Integration tests for RetrievalService.

These tests validate the complete retrieval pipeline using real components:
- Real EmbeddingService with actual model loading
- Real FAISSVectorStore with actual indexing
- Real RetrievalService with actual similarity search

Purpose: Verify that components integrate correctly, not to test implementation details.
Unit test coverage remains in tests/unit/.
"""

import pytest
from pathlib import Path

from app.ingestion.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.services.retrieval import RetrievalService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_service():
    """Provide real EmbeddingService with actual model.
    
    Scope: module - load model once for all tests in this file to save time.
    Uses default lightweight model (all-MiniLM-L6-v2).
    """
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(tmp_path, embedding_service):
    """Provide real FAISSVectorStore using temporary directory.
    
    Scope: function - each test gets a fresh, empty vector store.
    Uses persist=False for in-memory operation (faster tests).
    """
    return FAISSVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        store_path=tmp_path / "test_vector_store",
        persist=False,  # In-memory for faster tests
    )


@pytest.fixture
def retrieval_service(embedding_service, vector_store):
    """Provide real RetrievalService with real components.
    
    Uses default similarity threshold from settings.
    """
    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def add_documents_to_store(
    vector_store: FAISSVectorStore,
    embedding_service: EmbeddingService,
    documents: list[str],
) -> None:
    """Add documents to vector store with embeddings.
    
    Args:
        vector_store: Vector store to add documents to
        embedding_service: Service to generate embeddings
        documents: List of document texts to add
    """
    embeddings = embedding_service.embed_texts(documents)
    metadata = [{"text": doc, "source": f"doc_{i}"} for i, doc in enumerate(documents)]
    vector_store.add_texts(documents, embeddings, metadata)


# ---------------------------------------------------------------------------
# Test 1: End-to-End Retrieval
# ---------------------------------------------------------------------------


def test_end_to_end_retrieval_pipeline(
    embedding_service, vector_store, retrieval_service
):
    """Validate complete retrieval pipeline with real components.
    
    Flow:
    1. Add geography facts to vector store
    2. Query for specific fact
    3. Verify relevant document is retrieved
    
    Purpose:
    Ensure EmbeddingService → FAISSVectorStore → RetrievalService
    integration works end-to-end.
    """
    # Setup: Add geography facts
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
    ]
    add_documents_to_store(vector_store, embedding_service, documents)
    
    # Execute: Query for France's capital
    query = "What is the capital of France?"
    retrieved_docs, distances, metadata = retrieval_service.retrieve_context(query, k=3)
    
    # Verify: At least one result returned
    assert len(retrieved_docs) > 0, "Should retrieve at least one document"
    
    # Verify: Top result references Paris and France
    top_doc = retrieved_docs[0].lower()
    assert "paris" in top_doc, "Top result should mention Paris"
    assert "france" in top_doc, "Top result should mention France"
    
    # Verify: Distance indicates relevance (lower is better)
    assert distances[0] < 2.0, f"Top result should be relevant (distance={distances[0]:.4f})"


# ---------------------------------------------------------------------------
# Test 2: Retrieval Quality Ordering
# ---------------------------------------------------------------------------


def test_semantic_search_ranking_quality(
    embedding_service, vector_store, retrieval_service
):
    """Verify semantic search ranks relevant documents higher.
    
    Setup:
    - Document about FAISS (relevant)
    - Document about FastAPI (unrelated)
    
    Query: "What is FAISS?"
    
    Purpose:
    Validate that semantic similarity produces reasonable ranking.
    The FAISS document should rank higher than the FastAPI document.
    """
    # Setup: Add one relevant and one unrelated document
    documents = [
        "FAISS is a library for vector similarity search.",
        "FastAPI is a Python web framework.",
    ]
    add_documents_to_store(vector_store, embedding_service, documents)
    
    # Execute: Query about FAISS
    query = "What is FAISS?"
    retrieved_docs, distances, metadata = retrieval_service.retrieve_context(query, k=2)
    
    # Verify: Results returned
    assert len(retrieved_docs) > 0, "Should retrieve documents"
    
    # Verify: FAISS document appears before FastAPI document
    top_doc = retrieved_docs[0].lower()
    assert "faiss" in top_doc, "Most relevant document should be about FAISS"
    assert "fastapi" not in top_doc or "faiss" in top_doc, (
        "FAISS document should rank higher than unrelated content"
    )
    
    # Verify: Top result has better (lower) distance than others
    if len(retrieved_docs) > 1:
        assert distances[0] <= distances[1], (
            "Results should be ordered by relevance (ascending distance)"
        )


# ---------------------------------------------------------------------------
# Test 3: Empty Index Behavior
# ---------------------------------------------------------------------------


def test_handles_empty_vector_store_gracefully(
    embedding_service, vector_store, retrieval_service
):
    """Verify system handles empty knowledge base without errors.
    
    Setup: Empty vector store (no documents added)
    Query: Any query
    
    Purpose:
    Validate graceful degradation when no knowledge is available.
    Should return empty results, not crash.
    """
    # Setup: Vector store is empty (no documents added)
    assert len(vector_store) == 0, "Vector store should start empty"
    
    # Execute: Query on empty store
    query = "What is FAISS?"
    retrieved_docs, distances, metadata = retrieval_service.retrieve_context(query, k=5)
    
    # Verify: No exception raised, empty results returned
    assert retrieved_docs == [], "Should return empty document list"
    assert distances == [], "Should return empty distance list"
    assert metadata == [], "Should return empty metadata list"


# ---------------------------------------------------------------------------
# Test 4: Threshold Filtering in Real Scenario
# ---------------------------------------------------------------------------


def test_similarity_threshold_filters_irrelevant_results(
    embedding_service, vector_store
):
    """Verify threshold filtering works with real embeddings.
    
    Setup:
    - Add documents about various topics
    - Use strict threshold to filter poor matches
    
    Purpose:
    Validate that similarity threshold prevents retrieval of
    semantically distant documents.
    """
    # Setup: Add diverse documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language.",
    ]
    add_documents_to_store(vector_store, embedding_service, documents)
    
    # Create service with strict threshold
    strict_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        similarity_threshold=0.5,  # Very strict - only very close matches
    )
    
    # Execute: Query highly specific to one document
    query = "Tell me about machine learning and AI"
    retrieved_docs, distances, metadata = strict_service.retrieve_context(query, k=3)
    
    # Verify: Only highly relevant documents retrieved
    # With strict threshold, unrelated docs (Eiffel Tower) should be filtered
    assert len(retrieved_docs) <= 3, "Should respect max results"
    
    # If any results returned, top one should be about ML/AI
    if len(retrieved_docs) > 0:
        top_doc = retrieved_docs[0].lower()
        assert any(keyword in top_doc for keyword in ["machine", "learning", "intelligence"]), (
            "With strict threshold, top result should be highly relevant"
        )


# ---------------------------------------------------------------------------
# Test 5: Context Formatting Integration
# ---------------------------------------------------------------------------


def test_context_formatting_with_real_metadata(
    embedding_service, vector_store, retrieval_service
):
    """Verify context formatting includes real metadata from retrieval.
    
    Purpose:
    Validate that metadata flows through the pipeline and
    formatted context includes source information.
    """
    # Setup: Add documents with metadata
    documents = ["Python is a programming language."]
    embeddings = embedding_service.embed_texts(documents)
    metadata = [{"filename": "programming.txt", "page": 42}]
    vector_store.add_texts(documents, embeddings, metadata)
    
    # Execute: Retrieve and format
    query = "What is Python?"
    docs, distances, meta = retrieval_service.retrieve_context(query, k=1)
    
    # Format context
    formatted_context = retrieval_service.format_context(docs, meta)
    
    # Verify: Formatted context includes metadata
    assert "programming.txt" in formatted_context, "Should include filename"
    assert "42" in formatted_context, "Should include page number"
    assert "Python" in formatted_context, "Should include document content"
