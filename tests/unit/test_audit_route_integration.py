"""Unit tests for audit writes from the query route."""

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from app.api import routes
from app.audit.models import AuditStatus, AuditVerificationStatus
from app.retrieval.models import RetrievalDiagnostics, RetrievalResult


class FakeEmbeddingService:
    """Fake embedding service that must not be called directly by the route."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_text(self, query: str) -> list[float]:
        self.embed_calls += 1
        raise AssertionError("query route must not generate retrieval embeddings")


class FakeVectorStore:
    """Fake vector store that must not be called directly by the route."""

    def __init__(self) -> None:
        self.search_calls = 0

    def search(
        self,
        query_embedding: list[float],
        k: int,
    ) -> tuple[list[str], list[float], list[dict[str, Any]]]:
        self.search_calls += 1
        raise AssertionError("query route must not call vector search directly")


class FakeRetrievalService:
    """Fake retrieval service for exercising query_rag without FAISS."""

    def __init__(
        self,
        *,
        retrieval_status: str,
        raw_distances: list[float],
        documents: list[str],
        distances: list[float],
        metadata: list[dict[str, Any]],
        retrieve_exception: Exception | None = None,
    ) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.vector_store = FakeVectorStore()
        self.retrieval_status = retrieval_status
        self.raw_distances = raw_distances
        self.documents = documents
        self.distances = distances
        self.metadata = metadata
        self.retrieve_exception = retrieve_exception
        self.retrieve_calls = 0

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
    ) -> RetrievalResult:
        self.retrieve_calls += 1
        if self.retrieve_exception is not None:
            raise self.retrieve_exception

        threshold = 0.8 if self.retrieval_status == "WEAK" else 1.0
        return RetrievalResult(
            documents=self.documents,
            distances=self.distances,
            metadata=self.metadata,
            diagnostics=RetrievalDiagnostics(
                best_distance=min(self.raw_distances) if self.raw_distances else None,
                threshold=threshold,
                raw_distances=self.raw_distances,
                filtered_distances=self.distances,
                retrieved_chunks=len(self.documents),
                rejected_chunks=len(self.raw_distances) - len(self.documents),
            ),
        )

    def get_retrieval_quality(
        self,
        raw_distances: list[float],
        filtered_count: int,
    ) -> str:
        raise AssertionError("query route must not ask service for retrieval quality")

    def format_context(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> str:
        return "\n".join(documents)

    def generate_prompt(self, query: str, context: str) -> str:
        return f"{query}\n{context}"


class FakeLLMService:
    """Fake LLM service with a model name."""

    def __init__(
        self,
        answer: str = "final answer",
        exception: Exception | None = None,
    ) -> None:
        self.model_name = "test-model"
        self.answer = answer
        self.exception = exception
        self.generate_calls = 0

    def generate(self, prompt: str) -> str:
        self.generate_calls += 1
        if self.exception is not None:
            raise self.exception
        return self.answer


class RecordingAuditService:
    """Fake audit service that records audit writes."""

    def __init__(self) -> None:
        self.records = []

    def log(self, record):
        self.records.append(record)
        return len(self.records)


class FailingAuditService:
    """Fake audit service that fails audit writes."""

    def log(self, record):
        raise RuntimeError("audit database unavailable")


def _run_query(query: str = "test query") -> routes.QueryResponse:
    """Run the async query route from sync tests."""
    return asyncio.run(routes.query_rag(routes.QueryRequest(query=query, k=5)))


def test_successful_generated_request_creates_one_success_audit_record(monkeypatch):
    """Normal successful generation should write exactly one audit record."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="GOOD",
        raw_distances=[0.2, 0.7],
        documents=["retrieved document"],
        distances=[0.2],
        metadata=[{"filename": "doc.pdf"}],
    )
    llm_service = FakeLLMService(answer="generated answer")
    audit_service = RecordingAuditService()

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: audit_service)

    response = _run_query()

    assert response.answer == "generated answer"
    assert len(audit_service.records) == 1
    record = audit_service.records[0]
    assert record.status == AuditStatus.SUCCESS
    assert record.query == "test query"
    assert record.answer == "generated answer"
    assert record.model == "test-model"
    assert record.retrieval_status.value == "GOOD"
    assert record.top_distance == 0.2
    assert record.retrieved_chunks == 1
    assert record.response_time_ms >= 0
    assert record.verification == AuditVerificationStatus.DISABLED
    assert record.error_message is None
    assert retrieval_service.embedding_service.embed_calls == 0
    assert retrieval_service.vector_store.search_calls == 0
    assert retrieval_service.retrieve_calls == 1
    assert llm_service.generate_calls == 1


def test_rejected_retrieval_creates_one_success_audit_record(monkeypatch):
    """REJECTED retrieval should write one successful audit row without LLM use."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="REJECTED",
        raw_distances=[1.8],
        documents=[],
        distances=[],
        metadata=[],
    )
    llm_service = FakeLLMService()
    audit_service = RecordingAuditService()

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: audit_service)

    response = _run_query()

    assert response.answer == "I don't know based on the available documents."
    assert response.retrieval_status == "REJECTED"
    assert len(audit_service.records) == 1
    record = audit_service.records[0]
    assert record.status == AuditStatus.SUCCESS
    assert record.retrieval_status.value == "REJECTED"
    assert record.answer == "I don't know based on the available documents."
    assert record.model is None
    assert record.top_distance == 1.8
    assert record.retrieved_chunks == 0
    assert record.response_time_ms >= 0
    assert record.verification == AuditVerificationStatus.DISABLED
    assert record.error_message is None
    assert retrieval_service.embedding_service.embed_calls == 0
    assert retrieval_service.vector_store.search_calls == 0
    assert retrieval_service.retrieve_calls == 1
    assert llm_service.generate_calls == 0


def test_audit_persistence_failure_preserves_successful_response(monkeypatch):
    """Audit write failures should not convert successful queries into API errors."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="WEAK",
        raw_distances=[0.9],
        documents=["retrieved document"],
        distances=[0.9],
        metadata=[{"filename": "doc.pdf"}],
    )
    llm_service = FakeLLMService(answer="weak answer")

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: FailingAuditService())

    response = _run_query()

    assert response.answer == "weak answer"
    assert response.retrieval_status == "WEAK"
    assert response.retrieved_docs == ["retrieved document"]
    assert llm_service.generate_calls == 1


def test_weak_retrieval_success_creates_one_success_audit_record(monkeypatch):
    """WEAK retrieval followed by generation should write exactly one audit row."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="WEAK",
        raw_distances=[0.9],
        documents=["retrieved document"],
        distances=[0.9],
        metadata=[{"filename": "doc.pdf"}],
    )
    llm_service = FakeLLMService(answer="weak answer")
    audit_service = RecordingAuditService()

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: audit_service)

    response = _run_query()

    assert response.model_dump() == {
        "query": "test query",
        "answer": "weak answer",
        "context": "retrieved document",
        "retrieved_docs": ["retrieved document"],
        "distances": [0.9],
        "metadata": [{"filename": "doc.pdf"}],
        "sources": ["doc.pdf"],
        "retrieval_status": "WEAK",
    }
    assert len(audit_service.records) == 1
    record = audit_service.records[0]
    assert record.status == AuditStatus.SUCCESS
    assert record.retrieval_status.value == "WEAK"
    assert record.answer == "weak answer"


def test_value_error_path_creates_one_failed_audit_record(monkeypatch):
    """ValueError failures should write one FAILED audit record."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="GOOD",
        raw_distances=[0.2, 0.5],
        documents=["retrieved document"],
        distances=[0.2],
        metadata=[{"filename": "doc.pdf"}],
        retrieve_exception=ValueError("invalid retrieval result"),
    )
    llm_service = FakeLLMService()
    audit_service = RecordingAuditService()

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: audit_service)

    with pytest.raises(HTTPException) as exc_info:
        _run_query()

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid retrieval result"
    assert len(audit_service.records) == 1
    record = audit_service.records[0]
    assert record.status == AuditStatus.FAILED
    assert record.error_message == "invalid retrieval result"
    assert record.query == "test query"
    assert record.answer is None
    assert record.model is None
    assert record.retrieval_status is None
    assert record.top_distance is None
    assert record.retrieved_chunks is None
    assert record.response_time_ms >= 0
    assert record.verification == AuditVerificationStatus.DISABLED


def test_unexpected_exception_creates_one_failed_audit_record(monkeypatch):
    """Unexpected failures should write one FAILED audit record."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="GOOD",
        raw_distances=[0.2],
        documents=["retrieved document"],
        distances=[0.2],
        metadata=[{"filename": "doc.pdf"}],
    )
    llm_service = FakeLLMService(exception=RuntimeError("llm exploded"))
    audit_service = RecordingAuditService()

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: audit_service)

    with pytest.raises(HTTPException) as exc_info:
        _run_query()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Query failed"
    assert len(audit_service.records) == 1
    record = audit_service.records[0]
    assert record.status == AuditStatus.FAILED
    assert record.error_message == "llm exploded"
    assert record.answer is None
    assert record.model == "test-model"
    assert record.retrieval_status.value == "GOOD"
    assert record.top_distance == 0.2
    assert record.retrieved_chunks == 1
    assert record.response_time_ms >= 0


def test_audit_persistence_failure_preserves_http_400(monkeypatch):
    """Audit failures during ValueError handling should preserve HTTP 400."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="GOOD",
        raw_distances=[0.2],
        documents=["retrieved document"],
        distances=[0.2],
        metadata=[{"filename": "doc.pdf"}],
        retrieve_exception=ValueError("invalid retrieval result"),
    )

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: FailingAuditService())

    with pytest.raises(HTTPException) as exc_info:
        _run_query()

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid retrieval result"


def test_audit_persistence_failure_preserves_http_500(monkeypatch):
    """Audit failures during unexpected error handling should preserve HTTP 500."""
    retrieval_service = FakeRetrievalService(
        retrieval_status="GOOD",
        raw_distances=[0.2],
        documents=["retrieved document"],
        distances=[0.2],
        metadata=[{"filename": "doc.pdf"}],
    )
    llm_service = FakeLLMService(exception=RuntimeError("llm exploded"))

    monkeypatch.setattr(routes.settings, "ENABLE_ANSWER_VERIFICATION", False)
    monkeypatch.setattr(routes, "get_retrieval_service", lambda: retrieval_service)
    monkeypatch.setattr(routes, "get_llm_service", lambda: llm_service)
    monkeypatch.setattr(routes, "get_audit_service", lambda: FailingAuditService())

    with pytest.raises(HTTPException) as exc_info:
        _run_query()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Query failed"
