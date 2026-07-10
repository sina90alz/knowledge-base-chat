"""Unit tests for audit writes from the query route."""

import asyncio
from typing import Any

from app.api import routes
from app.audit.models import AuditStatus, AuditVerificationStatus


class FakeEmbeddingService:
    """Fake embedding service that records embedding calls."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_text(self, query: str) -> list[float]:
        self.embed_calls += 1
        return [1.0, 2.0, 3.0]


class FakeVectorStore:
    """Fake vector store that returns predefined raw search results."""

    def __init__(
        self,
        documents: list[str],
        distances: list[float],
        metadata: list[dict[str, Any]],
    ) -> None:
        self.documents = documents
        self.distances = distances
        self.metadata = metadata
        self.search_calls = 0

    def search(
        self,
        query_embedding: list[float],
        k: int,
    ) -> tuple[list[str], list[float], list[dict[str, Any]]]:
        self.search_calls += 1
        return self.documents, self.distances, self.metadata


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
    ) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.vector_store = FakeVectorStore(
            documents=["raw document"],
            distances=raw_distances,
            metadata=[{"filename": "raw.pdf"}],
        )
        self.retrieval_status = retrieval_status
        self.documents = documents
        self.distances = distances
        self.metadata = metadata
        self.retrieve_calls = 0

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
    ) -> tuple[list[str], list[float], list[dict[str, Any]]]:
        self.retrieve_calls += 1
        return self.documents, self.distances, self.metadata

    def get_retrieval_quality(
        self,
        raw_distances: list[float],
        filtered_count: int,
    ) -> str:
        return self.retrieval_status

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

    def __init__(self, answer: str = "final answer") -> None:
        self.model_name = "test-model"
        self.answer = answer
        self.generate_calls = 0

    def generate(self, prompt: str) -> str:
        self.generate_calls += 1
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
    assert retrieval_service.vector_store.search_calls == 1
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
    assert retrieval_service.vector_store.search_calls == 1
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
