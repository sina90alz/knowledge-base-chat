"""Unit tests for the SQLite-backed audit service."""

import sqlite3
from datetime import datetime
from pathlib import Path

from app.models import (
    AuditCreate,
    AuditDetailsResponse,
    AuditRetrievalStatus,
    AuditStatus,
    AuditSummaryResponse,
    AuditVerificationStatus,
)
from app.services.audit_service import AuditService


def _audit_create(
    query: str = "What is in the knowledge base?",
    *,
    timestamp: datetime | None = None,
    answer: str | None = "A grounded answer.",
    model: str | None = "gpt-3.5-turbo",
    retrieval_status: AuditRetrievalStatus | None = AuditRetrievalStatus.GOOD,
    top_distance: float | None = 0.42,
    retrieved_chunks: int | None = 3,
    response_time_ms: int = 250,
    verification: AuditVerificationStatus = AuditVerificationStatus.PASSED,
    status: AuditStatus = AuditStatus.SUCCESS,
    error_message: str | None = None,
) -> AuditCreate:
    """Create valid audit data for service tests."""
    return AuditCreate(
        timestamp=timestamp or datetime(2026, 7, 8, 12, 0, 0),
        query=query,
        answer=answer,
        model=model,
        retrieval_status=retrieval_status,
        top_distance=top_distance,
        retrieved_chunks=retrieved_chunks,
        response_time_ms=response_time_ms,
        verification=verification,
        status=status,
        error_message=error_message,
    )


def _db_path(tmp_path: Path) -> Path:
    """Return a nested temporary database path."""
    return tmp_path / "audit" / "audit.db"


def test_audit_service_creates_parent_directory_and_table(tmp_path):
    """Constructing the service should initialize the database table."""
    db_path = _db_path(tmp_path)

    AuditService(db_path)

    assert db_path.parent.exists()
    with sqlite3.connect(db_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "audit_logs"),
        ).fetchone()

    assert table == ("audit_logs",)


def test_log_inserts_record_and_returns_generated_id(tmp_path):
    """log should persist a record and return the generated primary key."""
    service = AuditService(_db_path(tmp_path))

    audit_id = service.log(_audit_create())

    assert audit_id == 1
    stored = service.get_by_id(audit_id)
    assert stored is not None
    assert stored.id == audit_id
    assert stored.query == "What is in the knowledge base?"


def test_get_recent_returns_newest_records_first_with_limit(tmp_path):
    """get_recent should order by descending timestamp and respect the limit."""
    service = AuditService(_db_path(tmp_path))
    first_id = service.log(
        _audit_create("first query", timestamp=datetime(2026, 7, 8, 12, 0, 0))
    )
    second_id = service.log(
        _audit_create("second query", timestamp=datetime(2026, 7, 8, 13, 0, 0))
    )
    third_id = service.log(
        _audit_create("third query", timestamp=datetime(2026, 7, 8, 14, 0, 0))
    )

    records = service.get_recent(limit=2)

    assert all(isinstance(record, AuditSummaryResponse) for record in records)
    assert [record.id for record in records] == [third_id, second_id]
    assert [record.query for record in records] == ["third query", "second query"]
    assert first_id == 1


def test_get_recent_supports_offset(tmp_path):
    """get_recent should skip records according to the requested offset."""
    service = AuditService(_db_path(tmp_path))
    service.log(_audit_create("first query", timestamp=datetime(2026, 7, 8, 12, 0, 0)))
    second_id = service.log(
        _audit_create("second query", timestamp=datetime(2026, 7, 8, 13, 0, 0))
    )
    service.log(_audit_create("third query", timestamp=datetime(2026, 7, 8, 14, 0, 0)))

    records = service.get_recent(limit=1, offset=1)

    assert [record.id for record in records] == [second_id]
    assert [record.query for record in records] == ["second query"]


def test_get_recent_returns_empty_list_when_no_records_exist(tmp_path):
    """get_recent should return an empty list for an empty audit table."""
    service = AuditService(_db_path(tmp_path))

    assert service.get_recent(limit=20, offset=0) == []


def test_get_recent_returns_only_summary_fields(tmp_path):
    """get_recent should not expose complete audit record columns."""
    service = AuditService(_db_path(tmp_path))
    service.log(_audit_create())

    record = service.get_recent(limit=1)[0]

    assert set(record.model_dump()) == {
        "id",
        "timestamp",
        "query",
        "status",
        "retrieval_status",
        "model",
        "response_time_ms",
    }


def test_get_by_id_returns_matching_record(tmp_path):
    """get_by_id should return the requested audit details."""
    service = AuditService(_db_path(tmp_path))
    audit_id = service.log(
        _audit_create(
            "failed query",
            answer=None,
            retrieval_status=AuditRetrievalStatus.REJECTED,
            top_distance=None,
            retrieved_chunks=0,
            verification=AuditVerificationStatus.FAILED,
            status=AuditStatus.FAILED,
            error_message="No relevant documents found.",
        )
    )

    record = service.get_by_id(audit_id)

    assert isinstance(record, AuditDetailsResponse)
    assert record.id == audit_id
    assert record.query == "failed query"
    assert record.retrieval_status == AuditRetrievalStatus.REJECTED
    assert record.verification_status == AuditVerificationStatus.FAILED
    assert record.status == AuditStatus.FAILED
    assert record.error_message == "No relevant documents found."


def test_get_by_id_returns_all_detail_fields(tmp_path):
    """get_by_id should expose the complete audit detail response fields."""
    service = AuditService(_db_path(tmp_path))
    audit_id = service.log(_audit_create())

    record = service.get_by_id(audit_id)

    assert record is not None
    assert set(record.model_dump()) == {
        "id",
        "timestamp",
        "query",
        "answer",
        "model",
        "retrieved_chunks",
        "top_distance",
        "response_time_ms",
        "retrieval_status",
        "verification_status",
        "status",
        "error_message",
    }


def test_get_by_id_returns_none_for_missing_id(tmp_path):
    """Missing audit IDs should return None."""
    service = AuditService(_db_path(tmp_path))

    assert service.get_by_id(999) is None


def test_nullable_fields_round_trip(tmp_path):
    """Nullable audit fields should persist and load as None."""
    service = AuditService(_db_path(tmp_path))
    audit_id = service.log(
        _audit_create(
            answer=None,
            model=None,
            retrieval_status=None,
            top_distance=None,
            retrieved_chunks=None,
            error_message=None,
        )
    )

    record = service.get_by_id(audit_id)

    assert record is not None
    assert record.answer is None
    assert record.model is None
    assert record.retrieval_status is None
    assert record.top_distance is None
    assert record.retrieved_chunks is None
    assert record.error_message is None
