"""Unit tests for audit Pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.audit.models import (
    AuditCreate,
    AuditRecord,
    AuditRetrievalStatus,
    AuditStatus,
    AuditVerificationStatus,
)


def _valid_audit_create_data() -> dict:
    """Return valid audit data before a database ID is assigned."""
    return {
        "timestamp": datetime(2026, 7, 8, 12, 0, 0),
        "query": "What is in the knowledge base?",
        "answer": "A grounded answer.",
        "model": "gpt-3.5-turbo",
        "retrieval_status": "GOOD",
        "top_distance": 0.42,
        "retrieved_chunks": 3,
        "response_time_ms": 250,
        "verification": "PASSED",
        "status": "SUCCESS",
        "error_message": None,
    }


def test_audit_create_accepts_stable_enum_values():
    """AuditCreate should validate expected audit enum labels."""
    audit = AuditCreate(**_valid_audit_create_data())

    assert audit.retrieval_status == AuditRetrievalStatus.GOOD
    assert AuditRetrievalStatus.REJECTED.value == "REJECTED"
    assert audit.verification == AuditVerificationStatus.PASSED
    assert audit.status == AuditStatus.SUCCESS


def test_audit_record_requires_database_id():
    """AuditRecord should include the assigned database ID."""
    audit = AuditRecord(id=1, **_valid_audit_create_data())

    assert audit.id == 1
    assert audit.query == "What is in the knowledge base?"


def test_audit_model_rejects_unknown_enum_values():
    """Unexpected status labels should fail validation."""
    data = _valid_audit_create_data()
    data["status"] = "PARTIAL"

    with pytest.raises(ValidationError):
        AuditCreate(**data)


def test_audit_model_rejects_negative_metrics():
    """Metrics that should be non-negative should fail validation."""
    data = _valid_audit_create_data()
    data["response_time_ms"] = -1

    with pytest.raises(ValidationError):
        AuditCreate(**data)


def test_audit_model_serializes_enums_as_text_values():
    """JSON-mode dumps should be ready for TEXT-backed persistence."""
    audit = AuditCreate(**_valid_audit_create_data())

    dumped = audit.model_dump(mode="json")

    assert dumped["retrieval_status"] == "GOOD"
    assert dumped["verification"] == "PASSED"
    assert dumped["status"] == "SUCCESS"
