"""Unit tests for audit API routes."""

from datetime import datetime

import pytest
from fastapi import HTTPException

from app.api import audit as audit_routes
from app.main import app
from app.models import (
    AuditDetailsResponse,
    AuditRetrievalStatus,
    AuditStatus,
    AuditSummaryResponse,
    AuditVerificationStatus,
)


class RecordingAuditService:
    """Fake audit service that records pagination requests."""

    def __init__(
        self,
        records: list[AuditSummaryResponse] | None = None,
        detail_record: AuditDetailsResponse | None = None,
    ) -> None:
        """Initialize with records returned by route-facing service methods."""
        self.records = records or []
        self.detail_record = detail_record
        self.calls: list[tuple[int, int]] = []
        self.detail_calls: list[int] = []

    def get_recent(self, limit: int, offset: int) -> list[AuditSummaryResponse]:
        """Record pagination arguments and return configured summaries."""
        self.calls.append((limit, offset))
        return self.records

    def get_by_id(self, audit_id: int) -> AuditDetailsResponse | None:
        """Record the requested ID and return configured audit details."""
        self.detail_calls.append(audit_id)
        return self.detail_record


def _summary() -> AuditSummaryResponse:
    """Return a valid audit summary response for route tests."""
    return AuditSummaryResponse(
        id=154,
        timestamp=datetime(2026, 7, 8, 14, 30, 0),
        query="Explain CNN",
        status=AuditStatus.SUCCESS,
        retrieval_status=AuditRetrievalStatus.GOOD,
        model="tinyllama",
        response_time_ms=842,
    )


def _details() -> AuditDetailsResponse:
    """Return a valid audit details response for route tests."""
    return AuditDetailsResponse(
        id=154,
        timestamp=datetime(2026, 7, 8, 14, 30, 0),
        query="Explain CNN",
        answer="A convolutional neural network answer.",
        model="tinyllama",
        retrieved_chunks=4,
        top_distance=0.63,
        response_time_ms=842,
        retrieval_status=AuditRetrievalStatus.GOOD,
        verification_status=AuditVerificationStatus.PASSED,
        status=AuditStatus.SUCCESS,
        error_message=None,
    )


def test_audit_router_is_registered_on_api_audit_path():
    """The application should expose the dedicated audit endpoint path."""
    paths = {route.path for route in app.routes}

    assert "/api/audit" in paths
    assert "/api/audit/{id}" in paths


def test_list_audit_summaries_uses_service_pagination(monkeypatch):
    """GET /api/audit should delegate pagination to AuditService."""
    service = RecordingAuditService(records=[_summary()])
    monkeypatch.setattr(audit_routes, "get_audit_service", lambda: service)

    response = audit_routes.list_audit_summaries(limit=20, offset=0)

    assert service.calls == [(20, 0)]
    assert response == [_summary()]


def test_list_audit_summaries_returns_empty_list(monkeypatch):
    """GET /api/audit should return an empty list when no audit rows exist."""
    service = RecordingAuditService()
    monkeypatch.setattr(audit_routes, "get_audit_service", lambda: service)

    response = audit_routes.list_audit_summaries(limit=20, offset=0)

    assert response == []


def test_get_audit_details_returns_matching_record(monkeypatch):
    """GET /api/audit/{id} should return complete audit details when found."""
    details = _details()
    service = RecordingAuditService(detail_record=details)
    monkeypatch.setattr(audit_routes, "get_audit_service", lambda: service)

    response = audit_routes.get_audit_details(id=154)

    assert service.detail_calls == [154]
    assert response == details


def test_get_audit_details_raises_404_when_missing(monkeypatch):
    """GET /api/audit/{id} should raise 404 when no audit row exists."""
    service = RecordingAuditService()
    monkeypatch.setattr(audit_routes, "get_audit_service", lambda: service)

    with pytest.raises(HTTPException) as exc_info:
        audit_routes.get_audit_details(id=999)

    assert service.detail_calls == [999]
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Audit record not found."
