"""Unit tests for audit API routes."""

from datetime import datetime

from app.api import audit as audit_routes
from app.main import app
from app.models import (
    AuditRetrievalStatus,
    AuditStatus,
    AuditSummaryResponse,
)


class RecordingAuditService:
    """Fake audit service that records pagination requests."""

    def __init__(self, records: list[AuditSummaryResponse] | None = None) -> None:
        """Initialize with records returned by get_recent."""
        self.records = records or []
        self.calls: list[tuple[int, int]] = []

    def get_recent(self, limit: int, offset: int) -> list[AuditSummaryResponse]:
        """Record pagination arguments and return configured summaries."""
        self.calls.append((limit, offset))
        return self.records


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


def test_audit_router_is_registered_on_api_audit_path():
    """The application should expose the dedicated audit endpoint path."""
    paths = {route.path for route in app.routes}

    assert "/api/audit" in paths


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
