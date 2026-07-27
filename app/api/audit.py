"""Audit API routes."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from app.infrastructure.bootstrap import get_application_container
from app.models import (
    AuditDetailsResponse,
    AuditRetrievalStatus,
    AuditStatus,
    AuditSummaryResponse,
)
from app.services.audit_service import AuditService

router = APIRouter(prefix="/api/audit", tags=["audit"])


def get_audit_service() -> AuditService:
    """Return the startup-wired audit service."""
    return get_application_container().audit_service


@router.get("", response_model=list[AuditSummaryResponse])
def list_audit_summaries(
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditSummaryResponse]:
    """Return paginated audit summaries ordered by most recent first."""
    return get_audit_service().get_recent(limit=limit, offset=offset)


@router.get("/search", response_model=list[AuditSummaryResponse])
def search_audit_summaries(
    status: Annotated[AuditStatus | None, Query()] = None,
    retrieval_status: Annotated[AuditRetrievalStatus | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
    min_response_time_ms: Annotated[int | None, Query(ge=0)] = None,
    max_response_time_ms: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditSummaryResponse]:
    """Return audit summaries matching optional search filters."""
    return get_audit_service().search(
        status=status,
        retrieval_status=retrieval_status,
        model=model,
        min_response_time_ms=min_response_time_ms,
        max_response_time_ms=max_response_time_ms,
        limit=limit,
        offset=offset,
    )


@router.get("/{id}", response_model=AuditDetailsResponse)
def get_audit_details(id: Annotated[int, Path(ge=1)]) -> AuditDetailsResponse:
    """Return complete details for a single audit record."""
    record = get_audit_service().get_by_id(id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found.")

    return record
