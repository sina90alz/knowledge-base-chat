"""Audit API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.params import Depends as DependsMarker

from app.api.dependencies import get_audit_service
from app.models import (
    AuditDetailsResponse,
    AuditRetrievalStatus,
    AuditStatus,
    AuditSummaryResponse,
)
from app.services.audit_service import AuditService

router = APIRouter(prefix="/api/audit", tags=["audit"])


def _resolve_direct_call_dependency(
    audit_service: AuditService | DependsMarker,
) -> AuditService:
    """Resolve dependency defaults when route functions are called directly."""
    if isinstance(audit_service, DependsMarker):
        return get_audit_service()

    return audit_service


@router.get("", response_model=list[AuditSummaryResponse])
def list_audit_summaries(
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    audit_service: AuditService = Depends(get_audit_service),
) -> list[AuditSummaryResponse]:
    """Return paginated audit summaries ordered by most recent first."""
    audit_service = _resolve_direct_call_dependency(audit_service)
    return audit_service.get_recent(limit=limit, offset=offset)


@router.get("/search", response_model=list[AuditSummaryResponse])
def search_audit_summaries(
    status: Annotated[AuditStatus | None, Query()] = None,
    retrieval_status: Annotated[AuditRetrievalStatus | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
    min_response_time_ms: Annotated[int | None, Query(ge=0)] = None,
    max_response_time_ms: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    audit_service: AuditService = Depends(get_audit_service),
) -> list[AuditSummaryResponse]:
    """Return audit summaries matching optional search filters."""
    audit_service = _resolve_direct_call_dependency(audit_service)
    return audit_service.search(
        status=status,
        retrieval_status=retrieval_status,
        model=model,
        min_response_time_ms=min_response_time_ms,
        max_response_time_ms=max_response_time_ms,
        limit=limit,
        offset=offset,
    )


@router.get("/{id}", response_model=AuditDetailsResponse)
def get_audit_details(
    id: Annotated[int, Path(ge=1)],
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditDetailsResponse:
    """Return complete details for a single audit record."""
    audit_service = _resolve_direct_call_dependency(audit_service)
    record = audit_service.get_by_id(id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found.")

    return record
