"""Audit API routes."""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from app.core.config import settings
from app.models import AuditDetailsResponse, AuditSummaryResponse
from app.services.audit_service import AuditService

router = APIRouter(prefix="/api/audit", tags=["audit"])


@lru_cache(maxsize=1)
def get_audit_service() -> AuditService:
    """Create and cache the audit service for audit API requests."""
    return AuditService(settings.AUDIT_DB_PATH)


@router.get("", response_model=list[AuditSummaryResponse])
def list_audit_summaries(
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditSummaryResponse]:
    """Return paginated audit summaries ordered by most recent first."""
    return get_audit_service().get_recent(limit=limit, offset=offset)


@router.get("/{id}", response_model=AuditDetailsResponse)
def get_audit_details(id: Annotated[int, Path(ge=1)]) -> AuditDetailsResponse:
    """Return complete details for a single audit record."""
    record = get_audit_service().get_by_id(id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found.")

    return record
