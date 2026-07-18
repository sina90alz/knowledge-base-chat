"""Pydantic models for audit records."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AuditStatus(str, Enum):
    """Persistence outcome for an audited query."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AuditRetrievalStatus(str, Enum):
    """Stable retrieval quality labels stored in audit records."""

    GOOD = "GOOD"
    WEAK = "WEAK"
    REJECTED = "REJECTED"


class AuditVerificationStatus(str, Enum):
    """Stable answer verification labels stored in audit records."""

    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    PASSED = "PASSED"
    FAILED = "FAILED"


class AuditCreate(BaseModel):
    """Audit record data before a database ID is assigned."""

    timestamp: datetime
    query: str = Field(min_length=1)
    answer: str | None = None
    model: str | None = None
    retrieval_status: AuditRetrievalStatus | None = None
    top_distance: float | None = Field(default=None, ge=0)
    retrieved_chunks: int | None = Field(default=None, ge=0)
    response_time_ms: int = Field(ge=0)
    verification: AuditVerificationStatus
    status: AuditStatus
    error_message: str | None = None


class AuditRecord(AuditCreate):
    """Complete stored audit record."""

    id: int = Field(ge=1)


class AuditSummaryResponse(BaseModel):
    """Public audit overview data returned by the audit API."""

    id: int = Field(ge=1)
    timestamp: datetime
    query: str
    status: AuditStatus
    retrieval_status: AuditRetrievalStatus | None = None
    model: str | None = None
    response_time_ms: int = Field(ge=0)
