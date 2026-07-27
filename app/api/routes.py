"""API routes for the application."""

from datetime import datetime, timezone
import logging
import time
from collections.abc import Callable
from typing import Any, List, TypeVar

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Depends as DependsMarker
from pydantic import BaseModel

from app.api.dependencies import (
    get_audit_service,
    get_llm_service,
    get_retrieval_service,
    get_verification_service,
)
from app.models import (
    AuditCreate,
    AuditRetrievalStatus,
    AuditStatus,
    AuditVerificationStatus,
)
from app.core.config import settings
from app.services.audit_service import AuditService
from app.services.retrieval import RetrievalService
from app.services.verification import AnswerVerificationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])
T = TypeVar("T")


class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    k: int = 5

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What information is available in the knowledge base?",
                "k": 5,
            }
        }
    }


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    answer: str
    context: str
    retrieved_docs: List[str]
    distances: List[float]
    metadata: List[dict[str, Any]]
    sources: List[str]
    retrieval_status: str


def _resolve_direct_call_dependency(
    value: T | DependsMarker,
    provider: Callable[[], T],
) -> T:
    """Resolve dependency defaults when route functions are called directly."""
    if isinstance(value, DependsMarker):
        return provider()

    return value


def extract_sources(metadata: List[dict[str, Any]]) -> List[str]:
    """Extract unique source filenames from retrieval metadata."""
    sources: List[str] = []
    seen_sources: set[str] = set()

    for item in metadata:
        source = item.get("filename") or item.get("source")
        if source and source not in seen_sources:
            sources.append(str(source))
            seen_sources.add(str(source))

    return sources


def _log_audit(
    *,
    audit_service: AuditService,
    query: str,
    answer: str | None,
    model: str | None,
    retrieval_status: str | None,
    top_distance: float | None,
    retrieved_chunks: int | None,
    response_time_ms: int,
    verification: AuditVerificationStatus,
    status: AuditStatus,
    error_message: str | None,
) -> None:
    """Best-effort audit write for query attempts."""
    try:
        audit_record = AuditCreate(
            timestamp=datetime.now(timezone.utc),
            query=query,
            answer=answer,
            model=model,
            retrieval_status=(
                AuditRetrievalStatus(retrieval_status)
                if retrieval_status is not None
                else None
            ),
            top_distance=top_distance,
            retrieved_chunks=retrieved_chunks,
            response_time_ms=response_time_ms,
            verification=verification,
            status=status,
            error_message=error_message,
        )
        audit_service.log(audit_record)
    except Exception:
        logger.exception("Failed to persist query audit record")


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_service: Any = Depends(get_llm_service),
    audit_service: AuditService = Depends(get_audit_service),
    verification_service: AnswerVerificationService = Depends(get_verification_service),
) -> QueryResponse:
    """Query the RAG system.

    Args:
        request: Query request

    Returns:
        Query response with retrieved context
    """
    request_started_at = time.perf_counter()
    audit_answer: str | None = None
    audit_model: str | None = None
    audit_retrieval_status: str | None = None
    audit_top_distance: float | None = None
    audit_retrieved_chunks: int | None = None
    audit_response_time_ms = 0
    audit_verification = (
        AuditVerificationStatus.ENABLED
        if settings.ENABLE_ANSWER_VERIFICATION
        else AuditVerificationStatus.DISABLED
    )

    try:
        retrieval_service = _resolve_direct_call_dependency(
            retrieval_service,
            get_retrieval_service,
        )
        audit_service = _resolve_direct_call_dependency(
            audit_service,
            get_audit_service,
        )

        result = retrieval_service.retrieve_context(
            query=request.query,
            k=request.k,
        )
        documents = result.documents
        distances = result.distances
        metadata = result.metadata
        diagnostics = result.diagnostics
        audit_top_distance = diagnostics.best_distance
        audit_retrieved_chunks = diagnostics.retrieved_chunks
        sources = extract_sources(metadata)

        retrieval_status = (
            "REJECTED"
            if diagnostics.retrieved_chunks == 0
            else (
                "WEAK"
                if (
                    diagnostics.best_distance is not None
                    and diagnostics.best_distance > diagnostics.threshold
                )
                else "GOOD"
            )
        )
        audit_retrieval_status = retrieval_status

        if diagnostics.retrieved_chunks == 0:
            logger.info(
                "Skipping generation due to insufficient retrieval context for query: %s",
                request.query,
            )
            audit_answer = "I don't know based on the available documents."
            audit_response_time_ms = int(
                (time.perf_counter() - request_started_at) * 1000
            )
            response = QueryResponse(
                query=request.query,
                answer=audit_answer,
                context="",
                retrieved_docs=[],
                distances=[],
                metadata=[],
                sources=[],
                retrieval_status=retrieval_status,
            )
            _log_audit(
                audit_service=audit_service,
                query=request.query,
                answer=audit_answer,
                model=audit_model,
                retrieval_status=audit_retrieval_status,
                top_distance=audit_top_distance,
                retrieved_chunks=audit_retrieved_chunks,
                response_time_ms=audit_response_time_ms,
                verification=audit_verification,
                status=AuditStatus.SUCCESS,
                error_message=None,
            )
            return response

        if retrieval_status == "WEAK":
            logger.warning(
                "Weak retrieval quality for query '%s'; proceeding with caution.",
                request.query,
            )

        context = retrieval_service.format_context(documents, metadata)
        prompt = retrieval_service.generate_prompt(request.query, context)
        llm_service = _resolve_direct_call_dependency(llm_service, get_llm_service)
        audit_model = getattr(llm_service, "model_name", None)
        answer = llm_service.generate(prompt)
        audit_answer = answer

        if settings.ENABLE_ANSWER_VERIFICATION:
            verification_service = _resolve_direct_call_dependency(
                verification_service,
                get_verification_service,
            )
            is_supported = verification_service.verify_answer(
                question=request.query,
                context=context,
                answer=answer,
            )
            audit_verification = (
                AuditVerificationStatus.PASSED
                if is_supported
                else AuditVerificationStatus.FAILED
            )

            if not is_supported:
                logger.warning(
                    "Answer verification failed for query: %s",
                    request.query,
                )
                answer = "I don't know based on the available documents."
                audit_answer = answer

        audit_response_time_ms = int((time.perf_counter() - request_started_at) * 1000)
        response = QueryResponse(
            query=request.query,
            answer=answer,
            context=context,
            retrieved_docs=documents,
            distances=distances,
            metadata=metadata,
            sources=sources,
            retrieval_status=retrieval_status,
        )
        _log_audit(
            audit_service=audit_service,
            query=request.query,
            answer=audit_answer,
            model=audit_model,
            retrieval_status=audit_retrieval_status,
            top_distance=audit_top_distance,
            retrieved_chunks=audit_retrieved_chunks,
            response_time_ms=audit_response_time_ms,
            verification=audit_verification,
            status=AuditStatus.SUCCESS,
            error_message=None,
        )
        return response
    except ValueError as e:
        audit_response_time_ms = int((time.perf_counter() - request_started_at) * 1000)
        audit_service = _resolve_direct_call_dependency(
            audit_service,
            get_audit_service,
        )
        _log_audit(
            audit_service=audit_service,
            query=request.query,
            answer=audit_answer,
            model=audit_model,
            retrieval_status=audit_retrieval_status,
            top_distance=audit_top_distance,
            retrieved_chunks=audit_retrieved_chunks,
            response_time_ms=audit_response_time_ms,
            verification=audit_verification,
            status=AuditStatus.FAILED,
            error_message=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        audit_response_time_ms = int((time.perf_counter() - request_started_at) * 1000)
        audit_service = _resolve_direct_call_dependency(
            audit_service,
            get_audit_service,
        )
        _log_audit(
            audit_service=audit_service,
            query=request.query,
            answer=audit_answer,
            model=audit_model,
            retrieval_status=audit_retrieval_status,
            top_distance=audit_top_distance,
            retrieved_chunks=audit_retrieved_chunks,
            response_time_ms=audit_response_time_ms,
            verification=audit_verification,
            status=AuditStatus.FAILED,
            error_message=str(e),
        )
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed") from e


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy"}
