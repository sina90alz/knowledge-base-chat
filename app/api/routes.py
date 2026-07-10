"""API routes for the application."""

import logging
import time
from functools import lru_cache
from typing import Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.audit.models import AuditVerificationStatus
from app.core.config import settings
from app.ingestion.embedder import EmbeddingService
from app.services.audit_service import AuditService
from app.services.llm import get_llm_service
from app.services.retrieval import RetrievalService
from app.services.verification import AnswerVerificationService
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


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


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Create and cache the retrieval service for API requests."""
    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    vector_store = FAISSVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        store_path=settings.VECTOR_STORE_PATH,
    )
    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


@lru_cache(maxsize=1)
def get_verification_service() -> AnswerVerificationService:
    """Create and cache the answer verification service."""
    return AnswerVerificationService(llm_service=get_llm_service())


@lru_cache(maxsize=1)
def get_audit_service() -> AuditService:
    """Create and cache the audit service for API requests."""
    return AuditService(settings.AUDIT_DB_PATH)


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


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
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
        retrieval_service = get_retrieval_service()
        search_k = max(request.k, 5)
        raw_documents, raw_distances, raw_metadata = retrieval_service.vector_store.search(
            retrieval_service.embedding_service.embed_text(request.query),
            k=search_k,
        )
        audit_top_distance = min(raw_distances) if raw_distances else None

        documents, distances, metadata = retrieval_service.retrieve_context(
            query=request.query,
            k=request.k,
        )
        audit_retrieved_chunks = len(documents)
        sources = extract_sources(metadata)
        retrieval_status = retrieval_service.get_retrieval_quality(
            raw_distances,
            len(documents),
        )
        audit_retrieval_status = retrieval_status

        if retrieval_status == "REJECTED":
            logger.info(
                "Skipping generation due to insufficient retrieval context for query: %s",
                request.query,
            )
            audit_answer = "I don't know based on the available documents."
            audit_response_time_ms = int(
                (time.perf_counter() - request_started_at) * 1000
            )
            return QueryResponse(
                query=request.query,
                answer=audit_answer,
                context="",
                retrieved_docs=[],
                distances=[],
                metadata=[],
                sources=[],
                retrieval_status=retrieval_status,
            )

        if retrieval_status == "WEAK":
            logger.warning(
                "Weak retrieval quality for query '%s'; proceeding with caution.",
                request.query,
            )

        context = retrieval_service.format_context(documents, metadata)
        prompt = retrieval_service.generate_prompt(request.query, context)
        llm_service = get_llm_service()
        audit_model = getattr(llm_service, "model_name", None)
        answer = llm_service.generate(prompt)
        audit_answer = answer

        if settings.ENABLE_ANSWER_VERIFICATION:
            verification_service = get_verification_service()
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
        return QueryResponse(
            query=request.query,
            answer=answer,
            context=context,
            retrieved_docs=documents,
            distances=distances,
            metadata=metadata,
            sources=sources,
            retrieval_status=retrieval_status,
        )
    except ValueError as e:
        audit_response_time_ms = int((time.perf_counter() - request_started_at) * 1000)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        audit_response_time_ms = int((time.perf_counter() - request_started_at) * 1000)
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed") from e


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy"}
