"""API routes for the application."""

import logging
from functools import lru_cache
from typing import Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService
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
    try:
        retrieval_service = get_retrieval_service()
        search_k = max(request.k, 5)
        raw_documents, raw_distances, raw_metadata = retrieval_service.vector_store.search(
            retrieval_service.embedding_service.embed_text(request.query),
            k=search_k,
        )

        documents, distances, metadata = retrieval_service.retrieve_context(
            query=request.query,
            k=request.k,
        )
        sources = extract_sources(metadata)
        retrieval_status = retrieval_service.get_retrieval_quality(
            raw_distances,
            len(documents),
        )

        if retrieval_status == "REJECTED":
            logger.info(
                "Skipping generation due to insufficient retrieval context for query: %s",
                request.query,
            )
            return QueryResponse(
                query=request.query,
                answer="I don't know based on the available documents.",
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
        answer = llm_service.generate(prompt)

        if settings.ENABLE_ANSWER_VERIFICATION:
            verification_service = get_verification_service()
            is_supported = verification_service.verify_answer(
                question=request.query,
                context=context,
                answer=answer,
            )

            if not is_supported:
                logger.warning(
                    "Answer verification failed for query: %s",
                    request.query,
                )
                answer = "I don't know based on the available documents."

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
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed") from e


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy"}
