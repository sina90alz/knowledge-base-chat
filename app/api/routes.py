"""API routes for the application."""

import logging
from functools import lru_cache
from typing import Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    k: int = 5


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    answer: str
    context: str
    retrieved_docs: List[str]
    distances: List[float]
    metadata: List[dict[str, Any]]
    sources: List[str]


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
def get_llm_service() -> LLMService:
    """Create and cache the LLM service for API requests."""
    return LLMService()


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
        documents, distances, metadata = retrieval_service.retrieve_context(
            query=request.query,
            k=request.k,
        )
        sources = extract_sources(metadata)

        if not documents:
            logger.info("No documents found for query: %s", request.query)
            return QueryResponse(
                query=request.query,
                answer="I don't know based on the available documents.",
                context="",
                retrieved_docs=[],
                distances=[],
                metadata=[],
                sources=[],
            )

        context = retrieval_service.format_context(documents, metadata)
        prompt = retrieval_service.generate_prompt(request.query, context)
        answer = get_llm_service().generate(prompt)

        return QueryResponse(
            query=request.query,
            answer=answer,
            context=context,
            retrieved_docs=documents,
            distances=distances,
            metadata=metadata,
            sources=sources,
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
