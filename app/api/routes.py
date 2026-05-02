"""API routes for the application."""

import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService
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
    context: str
    retrieved_docs: list
    distances: list
    metadata: list


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
        context = retrieval_service.format_context(documents)

        return QueryResponse(
            query=request.query,
            context=context,
            retrieved_docs=documents,
            distances=distances,
            metadata=metadata,
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
