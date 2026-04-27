"""API routes for the application."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Query the RAG system.

    Args:
        request: Query request

    Returns:
        Query response with retrieved context
    """
    # TODO: Integrate with retrieval service
    return QueryResponse(
        query=request.query, context="", retrieved_docs=[], distances=[]
    )


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy"}
