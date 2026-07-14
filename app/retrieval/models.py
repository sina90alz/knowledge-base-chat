"""Domain models for retrieval results."""

from typing import Any

from pydantic import BaseModel


class RetrievalDiagnostics(BaseModel):
    """Diagnostic details describing a retrieval operation."""

    best_distance: float | None
    threshold: float
    raw_distances: list[float]
    filtered_distances: list[float]
    retrieved_chunks: int
    rejected_chunks: int


class RetrievalResult(BaseModel):
    """Documents and diagnostics produced by a retrieval operation."""

    documents: list[str]
    distances: list[float]
    metadata: list[dict[str, Any]]
    diagnostics: RetrievalDiagnostics
