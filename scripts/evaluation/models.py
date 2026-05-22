"""Evaluation domain models."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    category: str


@dataclass
class EvaluationResult:
    query: str
    category: str
    raw_distances: List[float]
    retrieved_sources: List[str]
    retrieved_distances: List[float]
    answer: str
    verification_result: bool
    retrieval_status: str
    generation_status: str
    best_distance: Optional[float]
    raw_best_distance: Optional[float]
    retrieved_count: int


@dataclass
class ThresholdEvaluationMetrics:
    threshold: float
    total_retrieved_documents: int
    average_retrieved_documents: float
    supported_answers: int
    unsupported_answers: int
    rejected_queries: int
    average_best_distance: float
    results: List[EvaluationResult]
