"""Threshold calibration utilities for RAG evaluation."""

import logging
from statistics import mean
from typing import Any, Callable, List, Protocol

import app.services.retrieval as retrieval_module
from app.ingestion.embedder import EmbeddingService
from app.services.retrieval import RetrievalService
from app.services.verification import AnswerVerificationService
from app.vectorstore.faiss_store import FAISSVectorStore
from scripts.evaluation.models import (
    Difficulty,
    EvaluationCase,
    EvaluationResult,
    ExpectedBehavior,
    ThresholdEvaluationMetrics,
)

logger = logging.getLogger(__name__)

PrintCaseResultFn = Callable[[EvaluationResult], None]


class EvaluateCaseFn(Protocol):
    def __call__(
        self,
        *,
        query: str,
        category: str,
        expected_keywords: List[str],
        expected_source: str | None,
        expected_behavior: ExpectedBehavior | None,
        difficulty: Difficulty,
        notes: str | None,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        retrieval_service: RetrievalService,
        llm_service: Any,
        verification_service: AnswerVerificationService,
    ) -> EvaluationResult:
        ...


def evaluate_threshold(
    threshold: float,
    cases: List[EvaluationCase],
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
    retrieval_service: RetrievalService,
    llm_service: Any,
    verification_service: AnswerVerificationService,
    evaluate_case: EvaluateCaseFn,
    print_case_result: PrintCaseResultFn,
) -> ThresholdEvaluationMetrics:
    """Run the full evaluation suite under a specific retrieval threshold."""
    logger.info("%s", "#" * 80)
    logger.info("Evaluating threshold: %.2f", threshold)
    old_threshold = retrieval_module.SIMILARITY_THRESHOLD
    retrieval_module.SIMILARITY_THRESHOLD = threshold

    try:
        results: List[EvaluationResult] = []
        for case in cases:
            result = evaluate_case(
                query=case.query,
                category=case.category,
                expected_keywords=case.expected_keywords,
                expected_source=case.expected_source,
                expected_behavior=case.expected_behavior,
                difficulty=case.difficulty,
                notes=case.notes,
                embedding_service=embedding_service,
                vector_store=vector_store,
                retrieval_service=retrieval_service,
                llm_service=llm_service,
                verification_service=verification_service,
            )
            results.append(result)
            print_case_result(result)

        total_retrieved_documents = sum(result.retrieved_count for result in results)
        average_retrieved_documents = total_retrieved_documents / len(results) if results else 0.0
        supported_answers = sum(1 for result in results if result.verification_result)
        unsupported_answers = len(results) - supported_answers
        rejected_queries = sum(1 for result in results if result.retrieved_count == 0)
        best_distances = [result.raw_best_distance for result in results if result.raw_best_distance is not None]
        average_best_distance = mean(best_distances) if best_distances else float("nan")

        logger.info(
            "Threshold %.2f evaluation complete. Supported=%s Unsupported=%s Rejected=%s",
            threshold,
            supported_answers,
            unsupported_answers,
            rejected_queries,
        )

        return ThresholdEvaluationMetrics(
            threshold=threshold,
            total_retrieved_documents=total_retrieved_documents,
            average_retrieved_documents=average_retrieved_documents,
            supported_answers=supported_answers,
            unsupported_answers=unsupported_answers,
            rejected_queries=rejected_queries,
            average_best_distance=average_best_distance,
            results=results,
        )
    finally:
        retrieval_module.SIMILARITY_THRESHOLD = old_threshold


def print_threshold_comparison_table(metrics: List[ThresholdEvaluationMetrics]) -> None:
    """Print a compact comparison table for threshold sweep results."""
    logger.info("%s", "=" * 80)
    logger.info("THRESHOLD COMPARISON")
    logger.info(
        "%-10s %-20s %-18s %-18s %-16s %-18s",
        "Threshold",
        "TotalRetrieved",
        "AvgRetrieved",
        "Supported",
        "Rejected",
        "AvgBestDist",
    )

    for metric in metrics:
        logger.info(
            "%-10.2f %-20d %-18.2f %-18d %-16d %-18.4f",
            metric.threshold,
            metric.total_retrieved_documents,
            metric.average_retrieved_documents,
            metric.supported_answers,
            metric.rejected_queries,
            metric.average_best_distance,
        )
    logger.info("%s", "=" * 80)
