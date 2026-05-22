"""Evaluate retrieval quality, generation quality, and grounding resistance.

This script uses the existing RAG pipeline services to run a small
set of evaluation queries and print structured results.
"""

import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports when run from the scripts folder.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService
from app.services.llm import get_llm_service
from app.services.retrieval import RetrievalService
from app.services.verification import AnswerVerificationService
from app.vectorstore.faiss_store import FAISSVectorStore
from scripts.evaluation.datasets import build_test_cases
from scripts.evaluation.models import EvaluationCase, EvaluationResult, ThresholdEvaluationMetrics
from scripts.evaluation.reporting import build_markdown_report
from scripts.evaluation.thresholding import evaluate_threshold, print_threshold_comparison_table

logger = logging.getLogger(__name__)

EVALUATION_K = 5


def initialize_services() -> Tuple[EmbeddingService, FAISSVectorStore, RetrievalService, Any, AnswerVerificationService]:
    """Initialize the evaluation pipeline services."""
    logger.info("Initializing evaluation services")

    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    vector_store = FAISSVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        store_path=settings.VECTOR_STORE_PATH,
    )
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    llm_service = get_llm_service()
    verification_service = AnswerVerificationService(llm_service)

    logger.info("Services initialized successfully")
    logger.info("Vector store stats: %s", vector_store.get_stats())

    return embedding_service, vector_store, retrieval_service, llm_service, verification_service


def format_sources(metadata_list: List[Dict[str, Any]], distances: List[float]) -> List[str]:
    """Format source and distance information for display."""
    sources: List[str] = []
    for metadata, distance in zip(metadata_list, distances):
        source = metadata.get("filename") or metadata.get("source") or "Unknown"
        page = metadata.get("page") or metadata.get("page_number") or "N/A"
        sources.append(f"{source} (page={page}) - dist={distance:.4f}")
    return sources


def assess_retrieval_status(
    raw_distances: List[float],
    threshold: float,
) -> str:
    """Return a retrieval quality label based on the best available distance."""
    if not raw_distances:
        return "NO_RESULTS"

    best_distance = min(raw_distances)
    if best_distance > threshold:
        return "WEAK"

    return "GOOD"


def compute_distance_stats(distances: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute min, max, and average statistics for distances."""
    if not distances:
        return None, None, None

    return min(distances), max(distances), mean(distances)


def evaluate_query(
    query: str,
    category: str,
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
    retrieval_service: RetrievalService,
    llm_service: Any,
    verification_service: AnswerVerificationService,
) -> EvaluationResult:
    """Run retrieval, generation, and grounding verification for a single query."""
    logger.info("Evaluating query: %s", query)

    query_embedding = embedding_service.embed_text(query)
    raw_documents, raw_distances, raw_metadata = vector_store.search(query_embedding, k=EVALUATION_K)
    raw_min, raw_max, raw_avg = compute_distance_stats(raw_distances)

    logger.info(
        "Raw retrieval distances: %s",
        ", ".join(f"{distance:.4f}" for distance in raw_distances) if raw_distances else "N/A",
    )
    if raw_distances:
        logger.info(
            "Raw distance stats: min=%.4f max=%.4f avg=%.4f",
            raw_min,
            raw_max,
            raw_avg,
        )

    retrieved_documents, retrieved_distances, retrieved_metadata = retrieval_service.retrieve_context(
        query=query,
        k=EVALUATION_K,
    )

    retrieval_status = retrieval_service.get_retrieval_quality(
        raw_distances=raw_distances,
        filtered_count=len(retrieved_documents),
    )

    context = retrieval_service.format_context(retrieved_documents, retrieved_metadata)
    answer = "I don't know based on the available documents."
    verification_result = False
    generation_status = "UNSUPPORTED"

    if retrieval_status == "REJECTED":
        logger.info(
            "Skipping generation due to insufficient retrieval context for query: %s",
            query,
        )
    else:
        if retrieval_status == "WEAK":
            logger.warning(
                "Weak retrieval quality for query '%s'; proceeding with caution.",
                query,
            )

        prompt = retrieval_service.generate_prompt(query=query, context=context)

        try:
            answer = llm_service.generate(prompt)
        except Exception as exc:
            logger.error("LLM generation failed for query '%s': %s", query, exc)
            answer = f"ERROR: {exc}"

        if context and settings.ENABLE_ANSWER_VERIFICATION:
            try:
                verification_result = verification_service.verify_answer(
                    question=query,
                    context=context,
                    answer=answer,
                )
            except Exception as exc:
                logger.error("Answer verification failed for query '%s': %s", query, exc)
                verification_result = False

        generation_status = "SUPPORTED" if verification_result else "UNSUPPORTED"

    best_distance = min(retrieved_distances) if retrieved_distances else None
    raw_best_distance = min(raw_distances) if raw_distances else None
    retrieved_sources = format_sources(retrieved_metadata, retrieved_distances)

    return EvaluationResult(
        query=query,
        category=category,
        raw_distances=raw_distances,
        retrieved_sources=retrieved_sources,
        retrieved_distances=retrieved_distances,
        answer=answer.strip(),
        verification_result=verification_result,
        retrieval_status=retrieval_status,
        generation_status=generation_status,
        best_distance=best_distance,
        raw_best_distance=raw_best_distance,
        retrieved_count=len(retrieved_documents),
    )


def print_case_result(result: EvaluationResult) -> None:
    """Print detailed evaluation output for one query."""
    logger.info("%s", "=" * 80)
    logger.info("Query: %s", result.query)
    logger.info("Category: %s", result.category)
    logger.info("Retrieval status: %s", result.retrieval_status)
    logger.info("Generation status: %s", result.generation_status)
    logger.info("Verification result: %s", "SUPPORTED" if result.verification_result else "UNSUPPORTED")
    logger.info("Retrieved documents: %s", result.retrieved_count)
    logger.info("Raw best distance: %s", f"{result.raw_best_distance:.4f}" if result.raw_best_distance is not None else "N/A")

    if result.raw_distances:
        logger.info(
            "Raw retrieval distances: %s",
            ", ".join(f"{distance:.4f}" for distance in result.raw_distances),
        )
        raw_min, raw_max, raw_avg = compute_distance_stats(result.raw_distances)
        logger.info(
            "Raw distance stats: min=%.4f max=%.4f avg=%.4f",
            raw_min,
            raw_max,
            raw_avg,
        )

    if result.retrieved_sources:
        for source_line in result.retrieved_sources:
            logger.info("  - %s", source_line)
    else:
        logger.info("  - No retrieved documents available")

    logger.info("Answer: %s", result.answer)


def print_summary(results: List[EvaluationResult]) -> None:
    """Print a summary of evaluation metrics."""
    total_queries = len(results)
    supported_answers = sum(1 for result in results if result.verification_result)
    unsupported_answers = total_queries - supported_answers
    no_result_queries = sum(1 for result in results if result.retrieval_status == "NO_RESULTS")
    valid_distances = [result.raw_best_distance for result in results if result.raw_best_distance is not None]
    average_best_distance = mean(valid_distances) if valid_distances else float("nan")

    logger.info("%s", "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("Total queries: %s", total_queries)
    logger.info("Supported answers: %s", supported_answers)
    logger.info("Unsupported answers: %s", unsupported_answers)
    logger.info("No-result queries: %s", no_result_queries)
    logger.info(
        "Average best distance: %s",
        f"{average_best_distance:.4f}" if valid_distances else "N/A",
    )
    logger.info("%s", "=" * 80)


def main() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cases = build_test_cases()
    embedding_service, vector_store, retrieval_service, llm_service, verification_service = initialize_services()

    thresholds = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    sweep_metrics: List[ThresholdEvaluationMetrics] = []

    for threshold in thresholds:
        threshold_metrics = evaluate_threshold(
            threshold=threshold,
            cases=cases,
            embedding_service=embedding_service,
            vector_store=vector_store,
            retrieval_service=retrieval_service,
            llm_service=llm_service,
            verification_service=verification_service,
            evaluate_case=evaluate_query,
            print_case_result=print_case_result,
        )
        sweep_metrics.append(threshold_metrics)

    print_threshold_comparison_table(sweep_metrics)

    default_threshold = settings.SIMILARITY_THRESHOLD
    baseline_metrics = next(
        (metric for metric in sweep_metrics if abs(metric.threshold - default_threshold) < 1e-6),
        sweep_metrics[0] if sweep_metrics else None,
    )

    if baseline_metrics:
        report_path = settings.PROJECT_ROOT / "reports" / "evaluation_report.md"
        build_markdown_report(
            baseline_metrics=baseline_metrics,
            sweep_metrics=sweep_metrics,
            report_path=report_path,
            vector_count=vector_store.get_stats().get("total_vectors", 0),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Evaluation script failed: %s", exc)
        sys.exit(1)
