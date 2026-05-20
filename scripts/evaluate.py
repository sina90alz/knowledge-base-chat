"""Evaluate retrieval quality, generation quality, and grounding resistance.

This script uses the existing RAG pipeline services to run a small
set of evaluation queries and print structured results.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports when run from the scripts folder.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService
from app.services.llm import get_llm_service
from app.services.retrieval import RetrievalService
import app.services.retrieval as retrieval_module
from app.services.verification import AnswerVerificationService
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)

EVALUATION_K = 5


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


def build_test_cases() -> List[EvaluationCase]:
    """Return a small set of evaluation queries across common categories."""
    return [
        EvaluationCase(
            query="What is the mission or main purpose described in the documents?",
            category="factual",
        ),
        EvaluationCase(
            query="Describe how document ingestion works in this system.",
            category="factual",
        ),
        EvaluationCase(
            query="Which service verifies whether answers are grounded in the retrieved context?",
            category="factual",
        ),
        EvaluationCase(
            query="What should the system do if no relevant documents are available?",
            category="edge",
        ),
        EvaluationCase(
            query="How can I switch between OpenAI and Ollama providers?",
            category="ambiguous",
        ),
        EvaluationCase(
            query="What does grounding mean in the context of this retrieval system?",
            category="ambiguous",
        ),
        EvaluationCase(
            query="What environment variable controls the vector store path?",
            category="factual",
        ),
        EvaluationCase(
            query="How many chunks are retrieved by default if I ask a question?",
            category="edge",
        ),
        EvaluationCase(
            query="What is the weather in Paris today?",
            category="unrelated",
        ),
        EvaluationCase(
            query="Describe a question that cannot be answered from the available documents.",
            category="unrelated",
        ),
    ]


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
    filter_threshold: float,
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

    context = retrieval_service.format_context(retrieved_documents, retrieved_metadata)
    prompt = retrieval_service.generate_prompt(query=query, context=context)

    try:
        answer = llm_service.generate(prompt)
    except Exception as exc:
        logger.error("LLM generation failed for query '%s': %s", query, exc)
        answer = f"ERROR: {exc}"

    verification_result = False
    if context:
        try:
            verification_result = verification_service.verify_answer(
                question=query,
                context=context,
                answer=answer,
            )
        except Exception as exc:
            logger.error("Answer verification failed for query '%s': %s", query, exc)
            verification_result = False

    retrieval_status = assess_retrieval_status(raw_distances, filter_threshold)
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


def evaluate_threshold(
    threshold: float,
    cases: List[EvaluationCase],
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
    retrieval_service: RetrievalService,
    llm_service: Any,
    verification_service: AnswerVerificationService,
) -> ThresholdEvaluationMetrics:
    """Run the full evaluation suite under a specific retrieval threshold."""
    logger.info("%s", "#" * 80)
    logger.info("Evaluating threshold: %.2f", threshold)
    old_threshold = retrieval_module.SIMILARITY_THRESHOLD
    retrieval_module.SIMILARITY_THRESHOLD = threshold

    try:
        results: List[EvaluationResult] = []
        for case in cases:
            result = evaluate_query(
                query=case.query,
                category=case.category,
                filter_threshold=threshold,
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

        logger.info("Threshold %.2f evaluation complete. Supported=%s Unsupported=%s Rejected=%s",
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


def build_markdown_report(
    baseline_results: List[EvaluationResult],
    sweep_metrics: List[ThresholdEvaluationMetrics],
    report_path: Path,
    vector_count: int,
) -> None:
    """Build and write the Markdown evaluation report."""
    total_queries = len(baseline_results)
    supported_answers = sum(1 for result in baseline_results if result.verification_result)
    unsupported_answers = total_queries - supported_answers
    skipped_generations = sum(1 for result in baseline_results if result.answer.startswith("ERROR:"))
    average_best_distance = mean(
        [result.raw_best_distance for result in baseline_results if result.raw_best_distance is not None]
    ) if any(result.raw_best_distance is not None for result in baseline_results) else float("nan")

    lines: List[str] = [
        "# RAG Evaluation Report",
        "",
        "## System Configuration",
        "",
        f"- Embedding model: `{settings.EMBEDDING_MODEL}`",
        f"- LLM provider: `{settings.LLM_PROVIDER}`",
        f"- LLM model: `{settings.OPENAI_MODEL if settings.LLM_PROVIDER.strip().lower() == 'openai' else settings.OLLAMA_MODEL}`",
        f"- Similarity threshold: `{settings.SIMILARITY_THRESHOLD}`",
        f"- Vector count: `{vector_count}`",
        "",
        "## Evaluation Queries",
        "",
    ]

    for result in baseline_results:
        lines.extend([
            f"### Query: {result.query}",
            "",
            f"- Category: `{result.category}`",
            f"- Retrieval status: `{result.retrieval_status}`",
            f"- Generation status: `{result.generation_status}`",
            f"- Verification result: `{('SUPPORTED' if result.verification_result else 'UNSUPPORTED')}`",
            f"- Best distance: `{result.best_distance:.4f}`" if result.best_distance is not None else "- Best distance: `N/A`",
            "- Retrieved source files:",
        ])

        if result.retrieved_sources:
            for source in result.retrieved_sources:
                lines.append(f"  - `{source}`")
        else:
            lines.append("  - `No sources retrieved`")

        lines.extend([
            "",
            "**Generated answer:**",
            "",
            "```",
            result.answer,
            "```",
            "",
        ])

    lines.extend([
        "## Threshold Calibration Results",
        "",
        "| Threshold | Supported | Unsupported | Avg Distance | Retrieved Docs |",
        "|---|---|---|---|---|",
    ])

    for metric in sweep_metrics:
        lines.append(
            f"| {metric.threshold:.2f} | {metric.supported_answers} | {metric.unsupported_answers} | {metric.average_best_distance:.4f} | {metric.total_retrieved_documents} |"
        )

    lines.extend([
        "",
        "## Evaluation Summary",
        "",
        f"- Total queries: `{total_queries}`",
        f"- Supported answers: `{supported_answers}`",
        f"- Unsupported answers: `{unsupported_answers}`",
        f"- Skipped generations: `{skipped_generations}`",
        f"- Average best distance: `{average_best_distance:.4f}`",
        "",
        "## Key Findings",
        "",
    ])

    findings = generate_key_findings(baseline_metrics, sweep_metrics)
    if not findings:
        lines.append("- No strong findings identified. The calibration results are consistent across thresholds.")
    else:
        for finding in findings:
            lines.append(f"- {finding}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown evaluation report written to %s", report_path)


def generate_key_findings(
    baseline_metrics: ThresholdEvaluationMetrics,
    sweep_metrics: List[ThresholdEvaluationMetrics],
) -> List[str]:
    """Generate short report observations based on calibration results."""
    findings: List[str] = []

    if baseline_metrics.rejected_queries > 0:
        findings.append(
            "The default threshold appears too strict for some queries, causing rejected or weak retrievals."
        )

    best_supported = max(sweep_metrics, key=lambda item: (item.supported_answers, -item.unsupported_answers))
    if best_supported.threshold != baseline_metrics.threshold:
        findings.append(
            f"A non-default threshold (`{best_supported.threshold:.2f}`) achieved the highest supported answer count."
        )

    strong_support_thresholds = [item.threshold for item in sweep_metrics if item.supported_answers >= baseline_metrics.supported_answers]
    if strong_support_thresholds and baseline_metrics.threshold not in strong_support_thresholds:
        findings.append(
            "Retrieval quality improves after calibration, with broader thresholds yielding more supportable results."
        )

    hallucination_ratio = baseline_metrics.unsupported_answers / max(1, len(baseline_metrics.results))
    if hallucination_ratio < 0.5:
        findings.append("The hallucination guard is generally effective, with more than half of answers verified as supported.")
    else:
        findings.append("The hallucination guard may need refinement, because many generated answers are unsupported.")

    return findings


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
            baseline_results=baseline_metrics.results,
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
