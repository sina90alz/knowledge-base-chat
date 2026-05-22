from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Dict, List

from app.core.config import settings

if TYPE_CHECKING:
    from scripts.evaluation.models import EvaluationResult, ThresholdEvaluationMetrics

logger = logging.getLogger(__name__)


def build_markdown_report(
    baseline_metrics: "ThresholdEvaluationMetrics",
    sweep_metrics: List["ThresholdEvaluationMetrics"],
    report_path: Path,
    vector_count: int,
) -> None:
    """Build and write the Markdown evaluation report."""
    baseline_results = baseline_metrics.results
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
    ]
    lines.extend(_format_system_configuration(vector_count))
    lines.append("")
    lines.append("## Evaluation Queries")
    lines.append("")

    for result in baseline_results:
        lines.extend(_format_query_section(result))

    lines.extend(_format_threshold_calibration_table(sweep_metrics))
    lines.append("")
    lines.extend(_format_evaluation_summary(total_queries, supported_answers, unsupported_answers, skipped_generations, average_best_distance))
    lines.append("")
    lines.extend(_format_key_findings_section(baseline_metrics, sweep_metrics))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown evaluation report written to %s", report_path)


def _format_system_configuration(vector_count: int) -> List[str]:
    """Return the system configuration section lines."""
    llm_model = (
        settings.OPENAI_MODEL
        if settings.LLM_PROVIDER.strip().lower() == "openai"
        else settings.OLLAMA_MODEL
    )

    return [
        "## System Configuration",
        "",
        f"- Embedding model: `{settings.EMBEDDING_MODEL}`",
        f"- LLM provider: `{settings.LLM_PROVIDER}`",
        f"- LLM model: `{llm_model}`",
        f"- Similarity threshold: `{settings.SIMILARITY_THRESHOLD}`",
        f"- Vector count: `{vector_count}`",
        "",
    ]


def _format_query_section(result: "EvaluationResult") -> List[str]:
    """Return the markdown lines for a single evaluation query."""
    lines: List[str] = [
        f"### Query: {result.query}",
        "",
        f"- Category: `{result.category}`",
        f"- Retrieval status: `{result.retrieval_status}`",
        f"- Generation status: `{result.generation_status}`",
        f"- Verification result: `{('SUPPORTED' if result.verification_result else 'UNSUPPORTED')}`",
        f"- Best distance: `{result.best_distance:.4f}`" if result.best_distance is not None else "- Best distance: `N/A`",
        "- Retrieved source files:",
    ]

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
    return lines


def _format_threshold_calibration_table(sweep_metrics: List["ThresholdEvaluationMetrics"]) -> List[str]:
    """Return the threshold calibration table section lines."""
    lines: List[str] = [
        "## Threshold Calibration Results",
        "",
        "| Threshold | Supported | Unsupported | Avg Distance | Retrieved Docs |",
        "|---|---|---|---|---|",
    ]

    for metric in sweep_metrics:
        lines.append(
            f"| {metric.threshold:.2f} | {metric.supported_answers} | {metric.unsupported_answers} | {metric.average_best_distance:.4f} | {metric.total_retrieved_documents} |"
        )

    return lines


def _format_evaluation_summary(
    total_queries: int,
    supported_answers: int,
    unsupported_answers: int,
    skipped_generations: int,
    average_best_distance: float,
) -> List[str]:
    """Return the evaluation summary section lines."""
    return [
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
    ]


def _format_key_findings_section(
    baseline_metrics: "ThresholdEvaluationMetrics",
    sweep_metrics: List["ThresholdEvaluationMetrics"],
) -> List[str]:
    """Return the key findings section lines."""
    findings = generate_key_findings(baseline_metrics, sweep_metrics)
    lines: List[str] = []

    if not findings:
        lines.append("- No strong findings identified. The calibration results are consistent across thresholds.")
    else:
        for finding in findings:
            lines.append(f"- {finding}")

    return lines


def generate_key_findings(
    baseline_metrics: "ThresholdEvaluationMetrics",
    sweep_metrics: List["ThresholdEvaluationMetrics"],
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

    strong_support_thresholds = [
        item.threshold for item in sweep_metrics if item.supported_answers >= baseline_metrics.supported_answers
    ]
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
