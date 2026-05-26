from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, List, Sequence

from app.core.config import settings

if TYPE_CHECKING:
    from scripts.evaluation.models import EvaluationResult, ThresholdEvaluationMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeywordMatch:
    """Keyword coverage details for one generated answer."""

    matched_keywords: List[str]
    matched_count: int
    total_keywords: int
    coverage_ratio: float


@dataclass(frozen=True)
class RetrievalAccuracyMetrics:
    """Expected-source retrieval metrics."""

    total_source_aware_queries: int
    successful_source_retrievals: int
    missing_source_results: List["EvaluationResult"]

    @property
    def accuracy_ratio(self) -> float:
        if self.total_source_aware_queries == 0:
            return 0.0
        return self.successful_source_retrievals / self.total_source_aware_queries


@dataclass(frozen=True)
class GroundingMetrics:
    """Expected-keyword grounding metrics."""

    keyword_aware_queries: int
    average_keyword_coverage: float
    fully_grounded_answers: int
    partially_grounded_answers: int
    weakly_grounded_answers: int


@dataclass(frozen=True)
class RefusalMetrics:
    """Hallucination-resistance metrics for expected refusals."""

    refusal_tests: int
    correct_refusals: int
    hallucinated_responses: int

    @property
    def success_ratio(self) -> float:
        if self.refusal_tests == 0:
            return 0.0
        return self.correct_refusals / self.refusal_tests


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
    average_best_distance = _average_best_distance(baseline_results)

    lines: List[str] = [
        "# RAG Evaluation Report",
        "",
    ]
    lines.extend(_format_system_configuration(vector_count))
    lines.append("")
    lines.extend(_format_evaluation_summary(total_queries, supported_answers, unsupported_answers, skipped_generations, average_best_distance))
    lines.append("")
    lines.extend(_format_dataset_statistics(baseline_results))
    lines.append("")
    lines.extend(_format_retrieval_accuracy_section(baseline_results))
    lines.append("")
    lines.extend(_format_grounding_metrics_section(baseline_results))
    lines.append("")
    lines.extend(_format_hallucination_resistance_section(baseline_results))
    lines.append("")
    lines.extend(_format_performance_breakdowns(baseline_results))
    lines.append("")
    lines.append("## Evaluation Queries")
    lines.append("")

    for result in baseline_results:
        lines.extend(_format_query_section(result))

    lines.extend(_format_threshold_calibration_table(sweep_metrics))
    lines.append("")
    lines.extend(_format_failure_analysis_section(baseline_metrics, sweep_metrics))
    lines.append("")
    lines.extend(_format_best_worst_sections(baseline_results))
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
        f"- Average best distance: `{_format_float(average_best_distance)}`",
    ]


def _format_dataset_statistics(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return dataset composition tables."""
    category_counts = Counter(result.category for result in results)
    difficulty_counts = Counter(result.difficulty for result in results)
    refusal_tests = sum(1 for result in results if result.expected_behavior == "refusal")
    repository_grounded_tests = sum(1 for result in results if result.expected_source)

    lines = [
        "## Dataset Statistics",
        "",
        f"- Refusal tests: `{refusal_tests}`",
        f"- Repository-grounded tests: `{repository_grounded_tests}`",
        "",
        "### Queries by Category",
        "",
        "| Category | Queries |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | {count} |")

    lines.extend([
        "",
        "### Queries by Difficulty",
        "",
        "| Difficulty | Queries |",
        "|---|---:|",
    ])
    for difficulty, count in sorted(difficulty_counts.items()):
        lines.append(f"| `{difficulty}` | {count} |")

    return lines


def _format_retrieval_accuracy_section(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return expected-source retrieval accuracy details."""
    metrics = _compute_retrieval_accuracy(results)
    lines = [
        "## Retrieval Accuracy",
        "",
        f"- Total source-aware queries: `{metrics.total_source_aware_queries}`",
        f"- Successful source retrievals: `{metrics.successful_source_retrievals}`",
        f"- Queries missing expected source: `{len(metrics.missing_source_results)}`",
        f"- Source retrieval accuracy: `{_format_percent(metrics.accuracy_ratio)}`",
    ]

    if metrics.missing_source_results:
        lines.extend([
            "",
            "### Missing Expected Sources",
            "",
            "| Query | Expected Source | Retrieval Status | Best Distance |",
            "|---|---|---|---:|",
        ])
        for result in metrics.missing_source_results:
            lines.append(
                f"| {_escape_table_cell(result.query)} | `{result.expected_source}` | "
                f"`{result.retrieval_status}` | {_format_optional_float(result.raw_best_distance)} |"
            )

    return lines


def _format_grounding_metrics_section(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return expected-keyword grounding metrics."""
    metrics = _compute_grounding_metrics(results)
    return [
        "## Grounding Metrics",
        "",
        f"- Keyword-aware queries: `{metrics.keyword_aware_queries}`",
        f"- Average keyword coverage: `{_format_percent(metrics.average_keyword_coverage)}`",
        f"- Fully grounded answers: `{metrics.fully_grounded_answers}`",
        f"- Partially grounded answers: `{metrics.partially_grounded_answers}`",
        f"- Weakly grounded answers: `{metrics.weakly_grounded_answers}`",
    ]


def _format_hallucination_resistance_section(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return refusal correctness metrics for hallucination tests."""
    metrics = _compute_refusal_metrics(results)
    lines = [
        "## Hallucination Resistance",
        "",
        f"- Refusal tests: `{metrics.refusal_tests}`",
        f"- Correct refusals: `{metrics.correct_refusals}`",
        f"- Hallucinated responses: `{metrics.hallucinated_responses}`",
        f"- Refusal success rate: `{_format_percent(metrics.success_ratio)}`",
    ]

    hallucinated = [
        result
        for result in results
        if result.expected_behavior == "refusal" and not _is_correct_refusal(result)
    ]
    if hallucinated:
        lines.extend([
            "",
            "### Hallucinated Refusal Tests",
            "",
            "| Query | Retrieval Status | Verification | Best Distance |",
            "|---|---|---|---:|",
        ])
        for result in hallucinated:
            lines.append(
                f"| {_escape_table_cell(result.query)} | `{result.retrieval_status}` | "
                f"`{_verification_label(result)}` | {_format_optional_float(result.raw_best_distance)} |"
            )

    return lines


def _format_performance_breakdowns(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return category and difficulty performance tables."""
    lines = [
        "## Performance Breakdowns",
        "",
        "### Performance by Category",
        "",
        "| Category | Queries | Supported | Unsupported | Success Rate | Avg Keyword Coverage | Source Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for category in sorted({result.category for result in results}):
        group = [result for result in results if result.category == category]
        lines.append(_format_breakdown_row(category, group))

    lines.extend([
        "",
        "### Performance by Difficulty",
        "",
        "| Difficulty | Queries | Supported | Unsupported | Accuracy | Avg Keyword Coverage | Source Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for difficulty in sorted({result.difficulty for result in results}):
        group = [result for result in results if result.difficulty == difficulty]
        lines.append(_format_breakdown_row(difficulty, group))

    return lines


def _format_query_section(result: "EvaluationResult") -> List[str]:
    """Return the markdown lines for a single evaluation query."""
    keyword_match = _compute_keyword_match(result)

    lines: List[str] = [
        f"### Query: {result.query}",
        "",
        f"- Category: `{result.category}`",
        f"- Difficulty: `{result.difficulty}`",
        f"- Expected behavior: `{result.expected_behavior or 'N/A'}`",
        f"- Expected source: `{result.expected_source or 'N/A'}`",
        f"- Retrieved expected source: `{_format_expected_source_hit(result)}`",
        f"- Keyword coverage: `{_format_percent(keyword_match.coverage_ratio)}` "
        f"({keyword_match.matched_count}/{keyword_match.total_keywords})",
        f"- Retrieval status: `{result.retrieval_status}`",
        f"- Generation status: `{result.generation_status}`",
        f"- Verification result: `{_verification_label(result)}`",
        f"- Best distance: `{_format_optional_float(result.best_distance)}`",
        "- Retrieved source files:",
    ]

    if result.retrieved_sources:
        for source in result.retrieved_sources:
            lines.append(f"  - `{source}`")
    else:
        lines.append("  - `No sources retrieved`")

    if result.expected_keywords:
        lines.extend([
            "",
            f"- Expected keywords: `{', '.join(result.expected_keywords)}`",
            f"- Matched keywords: `{', '.join(keyword_match.matched_keywords) or 'None'}`",
        ])

    if result.notes:
        lines.append(f"- Notes: {result.notes}")

    lines.extend([
        "",
        "**Generated answer:**",
        "",
        "```",
        result.answer,
        "```",
        "",
        "---",
        "",
    ])
    return lines


def _format_threshold_calibration_table(sweep_metrics: List["ThresholdEvaluationMetrics"]) -> List[str]:
    """Return the threshold calibration table section lines."""
    lines: List[str] = [
        "## Threshold Calibration Results",
        "",
        "| Threshold | Supported | Unsupported | Rejected | Avg Distance | Retrieved Docs | Avg Retrieved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for metric in sweep_metrics:
        lines.append(
            f"| {metric.threshold:.2f} | {metric.supported_answers} | "
            f"{metric.unsupported_answers} | {metric.rejected_queries} | "
            f"{_format_float(metric.average_best_distance)} | "
            f"{metric.total_retrieved_documents} | {metric.average_retrieved_documents:.2f} |"
        )

    return lines


def _format_failure_analysis_section(
    baseline_metrics: "ThresholdEvaluationMetrics",
    sweep_metrics: List["ThresholdEvaluationMetrics"],
) -> List[str]:
    """Return automatic failure diagnostics."""
    results = baseline_metrics.results
    missing_source_results = _compute_retrieval_accuracy(results).missing_source_results
    weak_or_rejected = [
        result
        for result in results
        if result.retrieval_status in {"WEAK", "REJECTED", "NO_RESULTS"}
    ]
    unsupported = [result for result in results if not result.verification_result]

    lines = [
        "## Failure Analysis",
        "",
    ]

    if not missing_source_results and not weak_or_rejected and not unsupported:
        lines.append("- No major retrieval, threshold, or grounding failures were detected.")
        return lines

    if missing_source_results:
        source_counts = Counter(result.expected_source or "Unknown" for result in missing_source_results)
        lines.append(
            f"- Expected source misses affected `{len(missing_source_results)}` queries, "
            f"most often: {_format_counter_summary(source_counts)}."
        )
    else:
        lines.append("- Expected-source retrieval succeeded for all source-aware queries.")

    if weak_or_rejected:
        status_counts = Counter(result.retrieval_status for result in weak_or_rejected)
        category_counts = Counter(result.category for result in weak_or_rejected)
        lines.append(
            f"- Weak retrieval patterns: {_format_counter_summary(status_counts)}; "
            f"most affected categories: {_format_counter_summary(category_counts)}."
        )

    if unsupported:
        category_counts = Counter(result.category for result in unsupported)
        difficulty_counts = Counter(result.difficulty for result in unsupported)
        lines.append(
            f"- Unsupported answer trends by category: {_format_counter_summary(category_counts)}."
        )
        lines.append(
            f"- Unsupported answer trends by difficulty: {_format_counter_summary(difficulty_counts)}."
        )

    baseline_rejected = baseline_metrics.rejected_queries
    best_supported = _best_supported_threshold(sweep_metrics)
    if best_supported and best_supported.threshold != baseline_metrics.threshold:
        lines.append(
            f"- Threshold-related failures should be reviewed: threshold `{best_supported.threshold:.2f}` "
            f"produced `{best_supported.supported_answers}` supported answers versus "
            f"`{baseline_metrics.supported_answers}` at baseline `{baseline_metrics.threshold:.2f}`."
        )
    elif baseline_rejected:
        lines.append(
            f"- Baseline threshold `{baseline_metrics.threshold:.2f}` rejected `{baseline_rejected}` queries."
        )

    return lines


def _format_best_worst_sections(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return strongest and weakest query diagnostics."""
    lines = [
        "## Best and Worst Performing Queries",
        "",
    ]
    lines.extend(_format_strongest_grounded_queries(results))
    lines.append("")
    lines.extend(_format_weakest_retrieval_queries(results))
    lines.append("")
    lines.extend(_format_highest_distance_failures(results))
    return lines


def _format_strongest_grounded_queries(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return top keyword-grounded answers."""
    candidates = [
        result
        for result in results
        if result.expected_keywords and result.verification_result
    ]
    ranked = sorted(
        candidates,
        key=lambda result: (
            _compute_keyword_match(result).coverage_ratio,
            _expected_source_retrieved(result),
            -(result.raw_best_distance or 0.0),
        ),
        reverse=True,
    )[:5]

    lines = [
        "### Strongest Grounded Queries",
        "",
        "| Query | Category | Difficulty | Keyword Coverage | Expected Source Retrieved |",
        "|---|---|---|---:|---|",
    ]
    if not ranked:
        lines.append("| No supported keyword-grounded queries found. |  |  |  |  |")
        return lines

    for result in ranked:
        lines.append(
            f"| {_escape_table_cell(result.query)} | `{result.category}` | "
            f"`{result.difficulty}` | {_format_percent(_compute_keyword_match(result).coverage_ratio)} | "
            f"`{_format_expected_source_hit(result)}` |"
        )

    return lines


def _format_weakest_retrieval_queries(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return queries with weak retrieval signals."""
    candidates = [
        result
        for result in results
        if result.retrieval_status in {"WEAK", "REJECTED", "NO_RESULTS"}
        or (result.expected_source and not _expected_source_retrieved(result))
    ]
    ranked = sorted(
        candidates,
        key=lambda result: (
            result.retrieval_status not in {"WEAK", "REJECTED", "NO_RESULTS"},
            _compute_keyword_match(result).coverage_ratio,
            -(result.raw_best_distance or 0.0),
        ),
    )[:5]

    lines = [
        "### Weakest Retrieval Queries",
        "",
        "| Query | Expected Source | Retrieved Expected Source | Retrieval Status | Best Distance |",
        "|---|---|---|---|---:|",
    ]
    if not ranked:
        lines.append("| No weak retrieval queries found. |  |  |  |  |")
        return lines

    for result in ranked:
        lines.append(
            f"| {_escape_table_cell(result.query)} | `{result.expected_source or 'N/A'}` | "
            f"`{_format_expected_source_hit(result)}` | "
            f"`{result.retrieval_status}` | {_format_optional_float(result.raw_best_distance)} |"
        )

    return lines


def _format_highest_distance_failures(results: Sequence["EvaluationResult"]) -> List[str]:
    """Return unsupported failures with the largest best distance."""
    candidates = [
        result
        for result in results
        if not result.verification_result and result.raw_best_distance is not None
    ]
    ranked = sorted(candidates, key=lambda result: result.raw_best_distance or 0.0, reverse=True)[:5]

    lines = [
        "### Highest-Distance Failures",
        "",
        "| Query | Category | Difficulty | Retrieval Status | Raw Best Distance |",
        "|---|---|---|---|---:|",
    ]
    if not ranked:
        lines.append("| No unsupported distance failures found. |  |  |  |  |")
        return lines

    for result in ranked:
        lines.append(
            f"| {_escape_table_cell(result.query)} | `{result.category}` | "
            f"`{result.difficulty}` | `{result.retrieval_status}` | "
            f"{_format_optional_float(result.raw_best_distance)} |"
        )

    return lines


def _format_key_findings_section(
    baseline_metrics: "ThresholdEvaluationMetrics",
    sweep_metrics: List["ThresholdEvaluationMetrics"],
) -> List[str]:
    """Return the key findings section lines."""
    findings = generate_key_findings(baseline_metrics, sweep_metrics)
    lines: List[str] = [
        "## Key Findings",
        "",
    ]

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
    """Generate report observations from retrieval, grounding, and threshold metrics."""
    results = baseline_metrics.results
    findings: List[str] = []
    retrieval_metrics = _compute_retrieval_accuracy(results)
    grounding_metrics = _compute_grounding_metrics(results)
    refusal_metrics = _compute_refusal_metrics(results)

    if baseline_metrics.rejected_queries > 0:
        findings.append(
            "The default threshold appears too strict for some queries, causing rejected or weak retrievals."
        )

    if retrieval_metrics.total_source_aware_queries:
        findings.append(
            f"Expected-source retrieval accuracy is {_format_percent(retrieval_metrics.accuracy_ratio)} "
            f"across `{retrieval_metrics.total_source_aware_queries}` source-aware queries."
        )

    if grounding_metrics.keyword_aware_queries:
        findings.append(
            f"Average keyword grounding coverage is {_format_percent(grounding_metrics.average_keyword_coverage)} "
            f"across `{grounding_metrics.keyword_aware_queries}` keyword-aware queries."
        )

    if refusal_metrics.refusal_tests:
        findings.append(
            f"Hallucination-resistance refusal success is {_format_percent(refusal_metrics.success_ratio)} "
            f"across `{refusal_metrics.refusal_tests}` refusal tests."
        )

    best_supported = _best_supported_threshold(sweep_metrics)
    if best_supported and best_supported.threshold != baseline_metrics.threshold:
        findings.append(
            f"Threshold calibration improves supported-answer count at `{best_supported.threshold:.2f}` "
            f"compared with baseline `{baseline_metrics.threshold:.2f}`."
        )

    hard_failures = [
        result for result in results
        if result.difficulty == "hard" and _is_failure(result)
    ]
    all_failures = [result for result in results if _is_failure(result)]
    if all_failures and len(hard_failures) / len(all_failures) >= 0.5:
        findings.append("Retrieval and grounding failures are concentrated in hard queries.")

    refusal_hallucinations = [
        result for result in results
        if result.expected_behavior == "refusal" and not _is_correct_refusal(result)
    ]
    weak_refusal_hallucinations = [
        result for result in refusal_hallucinations
        if result.retrieval_status in {"WEAK", "REJECTED", "NO_RESULTS"}
    ]
    if refusal_hallucinations and len(weak_refusal_hallucinations) >= len(refusal_hallucinations) / 2:
        findings.append("Hallucination failures mostly occur under weak or rejected retrieval.")

    repository_success_rate = _success_rate(
        [result for result in results if result.category == "repository"]
    )
    ambiguous_success_rate = _success_rate(
        [result for result in results if result.category == "ambiguous"]
    )
    if repository_success_rate > ambiguous_success_rate:
        findings.append("Repository-specific queries outperform ambiguous queries.")

    return findings


def _compute_retrieval_accuracy(results: Sequence["EvaluationResult"]) -> RetrievalAccuracyMetrics:
    """Compute expected-source retrieval accuracy."""
    source_aware = [result for result in results if result.expected_source]
    missing = [result for result in source_aware if not _expected_source_retrieved(result)]
    return RetrievalAccuracyMetrics(
        total_source_aware_queries=len(source_aware),
        successful_source_retrievals=len(source_aware) - len(missing),
        missing_source_results=missing,
    )


def _compute_grounding_metrics(results: Sequence["EvaluationResult"]) -> GroundingMetrics:
    """Compute keyword coverage metrics across keyword-aware cases."""
    keyword_aware = [result for result in results if result.expected_keywords]
    matches = [_compute_keyword_match(result) for result in keyword_aware]
    coverage_values = [match.coverage_ratio for match in matches]
    fully_grounded = sum(1 for match in matches if match.total_keywords > 0 and match.coverage_ratio == 1.0)
    partially_grounded = sum(1 for match in matches if 0.0 < match.coverage_ratio < 1.0)
    weakly_grounded = sum(1 for match in matches if match.total_keywords > 0 and match.coverage_ratio == 0.0)

    return GroundingMetrics(
        keyword_aware_queries=len(keyword_aware),
        average_keyword_coverage=mean(coverage_values) if coverage_values else 0.0,
        fully_grounded_answers=fully_grounded,
        partially_grounded_answers=partially_grounded,
        weakly_grounded_answers=weakly_grounded,
    )


def _compute_refusal_metrics(results: Sequence["EvaluationResult"]) -> RefusalMetrics:
    """Compute refusal correctness metrics."""
    refusal_results = [result for result in results if result.expected_behavior == "refusal"]
    correct_refusals = sum(1 for result in refusal_results if _is_correct_refusal(result))
    return RefusalMetrics(
        refusal_tests=len(refusal_results),
        correct_refusals=correct_refusals,
        hallucinated_responses=len(refusal_results) - correct_refusals,
    )


def _compute_keyword_match(result: "EvaluationResult") -> KeywordMatch:
    """Compare generated answer text with expected keywords."""
    expected_keywords = result.expected_keywords
    if not expected_keywords:
        return KeywordMatch(
            matched_keywords=[],
            matched_count=0,
            total_keywords=0,
            coverage_ratio=0.0,
        )

    normalized_answer = result.answer.casefold()
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.casefold() in normalized_answer
    ]
    return KeywordMatch(
        matched_keywords=matched_keywords,
        matched_count=len(matched_keywords),
        total_keywords=len(expected_keywords),
        coverage_ratio=len(matched_keywords) / len(expected_keywords),
    )


def _expected_source_retrieved(result: "EvaluationResult") -> bool:
    """Return True if the expected source appears in retrieved source lines."""
    if not result.expected_source:
        return False

    expected_source = result.expected_source.casefold()
    return any(expected_source in source.casefold() for source in result.retrieved_sources)


def _is_correct_refusal(result: "EvaluationResult") -> bool:
    """Return True if a refusal test produced a non-hallucinated refusal."""
    if result.retrieval_status in {"REJECTED", "NO_RESULTS"}:
        return True

    return _is_refusal_answer(result.answer)


def _is_refusal_answer(answer: str) -> bool:
    """Detect common refusal phrasings used by the RAG prompts and API."""
    normalized = answer.casefold()
    refusal_markers = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "can't answer",
        "not in the context",
        "not found in the context",
        "available documents",
        "provided context",
        "insufficient information",
    ]
    return any(marker in normalized for marker in refusal_markers)


def _format_breakdown_row(label: str, results: Sequence["EvaluationResult"]) -> str:
    """Return one performance breakdown table row."""
    supported = sum(1 for result in results if result.verification_result)
    unsupported = len(results) - supported
    source_accuracy = _compute_retrieval_accuracy(results).accuracy_ratio
    keyword_coverage = _compute_grounding_metrics(results).average_keyword_coverage
    return (
        f"| `{label}` | {len(results)} | {supported} | {unsupported} | "
        f"{_format_percent(_success_rate(results))} | "
        f"{_format_percent(keyword_coverage)} | {_format_percent(source_accuracy)} |"
    )


def _success_rate(results: Sequence["EvaluationResult"]) -> float:
    """Compute behavior-aware success rate for report breakdowns."""
    if not results:
        return 0.0

    successful = sum(1 for result in results if _is_success(result))
    return successful / len(results)


def _is_success(result: "EvaluationResult") -> bool:
    """Return True when the result satisfies the expected evaluation behavior."""
    if result.expected_behavior == "refusal":
        return _is_correct_refusal(result)

    if result.expected_behavior == "weak_retrieval":
        return result.retrieval_status in {"WEAK", "REJECTED", "NO_RESULTS"}

    if result.expected_source and not _expected_source_retrieved(result):
        return False

    keyword_match = _compute_keyword_match(result)
    if keyword_match.total_keywords:
        return keyword_match.coverage_ratio >= 0.5 and result.verification_result

    return result.verification_result


def _is_failure(result: "EvaluationResult") -> bool:
    """Return True when a result is diagnostically unsuccessful."""
    return not _is_success(result)


def _average_best_distance(results: Sequence["EvaluationResult"]) -> float:
    """Return average raw best distance."""
    valid_distances = [
        result.raw_best_distance
        for result in results
        if result.raw_best_distance is not None
    ]
    return mean(valid_distances) if valid_distances else float("nan")


def _best_supported_threshold(
    sweep_metrics: Sequence["ThresholdEvaluationMetrics"],
) -> "ThresholdEvaluationMetrics | None":
    """Return threshold metrics with the strongest supported-answer count."""
    if not sweep_metrics:
        return None

    return max(
        sweep_metrics,
        key=lambda item: (item.supported_answers, -item.unsupported_answers, -item.rejected_queries),
    )


def _format_counter_summary(counter: Counter[str], limit: int = 3) -> str:
    """Format top counter values for a bullet sentence."""
    if not counter:
        return "none"

    return ", ".join(f"`{label}` ({count})" for label, count in counter.most_common(limit))


def _verification_label(result: "EvaluationResult") -> str:
    """Return human-readable verification label."""
    return "SUPPORTED" if result.verification_result else "UNSUPPORTED"


def _format_yes_no(value: bool) -> str:
    """Return YES/NO for markdown diagnostics."""
    return "YES" if value else "NO"


def _format_expected_source_hit(result: "EvaluationResult") -> str:
    """Return expected-source hit status for query diagnostics."""
    if not result.expected_source:
        return "N/A"

    return _format_yes_no(_expected_source_retrieved(result))


def _format_percent(value: float) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100:.1f}%"


def _format_float(value: float) -> str:
    """Format floats while handling NaN."""
    if value != value:
        return "N/A"
    return f"{value:.4f}"


def _format_optional_float(value: float | None) -> str:
    """Format optional floats for markdown tables."""
    if value is None:
        return "`N/A`"
    return f"`{value:.4f}`"


def _escape_table_cell(value: str) -> str:
    """Escape markdown table separators in free text."""
    return value.replace("|", "\\|").replace("\n", " ")
