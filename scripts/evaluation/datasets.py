"""Repository-aware evaluation query datasets.

The benchmark is split into reusable groups so evaluation runs can focus on
retrieval quality, grounded repository understanding, configuration lookup, or
hallucination resistance without changing the evaluator itself.
"""

from collections.abc import Collection, Sequence
from typing import Dict, List, Optional, TypeVar

from scripts.evaluation.models import (
    Difficulty,
    EvaluationCase,
    SUPPORTED_DIFFICULTIES,
)

FilterValue = TypeVar("FilterValue", bound=str)


DEFAULT_DATASET_NAME = "default"

REPOSITORY_QUERIES = "repository_queries"
HALLUCINATION_QUERIES = "hallucination_queries"
RETRIEVAL_QUERIES = "retrieval_queries"
CONFIGURATION_QUERIES = "configuration_queries"
EDGE_CASE_QUERIES = "edge_case_queries"


# Repository queries test whether retrieval can find concrete implementation
# details across source code, docs, API routes, and ingestion modules.
repository_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="What class handles answer grounding verification?",
        category="repository",
        expected_keywords=[
            "AnswerVerificationService",
            "verify_answer",
            "SUPPORTED",
            "UNSUPPORTED",
        ],
        expected_source="verification.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
        notes="Should retrieve the verification service, not generic RAG documentation.",
    ),
    EvaluationCase(
        query="Which service creates retrieval prompts and formats retrieved context?",
        category="repository",
        expected_keywords=[
            "RetrievalService",
            "format_context",
            "generate_prompt",
            "PromptTemplates",
        ],
        expected_source="retrieval.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Which FastAPI endpoint handles RAG queries?",
        category="repository",
        expected_keywords=["/api", "query_rag", "QueryRequest", "QueryResponse"],
        expected_source="routes.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What architecture pipeline is described for the knowledge base chat app?",
        category="repository",
        expected_keywords=[
            "Documents",
            "Chunking",
            "Embeddings",
            "FAISS",
            "Retrieval",
            "Verification",
        ],
        expected_source="README.md",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="What object represents loaded document content and metadata?",
        category="repository",
        expected_keywords=["Document", "content", "metadata", "__post_init__"],
        expected_source="loader.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="Which API response fields expose retrieved sources and retrieval status?",
        category="repository",
        expected_keywords=["QueryResponse", "sources", "retrieval_status"],
        expected_source="routes.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
]


# Retrieval queries target the code paths that rank, filter, prompt, report, and
# calibrate retrieval results. These cases should expose source-aware retrieval
# weaknesses more clearly than broad product questions.
retrieval_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="How does RetrievalService deduplicate chunks?",
        category="retrieval",
        expected_keywords=[
            "_filter_rank_and_deduplicate",
            "_get_dedupe_key",
            "seen_documents",
            "chunk_start_word",
            "chunk_end_word",
        ],
        expected_source="retrieval.py",
        expected_behavior="grounded_answer",
        difficulty="hard",
        notes="Requires connecting the ranking loop with the helper that builds stable dedupe keys.",
    ),
    EvaluationCase(
        query="What happens when retrieved chunk distances exceed the similarity threshold?",
        category="retrieval",
        expected_keywords=["SIMILARITY_THRESHOLD", "distance", "Skipping chunk"],
        expected_source="retrieval.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="How are raw retrieval diagnostics logged?",
        category="retrieval",
        expected_keywords=[
            "_log_retrieval_diagnostics",
            "best_distance",
            "rejected_by_threshold",
            "kept",
        ],
        expected_source="retrieval.py",
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="Which module generates Markdown evaluation reports?",
        category="retrieval",
        expected_keywords=["build_markdown_report", "RAG Evaluation Report"],
        expected_source="reporting.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What sections are included in the Markdown evaluation report?",
        category="retrieval",
        expected_keywords=[
            "System Configuration",
            "Evaluation Queries",
            "Threshold Calibration Results",
            "Evaluation Summary",
            "Key Findings",
        ],
        expected_source="reporting.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="How are similarity thresholds calibrated?",
        category="retrieval",
        expected_keywords=[
            "evaluate_threshold",
            "threshold",
            "SIMILARITY_THRESHOLD",
            "ThresholdEvaluationMetrics",
        ],
        expected_source="thresholding.py",
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="How does threshold evaluation restore the global similarity threshold after a sweep?",
        category="retrieval",
        expected_keywords=["old_threshold", "finally", "retrieval_module.SIMILARITY_THRESHOLD"],
        expected_source="thresholding.py",
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="Which prompt tells the model to answer using only retrieved context?",
        category="retrieval",
        expected_keywords=["RETRIEVAL_PROMPT", "STRICT RULES", "ONLY"],
        expected_source="prompts.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="How does prompt selection change for Ollama?",
        category="retrieval",
        expected_keywords=["OLLAMA_RETRIEVAL_PROMPT", "LLM_PROVIDER", "ollama"],
        expected_source="prompts.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="How many chunks are retrieved by default if I ask a question?",
        category="retrieval",
        expected_keywords=["MAX_CHUNKS", "5", "min(k, MAX_CHUNKS)"],
        expected_source="retrieval.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
]


# Configuration queries measure whether the system can retrieve environment
# variable defaults and operational settings from config.py and setup docs.
configuration_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="Which environment variable controls vector store persistence?",
        category="configuration",
        expected_keywords=["VECTOR_STORE_PATH", "data/vector_store"],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="Which setting controls the similarity threshold?",
        category="configuration",
        expected_keywords=["SIMILARITY_THRESHOLD", "1.2"],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="How can I switch between OpenAI and Ollama providers?",
        category="configuration",
        expected_keywords=["LLM_PROVIDER", "openai", "ollama"],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Which setting enables answer verification?",
        category="configuration",
        expected_keywords=["ENABLE_ANSWER_VERIFICATION", "True"],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What settings control chunk size and overlap?",
        category="configuration",
        expected_keywords=["CHUNK_SIZE", "CHUNK_OVERLAP", "500", "50"],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Where does the app look for raw source documents?",
        category="configuration",
        expected_keywords=["RAW_DATA_DIR", "data", "raw"],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Which settings choose the embedding model and generation model?",
        category="configuration",
        expected_keywords=[
            "EMBEDDING_MODEL",
            "OPENAI_MODEL",
            "OLLAMA_MODEL",
            "LLM_PROVIDER",
        ],
        expected_source="config.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
]


# Hallucination queries intentionally ask for information that should not be in
# the indexed repository corpus. Expected behavior is refusal instead of using
# external knowledge or guessing.
hallucination_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="What is the weather in Paris today?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="easy",
        notes="Temporal external fact; should not be answered from repository context.",
    ),
    EvaluationCase(
        query="Who won yesterday's Lakers game?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What is the current price of Bitcoin?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What are the latest visa rules for Japan?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Summarize the contents of a file named payroll_2026.xlsx.",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="medium",
        notes="Tests resistance to inventing unavailable private file contents.",
    ),
    EvaluationCase(
        query="What medical treatment should I use for chest pain?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="hard",
        notes="High-stakes external advice must not be generated from unrelated repo context.",
    ),
]


# Edge case queries exercise ambiguous, broad, and weak-retrieval behavior. Some
# should retrieve repository overview docs, while others should trigger cautious
# answers because the query underspecifies the target source or detail.
edge_case_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="Explain the system.",
        category="ambiguous",
        expected_keywords=["RAG", "retrieval", "documents"],
        expected_behavior="ambiguous",
        difficulty="medium",
        notes="Broad query should either ask for scope or answer from retrieved overview context.",
    ),
    EvaluationCase(
        query="How does it work?",
        category="ambiguous",
        expected_behavior="ambiguous",
        difficulty="hard",
        notes="Pronoun-only query tests whether retrieval returns weak or overly broad context.",
    ),
    EvaluationCase(
        query="Describe the architecture.",
        category="ambiguous",
        expected_keywords=["Documents", "FAISS", "Retrieval", "Verification"],
        expected_source="README.md",
        expected_behavior="ambiguous",
        difficulty="medium",
    ),
    EvaluationCase(
        query="What should happen if the vector store is empty?",
        category="edge",
        expected_keywords=["Vector store is empty", "return [], [], []"],
        expected_source="faiss_store.py",
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="What should the API return when retrieval is rejected?",
        category="edge",
        expected_keywords=["REJECTED", "I don't know based on the available documents"],
        expected_source="routes.py",
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="What happens if an evaluation query has insufficient retrieval context?",
        category="edge",
        expected_keywords=["Skipping generation", "REJECTED", "UNSUPPORTED"],
        expected_source="evaluate.py",
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="Compare every threshold setting and choose the best production value.",
        category="edge",
        expected_keywords=["Threshold Calibration Results", "Key Findings"],
        expected_source="reporting.py",
        expected_behavior="weak_retrieval",
        difficulty="hard",
        notes="The report can describe calibration output, but selecting production policy is underdetermined.",
    ),
]


EVALUATION_DATASETS: Dict[str, List[EvaluationCase]] = {
    REPOSITORY_QUERIES: repository_queries,
    HALLUCINATION_QUERIES: hallucination_queries,
    RETRIEVAL_QUERIES: retrieval_queries,
    CONFIGURATION_QUERIES: configuration_queries,
    EDGE_CASE_QUERIES: edge_case_queries,
}

EVALUATION_DATASETS[DEFAULT_DATASET_NAME] = [
    *repository_queries,
    *retrieval_queries,
    *configuration_queries,
    *hallucination_queries,
    *edge_case_queries,
]


def build_test_cases(
    dataset_name: str | Sequence[str] | None = DEFAULT_DATASET_NAME,
    *,
    dataset_groups: str | Sequence[str] | None = None,
    category: Optional[str] = None,
    categories: Optional[Collection[str]] = None,
    difficulty: Optional[Difficulty] = None,
    difficulties: Optional[Collection[Difficulty]] = None,
) -> List[EvaluationCase]:
    """Return evaluation cases for one or more named dataset groups.

    Args:
        dataset_name: Backward-compatible dataset selector. Accepts a single
            group name or a sequence of group names.
        dataset_groups: Explicit group selector for combining reusable datasets.
            Accepts one group or multiple groups. When provided, this takes
            precedence over dataset_name.
        category: Optional single category filter.
        categories: Optional set of category filters.
        difficulty: Optional single difficulty filter.
        difficulties: Optional set of difficulty filters.

    Returns:
        A new list of evaluation cases matching the requested groups and filters.
    """
    selected_dataset_names = _normalize_dataset_names(dataset_name, dataset_groups)
    selected_categories = _merge_filters(category, categories)
    selected_difficulties = _merge_filters(difficulty, difficulties)

    if selected_difficulties:
        unsupported = selected_difficulties.difference(SUPPORTED_DIFFICULTIES)
        if unsupported:
            raise ValueError(
                f"Unsupported difficulty filter(s): {', '.join(sorted(unsupported))}"
            )

    cases: List[EvaluationCase] = []
    for name in selected_dataset_names:
        try:
            dataset_cases = EVALUATION_DATASETS[name]
        except KeyError as exc:
            available = ", ".join(sorted(EVALUATION_DATASETS))
            raise KeyError(f"Unknown evaluation dataset '{name}'. Available: {available}") from exc

        cases.extend(dataset_cases)

    return [
        case
        for case in cases
        if (not selected_categories or case.category in selected_categories)
        and (not selected_difficulties or case.difficulty in selected_difficulties)
    ]


def _normalize_dataset_names(
    dataset_name: str | Sequence[str] | None,
    dataset_groups: str | Sequence[str] | None,
) -> List[str]:
    """Normalize backward-compatible and explicit dataset selectors."""
    selected = dataset_groups if dataset_groups is not None else dataset_name

    if selected is None:
        return [DEFAULT_DATASET_NAME]

    if isinstance(selected, str):
        return [selected]

    return list(selected)


def _merge_filters(
    single_value: Optional[FilterValue],
    multiple_values: Optional[Collection[FilterValue]],
) -> set[FilterValue]:
    """Merge singular and plural filter arguments into a set."""
    values = set(multiple_values or [])
    if single_value is not None:
        values.add(single_value)

    return values
