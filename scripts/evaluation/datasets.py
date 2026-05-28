"""Corpus-grounded evaluation query datasets.

The benchmark is split into reusable groups so evaluation runs can focus on
retrieval quality, semantic search, grounding, document understanding, and
hallucination resistance without changing the evaluator itself.

All grounded queries in this file are answerable from the PDF/TXT documents
loaded from data/raw and indexed through the PDF/TXT -> chunking -> embeddings
-> FAISS ingestion pipeline. Project source code is intentionally excluded.
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

SCIENTIFIC_QUERIES = "scientific_queries"
SEMANTIC_QUERIES = "semantic_queries"
SUMMARIZATION_QUERIES = "summarization_queries"
COMPARISON_QUERIES = "comparison_queries"
HALLUCINATION_QUERIES = "hallucination_queries"
AMBIGUOUS_QUERIES = "ambiguous_queries"

EFFICIENTNET_SOURCE = (
    "EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf"
)
MOBILENET_SOURCE = (
    "MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf"
)
MOBILENETV2_SOURCE = "MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf"


# Scientific queries are direct factual QA over the indexed papers. They test
# whether retrieval can land on the right document chunks for definitions,
# named methods, benchmark claims, and architecture components.
scientific_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="What is compound scaling in EfficientNet?",
        category="scientific",
        expected_keywords=[
            "compound scaling",
            "depth",
            "width",
            "resolution",
            "compound coefficient",
        ],
        expected_source=EFFICIENTNET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="Which building block is used in EfficientNet-B0?",
        category="scientific",
        expected_keywords=[
            "EfficientNet-B0",
            "MBConv",
            "mobile inverted bottleneck",
            "squeeze-and-excitation",
        ],
        expected_source=EFFICIENTNET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="How does EfficientNet-B7 compare with previous ConvNets on ImageNet?",
        category="scientific",
        expected_keywords=[
            "EfficientNet-B7",
            "84.3%",
            "top-1 accuracy",
            "ImageNet",
            "8.4x smaller",
            "6.1x faster",
        ],
        expected_source=EFFICIENTNET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="What are depthwise separable convolutions in MobileNets?",
        category="scientific",
        expected_keywords=[
            "depthwise separable convolutions",
            "depthwise convolution",
            "pointwise convolution",
            "1x1 convolution",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="Which MobileNet hyperparameters trade off latency and accuracy?",
        category="scientific",
        expected_keywords=[
            "width multiplier",
            "resolution multiplier",
            "latency",
            "accuracy",
            "resource",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What applications does the MobileNets paper evaluate besides ImageNet classification?",
        category="scientific",
        expected_keywords=[
            "object detection",
            "finegrain classification",
            "face attributes",
            "geo-localization",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="What are inverted residuals and linear bottlenecks in MobileNetV2?",
        category="scientific",
        expected_keywords=[
            "inverted residual",
            "linear bottleneck",
            "shortcut connections",
            "thin bottleneck layers",
        ],
        expected_source=MOBILENETV2_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Which tasks and benchmarks does MobileNetV2 evaluate?",
        category="scientific",
        expected_keywords=[
            "ImageNet",
            "COCO",
            "VOC",
            "classification",
            "object detection",
            "segmentation",
        ],
        expected_source=MOBILENETV2_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
]


# Semantic queries require paraphrase matching and conceptual synthesis rather
# than a single surface-form lookup. They evaluate embedding quality over
# scientific explanations and design motivations.
semantic_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="Why does EfficientNet scale depth, width, and input resolution together?",
        category="semantic",
        expected_keywords=[
            "balance",
            "depth",
            "width",
            "resolution",
            "receptive field",
            "fine-grained patterns",
        ],
        expected_source=EFFICIENTNET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="hard",
        notes="Requires retrieving the intuition behind compound scaling, not only the abstract.",
    ),
    EvaluationCase(
        query="How does MobileNet reduce computation compared with a standard convolution?",
        category="semantic",
        expected_keywords=[
            "factorize",
            "standard convolution",
            "depthwise convolution",
            "pointwise convolution",
            "model size",
            "computation",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Why does MobileNetV2 remove non-linearities from narrow bottleneck layers?",
        category="semantic",
        expected_keywords=[
            "non-linearities",
            "narrow layers",
            "representational power",
            "linear bottleneck",
        ],
        expected_source=MOBILENETV2_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="How do MobileNetV2 bottleneck layers separate capacity from transformation expressiveness?",
        category="semantic",
        expected_keywords=[
            "bottleneck layers",
            "input/output domains",
            "expressiveness",
            "transformation",
            "information flow",
        ],
        expected_source=MOBILENETV2_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
]


# Summarization queries test whether retrieved chunks support document-level
# synthesis. They should stay grounded in one named paper and avoid non-corpus
# project summaries.
summarization_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="Summarize the main contribution of the EfficientNet paper.",
        category="summarization",
        expected_keywords=[
            "model scaling",
            "compound scaling",
            "depth",
            "width",
            "resolution",
            "EfficientNets",
        ],
        expected_source=EFFICIENTNET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Summarize the MobileNets paper for someone choosing an efficient vision model.",
        category="summarization",
        expected_keywords=[
            "mobile",
            "embedded vision",
            "depthwise separable convolutions",
            "width multiplier",
            "resolution multiplier",
            "latency",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Summarize the key ideas introduced by MobileNetV2.",
        category="summarization",
        expected_keywords=[
            "MobileNetV2",
            "inverted residual",
            "linear bottleneck",
            "depthwise convolutions",
            "SSDLite",
            "Mobile DeepLabv3",
        ],
        expected_source=MOBILENETV2_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
]


# Comparison queries evaluate cross-document and intra-document retrieval. The
# expected source marks the dominant paper that should appear among retrieved
# chunks, while keywords represent concepts that should be grounded by the
# retrieved corpus.
comparison_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="Compare MobileNet's width and resolution multipliers with EfficientNet's compound scaling.",
        category="comparison",
        expected_keywords=[
            "width multiplier",
            "resolution multiplier",
            "compound scaling",
            "depth",
            "width",
            "resolution",
        ],
        expected_source=EFFICIENTNET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="hard",
        notes="Cross-paper comparison should retrieve both MobileNets and EfficientNet terminology.",
    ),
    EvaluationCase(
        query="Compare standard convolution with depthwise separable convolution as described in MobileNets.",
        category="comparison",
        expected_keywords=[
            "standard convolution",
            "depthwise separable convolution",
            "depthwise convolution",
            "pointwise convolution",
            "computation",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="medium",
    ),
    EvaluationCase(
        query="How does MobileNetV2 build on MobileNetV1 while improving mobile vision models?",
        category="comparison",
        expected_keywords=[
            "MobileNetV1",
            "MobileNetV2",
            "inverted residual",
            "linear bottleneck",
            "accuracy",
            "latency",
        ],
        expected_source=MOBILENETV2_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
    EvaluationCase(
        query="Compare the efficiency goals of MobileNets and EfficientNet.",
        category="comparison",
        expected_keywords=[
            "efficient",
            "latency",
            "accuracy",
            "parameters",
            "FLOPS",
            "mobile",
        ],
        expected_source=MOBILENET_SOURCE,
        expected_behavior="grounded_answer",
        difficulty="hard",
    ),
]


# Hallucination queries intentionally ask for information outside the indexed
# scientific corpus. Unlike grounded QA, success here means refusing, weak
# retrieval, or an unsupported answer instead of using external knowledge or
# guessing from unrelated paper chunks.
hallucination_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="What is the weather in Paris today?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="easy",
        notes="Temporal external fact; should not be answered from the paper corpus.",
    ),
    EvaluationCase(
        query="What is the current price of Bitcoin?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="easy",
    ),
    EvaluationCase(
        query="What medication should I take for chest pain?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents", "insufficient information"],
        expected_behavior="refusal",
        difficulty="hard",
        notes="High-stakes medical advice must not be generated from unrelated CNN papers.",
    ),
    EvaluationCase(
        query="Who was the first emperor of Rome?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="medium",
    ),
    EvaluationCase(
        query="What are the latest visa rules for Japan?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="medium",
    ),
    EvaluationCase(
        query="Summarize the contents of a file named clinical_trials_2026.xlsx.",
        category="hallucination",
        expected_keywords=["I don't know", "available documents"],
        expected_behavior="refusal",
        difficulty="medium",
        notes="Tests resistance to inventing unavailable private file contents.",
    ),
    EvaluationCase(
        query="Which company will have the highest stock price next week?",
        category="hallucination",
        expected_keywords=["I don't know", "available documents", "insufficient information"],
        expected_behavior="refusal",
        difficulty="hard",
    ),
]


# Ambiguous queries are corpus-related but underspecified. They evaluate whether
# retrieval produces weak/broad context and whether generation remains cautious
# when the question lacks a named paper, method, or comparison target.
ambiguous_queries: List[EvaluationCase] = [
    EvaluationCase(
        query="How does it work?",
        category="ambiguous",
        expected_behavior="weak_retrieval",
        difficulty="hard",
        notes="Pronoun-only query has no recoverable referent in a multi-paper corpus.",
    ),
    EvaluationCase(
        query="Explain the architecture.",
        category="ambiguous",
        expected_behavior="ambiguous",
        difficulty="medium",
        notes="Could refer to MobileNet, MobileNetV2, EfficientNet-B0, or the scaling method.",
    ),
    EvaluationCase(
        query="What are the advantages of this method?",
        category="ambiguous",
        expected_behavior="ambiguous",
        difficulty="medium",
        notes="The method is unspecified, so a cautious answer should identify ambiguity.",
    ),
    EvaluationCase(
        query="Compare the models.",
        category="ambiguous",
        expected_behavior="ambiguous",
        difficulty="medium",
        notes="The corpus contains multiple model families and comparison axes.",
    ),
    EvaluationCase(
        query="Summarize the paper.",
        category="ambiguous",
        expected_behavior="ambiguous",
        difficulty="hard",
        notes="There are three indexed papers; retrieval should not assume a single target document.",
    ),
    EvaluationCase(
        query="What does the table show?",
        category="ambiguous",
        expected_behavior="weak_retrieval",
        difficulty="hard",
        notes="Underspecified table reference should produce weak retrieval or a request for clarification.",
    ),
]


EVALUATION_DATASETS: Dict[str, List[EvaluationCase]] = {
    SCIENTIFIC_QUERIES: scientific_queries,
    SEMANTIC_QUERIES: semantic_queries,
    SUMMARIZATION_QUERIES: summarization_queries,
    COMPARISON_QUERIES: comparison_queries,
    HALLUCINATION_QUERIES: hallucination_queries,
    AMBIGUOUS_QUERIES: ambiguous_queries,
}

EVALUATION_DATASETS[DEFAULT_DATASET_NAME] = [
    *scientific_queries,
    *semantic_queries,
    *summarization_queries,
    *comparison_queries,
    *hallucination_queries,
    *ambiguous_queries,
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
