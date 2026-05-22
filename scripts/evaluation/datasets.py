"""Reusable evaluation query datasets."""

from typing import Dict, List

from scripts.evaluation.models import EvaluationCase


DEFAULT_DATASET_NAME = "default"

_DEFAULT_TEST_CASES = [
    # Factual queries should be answerable directly from available documents.
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
    # Edge cases exercise boundary behavior such as missing context or defaults.
    EvaluationCase(
        query="What should the system do if no relevant documents are available?",
        category="edge",
    ),
    # Ambiguous queries may require disambiguation or broader retrieval context.
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
    # Unrelated queries test refusal and hallucination resistance.
    EvaluationCase(
        query="What is the weather in Paris today?",
        category="unrelated",
    ),
    EvaluationCase(
        query="Describe a question that cannot be answered from the available documents.",
        category="unrelated",
    ),
]

EVALUATION_DATASETS: Dict[str, List[EvaluationCase]] = {
    DEFAULT_DATASET_NAME: _DEFAULT_TEST_CASES,
}


def build_test_cases(dataset_name: str = DEFAULT_DATASET_NAME) -> List[EvaluationCase]:
    """Return evaluation cases for a named dataset."""
    return list(EVALUATION_DATASETS[dataset_name])
