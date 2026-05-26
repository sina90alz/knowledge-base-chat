"""Evaluation domain models."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


Difficulty = Literal["easy", "medium", "hard"]
ExpectedBehavior = Literal[
    "grounded_answer",
    "refusal",
    "ambiguous",
    "weak_retrieval",
]

SUPPORTED_DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
SUPPORTED_EXPECTED_BEHAVIORS: tuple[ExpectedBehavior, ...] = (
    "grounded_answer",
    "refusal",
    "ambiguous",
    "weak_retrieval",
)


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    category: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_source: Optional[str] = None
    expected_behavior: Optional[ExpectedBehavior] = None
    difficulty: Difficulty = "medium"
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.difficulty not in SUPPORTED_DIFFICULTIES:
            raise ValueError(
                f"Unsupported difficulty '{self.difficulty}'. "
                f"Expected one of: {', '.join(SUPPORTED_DIFFICULTIES)}"
            )

        if (
            self.expected_behavior is not None
            and self.expected_behavior not in SUPPORTED_EXPECTED_BEHAVIORS
        ):
            raise ValueError(
                f"Unsupported expected_behavior '{self.expected_behavior}'. "
                f"Expected one of: {', '.join(SUPPORTED_EXPECTED_BEHAVIORS)}"
            )


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
    expected_keywords: list[str] = field(default_factory=list)
    expected_source: Optional[str] = None
    expected_behavior: Optional[ExpectedBehavior] = None
    difficulty: Difficulty = "medium"
    notes: Optional[str] = None


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
