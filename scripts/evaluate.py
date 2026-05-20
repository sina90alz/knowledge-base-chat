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
    retrieved_sources: List[str]
    retrieved_distances: List[float]
    answer: str
    verification_result: bool
    retrieval_status: str
    generation_status: str
    best_distance: Optional[float]
    raw_best_distance: Optional[float]
    retrieved_count: int


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

    retrieval_status = assess_retrieval_status(raw_distances, settings.SIMILARITY_THRESHOLD)
    generation_status = "SUPPORTED" if verification_result else "UNSUPPORTED"
    best_distance = min(retrieved_distances) if retrieved_distances else None
    raw_best_distance = min(raw_distances) if raw_distances else None
    retrieved_sources = format_sources(retrieved_metadata, retrieved_distances)

    return EvaluationResult(
        query=query,
        category=category,
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

    evaluation_results: List[EvaluationResult] = []

    for case in cases:
        result = evaluate_query(
            query=case.query,
            category=case.category,
            embedding_service=embedding_service,
            vector_store=vector_store,
            retrieval_service=retrieval_service,
            llm_service=llm_service,
            verification_service=verification_service,
        )
        evaluation_results.append(result)
        print_case_result(result)

    print_summary(evaluation_results)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Evaluation script failed: %s", exc)
        sys.exit(1)
