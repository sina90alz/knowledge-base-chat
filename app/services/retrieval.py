"""RAG retrieval service."""

import logging
from typing import List, Tuple, Dict, Any, Callable
from app.ingestion.embedder import EmbeddingService
from app.vectorstore.base import VectorStore
from app.core.prompts import PromptTemplates
from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant documents and generating responses."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        similarity_threshold: float | None = None,
        max_chunks: int = 5,
        prompt_formatter: Callable[[str, str], str] | None = None,
    ) -> None:
        """Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            similarity_threshold: Maximum distance for similarity filtering.
                If None, uses value from settings.
            max_chunks: Maximum number of chunks to return after filtering
            prompt_formatter: Optional function to format prompts. Should accept
                (context, query) and return formatted prompt string.
                If None, uses PromptTemplates.get_retrieval_prompt.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        self.max_chunks = max_chunks
        self._prompt_formatter = prompt_formatter or PromptTemplates.get_retrieval_prompt
        logger.info("RetrievalService initialized")

    def retrieve_context(
        self, query: str, k: int = 5
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            Tuple of (documents, distances, metadata)

        Raises:
            ValueError: If query is empty or vector store is empty
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        try:
            logger.info(
                "Retrieving up to %s documents for query: '%s...'",
                min(k, self.max_chunks),
                query[:50],
            )

            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)

            # Search vector store
            search_k = max(k, self.max_chunks)
            documents, distances, metadata = self.vector_store.search(
                query_embedding,
                k=search_k,
            )

            if not documents:
                logger.warning("No documents found for query")
                return [], [], []

            filtered_documents, filtered_distances, filtered_metadata = (
                self._filter_rank_and_deduplicate(
                    documents=documents,
                    distances=distances,
                    metadata=metadata,
                    max_chunks=min(k, self.max_chunks),
                )
            )

            self._log_retrieval_diagnostics(
                raw_distances=distances,
                kept_count=len(filtered_documents),
                max_chunks=min(k, self.max_chunks),
            )

            logger.info(
                "Retrieved %s documents, kept %s after filtering",
                len(documents),
                len(filtered_documents),
            )
            return filtered_documents, filtered_distances, filtered_metadata

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def _filter_rank_and_deduplicate(
        self,
        documents: List[str],
        distances: List[float],
        metadata: List[Dict[str, Any]],
        max_chunks: int,
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Apply distance filtering, ranking, deduplication, and result limiting."""
        ranked_results = sorted(
            zip(documents, distances, metadata),
            key=lambda result: result[1],
        )

        filtered_documents: List[str] = []
        filtered_distances: List[float] = []
        filtered_metadata: List[Dict[str, Any]] = []
        seen_documents: set[str] = set()

        for document, distance, document_metadata in ranked_results:
            if distance > self.similarity_threshold:
                logger.debug(
                    "Skipping chunk with distance %.4f above threshold %.4f",
                    distance,
                    self.similarity_threshold,
                )
                continue

            dedupe_key = self._get_dedupe_key(document, document_metadata)
            if dedupe_key in seen_documents:
                logger.debug("Skipping duplicate chunk: %s", dedupe_key)
                continue

            seen_documents.add(dedupe_key)
            filtered_documents.append(document)
            filtered_distances.append(distance)
            filtered_metadata.append(document_metadata)

            if len(filtered_documents) >= max_chunks:
                break

        return filtered_documents, filtered_distances, filtered_metadata

    def _log_retrieval_diagnostics(
        self,
        raw_distances: List[float],
        kept_count: int,
        max_chunks: int,
    ) -> None:
        """Log retrieval diagnostics for the current similarity threshold."""
        if not raw_distances:
            logger.info("Retrieval diagnostics: no raw distances returned")
            return

        rejected_by_threshold = sum(1 for distance in raw_distances if distance > self.similarity_threshold)
        best_distance = min(raw_distances)
        raw_summary = ", ".join(f"{distance:.4f}" for distance in raw_distances)

        logger.info(
            "Raw retrieval distances: %s",
            raw_summary if len(raw_distances) <= 6 else f"{raw_summary[:180]}...",
        )
        logger.info(
            "Retrieval diagnostics: best_distance=%.4f, threshold=%.4f, rejected_by_threshold=%d, kept=%d/%d",
            best_distance,
            self.similarity_threshold,
            rejected_by_threshold,
            kept_count,
            max_chunks,
        )

    def get_retrieval_quality(
        self,
        raw_distances: List[float],
        filtered_count: int,
    ) -> str:
        """Assess retrieval quality based on raw distances and filtered result count."""
        if filtered_count == 0:
            return "REJECTED"

        if not raw_distances:
            return "REJECTED"

        best_distance = min(raw_distances)
        if best_distance > self.similarity_threshold:
            return "WEAK"

        return "GOOD"

    @staticmethod
    def _get_dedupe_key(document: str, metadata: Dict[str, Any]) -> str:
        """Build a stable key to avoid repeated chunks in retrieval output."""
        source = metadata.get("source") or metadata.get("filename", "")
        chunk_start = metadata.get("chunk_start_word")
        chunk_end = metadata.get("chunk_end_word")

        if source and chunk_start is not None and chunk_end is not None:
            return f"{source}:{chunk_start}:{chunk_end}"

        return " ".join(document.split()).lower()

    def format_context(
        self,
        documents: List[str],
        metadata: List[Dict[str, Any]] | None = None,
        max_length: int = 2000,
    ) -> str:
        """Format retrieved documents as context.

        Args:
            documents: List of retrieved documents
            metadata: Metadata for each retrieved document
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        if metadata is None:
            metadata = [{} for _ in documents]

        context_parts: List[str] = []
        for index, document in enumerate(documents, 1):
            document_metadata = metadata[index - 1] if index <= len(metadata) else {}
            source = document_metadata.get("filename") or document_metadata.get("source", "Unknown")
            page = document_metadata.get("page", "Unknown")

            context_parts.append(
                "\n".join(
                    [
                        f"[Document {index}]",
                        f"Source: {source}",
                        f"Page: {page}",
                        f"Content: {document}",
                    ]
                )
            )

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > max_length:
            context = context[:max_length] + "..."

        return context

    def generate_prompt(self, query: str, context: str) -> str:
        """Generate prompt with context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        if not context:
            logger.warning("Generating prompt with empty context")

        return self._prompt_formatter(context, query)

    def get_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with store statistics
        """
        return self.vector_store.get_stats()
