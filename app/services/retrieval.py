"""RAG retrieval service."""

import logging
from typing import List, Tuple, Dict, Any
from app.ingestion.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.core.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant documents and generating responses."""

    def __init__(
        self, embedding_service: EmbeddingService, vector_store: FAISSVectorStore
    ) -> None:
        """Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
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
            logger.info(f"Retrieving {k} documents for query: '{query[:50]}...'")

            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)

            # Search vector store
            documents, distances, metadata = self.vector_store.search(query_embedding, k=k)

            if not documents:
                logger.warning("No documents found for query")
                return [], [], []

            logger.info(f"Retrieved {len(documents)} documents")
            return documents, distances, metadata

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def format_context(
        self, documents: List[str], max_length: int = 2000
    ) -> str:
        """Format retrieved documents as context.

        Args:
            documents: List of retrieved documents
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context = "\n---\n".join(documents)
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

        return PromptTemplates.get_retrieval_prompt(context, query)

    def get_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with store statistics
        """
        return self.vector_store.get_stats()
