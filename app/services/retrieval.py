"""RAG retrieval service."""

from typing import List, Tuple
from app.ingestion.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.core.prompts import PromptTemplates


class RetrievalService:
    """Service for retrieving relevant documents and generating responses."""

    def __init__(
        self, embedding_service: EmbeddingService, vector_store: FAISSVectorStore
    ):
        """Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve_context(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            Tuple of (documents, distances)
        """
        query_embedding = self.embedding_service.embed_text(query)
        documents, distances = self.vector_store.search(query_embedding, k=k)
        return documents, distances

    def format_context(self, documents: List[str], max_length: int = 2000) -> str:
        """Format retrieved documents as context.

        Args:
            documents: List of retrieved documents
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
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
        return PromptTemplates.get_retrieval_prompt(context, query)
