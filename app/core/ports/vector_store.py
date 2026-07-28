"""Port for document retrieval from an embedded knowledge base."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

EmbeddingVector = Sequence[float]
DocumentEmbeddings = Sequence[EmbeddingVector]
DocumentMetadata = Dict[str, Any]
RelevantDocuments = Tuple[List[str], List[float], List[DocumentMetadata]]


class VectorStore(ABC):
    """Interface for storing and retrieving embedded documents."""

    @abstractmethod
    def add_documents(
        self,
        document_chunks: List[str],
        embeddings: DocumentEmbeddings,
        metadata_list: List[DocumentMetadata] | None = None,
    ) -> None:
        """Store document chunks and their embeddings in the collection."""
        raise NotImplementedError

    @abstractmethod
    def find_relevant_documents(
        self,
        query_embedding: EmbeddingVector,
        limit: int = 5,
    ) -> RelevantDocuments:
        """Return documents relevant to the supplied query embedding."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics for the document collection."""
        raise NotImplementedError
