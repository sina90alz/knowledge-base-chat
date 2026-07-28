"""ChromaDB vector store implementation."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.core.ports.vector_store import (
    DocumentEmbeddings,
    DocumentMetadata,
    EmbeddingVector,
    RelevantDocuments,
    VectorStore,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB vector store for precomputed embeddings."""

    _METADATA_FIELD = "_metadata_json"

    def __init__(
        self,
        dimension: int,
        store_path: str | Path = "data/vector_store",
        collection_name: str = "documents",
    ):
        """Initialize a persistent ChromaDB vector store.

        Args:
            dimension: Dimension of embedding vectors.
            store_path: Directory where ChromaDB persists collection data.
            collection_name: ChromaDB collection name.

        Raises:
            ValueError: If dimension is invalid.
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")

        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(self.store_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"},
            embedding_function=None,
        )

        logger.info(
            "Loaded ChromaDB collection '%s' with %s vectors from %s",
            self.collection_name,
            self.vector_count,
            self.store_path,
        )

    @property
    def vector_count(self) -> int:
        """Return the number of vectors in the collection."""
        return self.collection.count()

    def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Add texts and their precomputed embeddings to the store."""
        embeddings_array = np.asarray(embeddings)

        if len(texts) != len(embeddings_array):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of embeddings ({len(embeddings_array)})"
            )

        if embeddings_array.ndim != 2:
            raise ValueError("Embeddings must be a 2-dimensional matrix")

        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings_array.shape[1]} does not match store dimension {self.dimension}"
            )

        if metadata_list is None:
            metadata_list = [{} for _ in texts]

        if len(metadata_list) != len(texts):
            raise ValueError(
                f"Number of metadata items ({len(metadata_list)}) must match number of texts ({len(texts)})"
            )

        if not texts:
            return

        try:
            ids = [uuid4().hex for _ in texts]
            metadatas = [
                self._to_chroma_metadata(self._normalize_metadata(metadata))
                for metadata in metadata_list
            ]
            logger.info(
                "Metadata before Chroma insert: %s",
                self._normalize_metadata(metadata_list[0]) if metadata_list else {},
            )
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings_array.astype(np.float32).tolist(),
                metadatas=metadatas,
            )
            logger.info(
                "Added %s vectors to ChromaDB store. Total vectors: %s",
                len(texts),
                self.vector_count,
            )
        except Exception as exc:
            logger.error("Error adding texts to ChromaDB vector store: %s", exc)
            raise

    def add_documents(
        self,
        document_chunks: List[str],
        embeddings: DocumentEmbeddings,
        metadata_list: List[DocumentMetadata] | None = None,
    ) -> None:
        """Store document chunks and embeddings through the vector store port."""
        self.add_texts(document_chunks, np.asarray(embeddings), metadata_list)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Search for similar texts using a precomputed query embedding."""
        if self.vector_count == 0:
            logger.warning("Vector store is empty")
            return [], [], []

        query_array = np.asarray(query_embedding)
        if query_array.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")

        if query_array.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query_array.shape[0]} does not match store dimension {self.dimension}"
            )

        try:
            result = self.collection.query(
                query_embeddings=[query_array.astype(np.float32).tolist()],
                n_results=min(k, self.vector_count),
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents") or [[]]
            distances = result.get("distances") or [[]]
            metadatas = result.get("metadatas") or [[]]

            result_documents = [document or "" for document in documents[0]]
            result_distances = [float(distance) for distance in distances[0]]
            result_metadata = [
                self._from_chroma_metadata(metadata or {})
                for metadata in metadatas[0]
            ]

            if result_metadata:
                logger.info("Retrieved metadata: %s", result_metadata[0])
            logger.debug("ChromaDB search returned %s results", len(result_documents))
            return result_documents, result_distances, result_metadata
        except Exception as exc:
            logger.error("Error searching ChromaDB vector store: %s", exc)
            raise

    def find_relevant_documents(
        self,
        query_embedding: EmbeddingVector,
        limit: int = 5,
    ) -> RelevantDocuments:
        """Find documents relevant to a query embedding."""
        return self.search(np.asarray(query_embedding), k=limit)

    def clear(self) -> None:
        """Clear the ChromaDB collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            logger.debug("ChromaDB collection '%s' did not exist", self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"},
            embedding_function=None,
        )
        logger.info("ChromaDB vector store cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": self.vector_count,
            "embedding_dimension": self.dimension,
            "store_path": str(self.store_path),
            "index_file_exists": self.store_path.exists()
            and any(self.store_path.iterdir()),
            "collection_name": self.collection_name,
        }

    def __len__(self) -> int:
        """Return number of vectors in store."""
        return self.vector_count

    @classmethod
    def _to_chroma_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Encode caller metadata into a Chroma-compatible primitive dict."""
        return {
            cls._METADATA_FIELD: json.dumps(
                metadata,
                default=str,
                ensure_ascii=True,
                sort_keys=True,
            )
        }

    @classmethod
    def _from_chroma_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Decode caller metadata from Chroma storage."""
        raw_metadata = metadata.get(cls._METADATA_FIELD, "{}")
        try:
            decoded = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            decoded = {}

        return cls._normalize_metadata(decoded)

    @classmethod
    def _normalize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Return metadata standardized on metadata["page"]."""
        normalized = dict(metadata)

        if "page" not in normalized and "page_number" in normalized:
            normalized["page"] = normalized["page_number"]

        if "page" not in normalized and isinstance(normalized.get("text"), str):
            match = re.search(r"---\s*Page\s+(\d+)\s*---", normalized["text"])
            if match:
                normalized["page"] = int(match.group(1))

        normalized.pop("page_number", None)
        normalized.pop("text", None)
        return normalized
