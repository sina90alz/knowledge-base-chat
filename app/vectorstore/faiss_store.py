"""FAISS vector store implementation."""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for similarity search with metadata support."""

    def __init__(self, dimension: int, store_path: str | Path = "data/vector_store"):
        """Initialize FAISS vector store.

        Args:
            dimension: Dimension of embedding vectors
            store_path: Path to store index and metadata

        Raises:
            ValueError: If dimension is invalid
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")

        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.store_path / "faiss.index"
        self.metadata_path = self.store_path / "metadata.pkl"

        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.vector_count = 0

        # Load existing index if available
        if self.index_path.exists():
            self._load_index()
            logger.info(
                f"Loaded existing FAISS index: {self.vector_count} vectors from {self.index_path}"
            )
        else:
            logger.info(f"Created new FAISS index with dimension {dimension}")

    def add_texts(
        self, texts: List[str], embeddings: np.ndarray, metadata_list: List[Dict[str, Any]] | None = None
    ) -> None:
        """Add texts and their embeddings to the store.

        Args:
            texts: List of text strings
            embeddings: Matrix of embedding vectors (num_texts x dimension)
            metadata_list: Optional list of metadata dicts for each text

        Raises:
            ValueError: If inputs are invalid or mismatched
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of embeddings ({len(embeddings)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match store dimension {self.dimension}"
            )

        if metadata_list is None:
            metadata_list = [{} for _ in texts]

        if len(metadata_list) != len(texts):
            raise ValueError(
                f"Number of metadata items ({len(metadata_list)}) must match number of texts ({len(texts)})"
            )

        try:
            # Convert embeddings to float32
            embeddings_float32 = embeddings.astype(np.float32)

            # Add to FAISS index
            self.index.add(embeddings_float32)

            # Store metadata with text
            for text, metadata in zip(texts, metadata_list):
                self.metadata.append(
                    {
                        "text": text,
                        **metadata,
                    }
                )

            self.vector_count = self.index.ntotal

            # Persist to disk
            self._save_index()

            logger.info(f"Added {len(texts)} vectors to store. Total vectors: {self.vector_count}")

        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Search for similar texts.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Tuple of (texts, distances, metadata_list)

        Raises:
            ValueError: If vector store is empty or invalid input
        """
        if self.vector_count == 0:
            logger.warning("Vector store is empty")
            return [], [], []

        if len(query_embedding.shape) != 1:
            raise ValueError("Query embedding must be 1-dimensional")

        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} does not match store dimension {self.dimension}"
            )

        try:
            query_float32 = query_embedding.astype(np.float32).reshape(1, -1)
            k = min(k, self.vector_count)

            distances, indices = self.index.search(query_float32, k)

            results: List[str] = []
            result_distances: List[float] = []
            result_metadata: List[Dict[str, Any]] = []

            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.metadata):
                    metadata = self.metadata[int(idx)]
                    results.append(metadata.get("text", ""))
                    result_distances.append(float(distance))
                    result_metadata.append({k: v for k, v in metadata.items() if k != "text"})

            logger.debug(f"Search returned {len(results)} results")
            return results, result_distances, result_metadata

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.debug(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.vector_count = self.index.ntotal

            if self.metadata_path.exists():
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)

            logger.debug(f"Loaded index with {self.vector_count} vectors")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def clear(self) -> None:
        """Clear the vector store and remove persisted files."""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            self.vector_count = 0

            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()

            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with store statistics
        """
        return {
            "total_vectors": self.vector_count,
            "embedding_dimension": self.dimension,
            "store_path": str(self.store_path),
            "index_file_exists": self.index_path.exists(),
        }

    def __len__(self) -> int:
        """Return number of vectors in store."""
        return self.vector_count
