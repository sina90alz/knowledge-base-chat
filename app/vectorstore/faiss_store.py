"""FAISS vector store implementation."""

import os
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss


class FAISSVectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, dimension: int, store_path: str | Path = "data/vector_store"):
        """Initialize FAISS vector store.

        Args:
            dimension: Dimension of embedding vectors
            store_path: Path to store index and metadata
        """
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.store_path / "faiss.index"
        self.metadata_path = self.store_path / "metadata.pkl"

        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

        # Load existing index if available
        if self.index_path.exists():
            self._load_index()

    def add_texts(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Add texts and their embeddings to the store.

        Args:
            texts: List of text strings
            embeddings: Matrix of embedding vectors
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        # Add to FAISS index
        embeddings_float32 = embeddings.astype(np.float32)
        self.index.add(embeddings_float32)

        # Store metadata
        self.metadata.extend(texts)

        # Persist to disk
        self._save_index()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        """Search for similar texts.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Tuple of (texts, distances)
        """
        if len(self.metadata) == 0:
            return [], []

        query_float32 = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_float32, min(k, len(self.metadata)))

        results = []
        result_distances = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
                result_distances.append(float(distance))

        return results, result_distances

    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

        if self.metadata_path.exists():
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def clear(self) -> None:
        """Clear the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save_index()
