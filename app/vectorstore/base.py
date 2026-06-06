"""Base protocol for vector store implementations."""

from typing import Protocol, List, Tuple, Dict, Any
import numpy as np


class VectorStore(Protocol):
    """Protocol for vector store implementations.
    
    This protocol defines the minimal interface required for vector stores
    to be used with RetrievalService. Implementations can add additional
    methods as needed.
    """

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Search for similar texts.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (texts, distances, metadata_list)
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        ...
