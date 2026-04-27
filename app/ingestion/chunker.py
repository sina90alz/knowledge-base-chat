"""Text chunking utilities."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return chunk length in characters."""
        return len(self.content)

    def word_count(self) -> int:
        """Return chunk length in words."""
        return len(self.content.split())


class TextChunker:
    """Split text into word-based chunks with metadata preservation."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize chunker.

        Args:
            chunk_size: Size of each chunk in words
            overlap: Overlap between chunks in words
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(
            f"TextChunker initialized: chunk_size={chunk_size} words, overlap={overlap} words"
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Chunk]:
        """Split text into overlapping chunks by word count.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects with preserved metadata
        """
        if not text or not isinstance(text, str):
            return []

        if metadata is None:
            metadata = {}

        # Split text into words while preserving structure
        words = text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Try to break at sentence boundary if not at end
            if end < len(words):
                chunk_text = self._break_at_sentence(chunk_text)

            if chunk_text.strip():
                chunk_metadata = {
                    **metadata,
                    "chunk_start_word": start,
                    "chunk_end_word": end,
                    "chunk_word_count": len(chunk_text.split()),
                }

                chunks.append(Chunk(content=chunk_text.strip(), metadata=chunk_metadata))

            start = end - self.overlap

        logger.debug(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document while preserving metadata.

        Args:
            document: Document object to chunk

        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking document: {document.metadata.get('filename', 'unknown')}")

        chunks = self.chunk_text(document.content, metadata=document.metadata.copy())

        logger.info(
            f"Document {document.metadata.get('filename')} chunked into {len(chunks)} chunks"
        )

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of Document objects

        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks: List[Chunk] = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks

    @staticmethod
    def _break_at_sentence(text: str) -> str:
        """Try to break text at sentence boundary.

        Args:
            text: Text to break

        Returns:
            Text broken at sentence boundary, or original if no boundary found
        """
        # Look for sentence endings: . ! ? followed by space
        sentence_pattern = r"([.!?])\s+"
        matches = list(re.finditer(sentence_pattern, text))

        if matches:
            # Use the last sentence boundary
            last_match = matches[-1]
            return text[: last_match.end() - 1]  # Exclude the space

        return text
