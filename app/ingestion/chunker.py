"""Text chunking utilities."""

from typing import List


class TextChunker:
    """Split text into chunks for embedding."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize chunker.

        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for period, question mark, or exclamation within last 100 chars
                last_period = chunk.rfind(".")
                last_question = chunk.rfind("?")
                last_exclamation = chunk.rfind("!")

                last_boundary = max(last_period, last_question, last_exclamation)
                if last_boundary > self.chunk_size * 0.8:  # At least 80% of chunk
                    end = start + last_boundary + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc))
        return all_chunks
