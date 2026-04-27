"""Document loading utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document with metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate document after initialization."""
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Document content must be non-empty string")

    def __len__(self) -> int:
        """Return document length in characters."""
        return len(self.content)


class DocumentLoader:
    """Load documents from various sources with metadata."""

    @staticmethod
    def load_pdf(file_path: str | Path) -> Document:
        """Load text from PDF file with metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            Document object with content and metadata

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If PDF cannot be read
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text_parts: List[str] = []
            page_count: int = 0

            reader = PdfReader(file_path)
            page_count = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(f"--- Page {page_num} ---\n{extracted}")

            if not text_parts:
                raise ValueError(f"No text extracted from {file_path}")

            text = "\n".join(text_parts)

            metadata: Dict[str, Any] = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "pdf",
                "page_count": page_count,
                "character_count": len(text),
            }

            logger.info(
                f"Loaded PDF: {file_path.name} ({page_count} pages, {len(text)} chars)"
            )

            return Document(content=text, metadata=metadata)

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

    @staticmethod
    def load_txt(file_path: str | Path) -> Document:
        """Load text from text file with metadata.

        Args:
            file_path: Path to text file

        Returns:
            Document object with content and metadata

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text:
                raise ValueError(f"File {file_path} is empty")

            metadata: Dict[str, Any] = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "txt",
                "character_count": len(text),
            }

            logger.info(f"Loaded TXT: {file_path.name} ({len(text)} chars)")

            return Document(content=text, metadata=metadata)

        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise

    @staticmethod
    def load_directory(
        directory: str | Path, pattern: str = "*.pdf"
    ) -> List[Document]:
        """Load all documents matching pattern in directory.

        Args:
            directory: Directory path
            pattern: File pattern (default: *.pdf)

        Returns:
            List of Document objects

        Raises:
            FileNotFoundError: If directory does not exist
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents: List[Document] = []
        file_paths = list(directory.glob(pattern))

        logger.info(f"Found {len(file_paths)} files matching pattern '{pattern}'")

        for file_path in file_paths:
            try:
                if file_path.suffix.lower() == ".pdf":
                    doc = DocumentLoader.load_pdf(file_path)
                elif file_path.suffix.lower() == ".txt":
                    doc = DocumentLoader.load_txt(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue

                documents.append(doc)

            except Exception as e:
                logger.error(f"Skipping {file_path}: {e}")
                continue

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
