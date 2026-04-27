"""Document loading utilities."""

from pathlib import Path
from typing import List
from pypdf import PdfReader


class DocumentLoader:
    """Load documents from various sources."""

    @staticmethod
    def load_pdf(file_path: str | Path) -> str:
        """Load text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text from PDF
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text

    @staticmethod
    def load_txt(file_path: str | Path) -> str:
        """Load text from text file.

        Args:
            file_path: Path to text file

        Returns:
            File contents
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def load_directory(directory: str | Path, pattern: str = "*.pdf") -> List[str]:
        """Load all documents matching pattern in directory.

        Args:
            directory: Directory path
            pattern: File pattern (default: *.pdf)

        Returns:
            List of file paths
        """
        directory = Path(directory)
        return [str(f) for f in directory.glob(pattern)]
