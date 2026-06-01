import tempfile
import unittest
from pathlib import Path

import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader
from app.vectorstore.faiss_store import FAISSVectorStore


def create_text_pdf(path: Path, page_texts: list[str]) -> None:
    writer = PdfWriter()

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): font,
                    }
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("utf-8"))
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream

    with path.open("wb") as file:
        writer.write(file)


class PageMetadataTests(unittest.TestCase):
    def test_pdf_ingestion_creates_page_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            create_text_pdf(pdf_path, ["first page content", "second page content"])

            documents = DocumentLoader.load_directory(temp_dir, pattern="*.pdf")

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["page"], 1)
        self.assertEqual(documents[1].metadata["page"], 2)
        self.assertEqual(documents[0].metadata["filename"], "sample.pdf")

    def test_chunking_preserves_page_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            create_text_pdf(pdf_path, ["alpha beta gamma", "delta epsilon zeta"])
            documents = DocumentLoader.load_pdf_pages(pdf_path)

            chunks = TextChunker(chunk_size=2, overlap=1).chunk_documents(documents)

        pages = [chunk.metadata["page"] for chunk in chunks]
        self.assertIn(1, pages)
        self.assertIn(2, pages)
        self.assertTrue(all("page" in chunk.metadata for chunk in chunks))

    def test_faiss_persistence_and_retrieval_keep_page_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FAISSVectorStore(dimension=2, store_path=temp_dir)
            store.add_texts(
                texts=["first page chunk", "second page chunk"],
                embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                metadata_list=[
                    {"filename": "sample.pdf", "page": 1},
                    {"filename": "sample.pdf", "page": 2},
                ],
            )

            reloaded_store = FAISSVectorStore(dimension=2, store_path=temp_dir)
            _, _, metadata = reloaded_store.search(
                np.array([0.0, 1.0], dtype=np.float32),
                k=1,
            )

        self.assertEqual(metadata[0]["filename"], "sample.pdf")
        self.assertEqual(metadata[0]["page"], 2)
        self.assertGreater(metadata[0]["page"], 0)
        self.assertNotIn("page_number", metadata[0])


if __name__ == "__main__":
    unittest.main()
