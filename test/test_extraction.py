"""File to test PDF extraction into paragraphs. Written by Prajwala Immareddy."""

from typing import Iterator, List
import unittest
from unittest.mock import MagicMock, patch

from main.extract_paragraphs import ExtractParagraphs

class MockPDF():
    """Minimal mock to simulate the PDF object used by the extractor."""

    def __init__(self, pages: List[MagicMock]) -> None:
        """Initialize PDF pages."""
        self._pages = pages

    def __iter__(self) -> Iterator[MagicMock]:
        """Iterate through pages."""
        return iter(self._pages)

    def close(self) -> None:
        """Close PDF access."""
        return None


class TestExtractParagraphs(unittest.TestCase):
    """Unit tests for ExtractParagraphs calling the actual constructor."""

    def test_constructor_normalizes(self) -> None:
        """Constructor should read pages, remove header/footer, and normalize ligatures."""
        page: MagicMock = MagicMock()
        page.get_text.return_value = "Header\nHello\uFB01 world\nFooter"

        with patch('main.extract_paragraphs.fitz.open', return_value=MockPDF([page])):
            extractor = ExtractParagraphs("fake.pdf")

        self.assertEqual(extractor.text, "Hellofi world")

    def test_text_to_paragraph_splits(self) -> None:
        """Ensure method splits on actual newlines."""
        page: MagicMock = MagicMock()
        page.get_text.return_value = "Header\npara1\n\npara2\nFooter"

        with patch('main.extract_paragraphs.fitz.open', return_value=MockPDF([page])):
            extractor = ExtractParagraphs("fake.pdf")

        paragraphs = extractor.text_to_paragraph()

        self.assertEqual(paragraphs, ["para1", "para2"])

    def test_literal_escape_not_split(self) -> None:
        """Verify that literal backslash escape sequences are not treated as newlines."""
        page: MagicMock = MagicMock()

        page.get_text.return_value = "Header\npara1\\n\\npara2\nFooter"

        with patch('main.extract_paragraphs.fitz.open', return_value=MockPDF([page])):
            extractor = ExtractParagraphs("fake.pdf")

        paragraphs = extractor.text_to_paragraph()

        self.assertEqual(paragraphs, ["para1\\n\\npara2"])
