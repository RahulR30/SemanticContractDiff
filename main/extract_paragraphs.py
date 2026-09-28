"""Python file for ExtractParagraphs.py. Written by Prajwala Immareddy."""

from typing import List
import re
import unicodedata
import fitz

class ExtractParagraphs:
    """Python class to convert PDF input into a list of paragraphs."""

    def __init__(self, filepath: str) -> None:
        """Reads filepath to a PDF file and initializes it as a normalized string."""
        pdf = fitz.open(filepath)
        try:
            # PDF text has no reliable implicit header/footer marker. Preserve
            # every line and keep page boundaries from concatenating clauses.
            text = "\n\n".join(page.get_text() for page in pdf)
            self.text = unicodedata.normalize('NFKC', text.strip())
        finally:
            pdf.close()

    def text_to_paragraph(self) -> List[str]:
        """Returns a list of strings where each element is a paragraph."""
        paragraphs: List[str] = []

        # Regex to recognize newlines
        parts = re.split(r"\r?\n\s*\r?\n", self.text)

        # Split into paragraphs
        paragraphs = [p.strip() for p in parts if p.strip()]

        return paragraphs
