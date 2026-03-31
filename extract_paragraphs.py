"""Python file for ExtractParagraphs.py. Written by Prajwala Immareddy."""

from typing import List
import unicodedata
import fitz

class ExtractParagraphs:
    """Python class to convert PDF input into a list of paragraphs."""

    def __init__(self, filepath: str) -> None:
        """Reads filepath to a PDF file and initializes it as a normalized string."""
        pdf = fitz.open(filepath)
        self.text = ""

        for page in pdf:
            self.text += page.get_text()

        # Strip newlines
        self.text = self.text.strip()

        #Drop header/footer
        lines = self.text.split('\n')

        main_lines = lines[1:-1]

        self.text = '\n'.join(main_lines)

        # Split ligatures into two characters
        self.text = unicodedata.normalize('NFKC', self.text)

        # Close PDF to save memory
        pdf.close()

    def text_to_paragraph(self) -> List[str]:
        """Returns a list of strings where each element is a paragraph."""
        paragraphs: List[str] = []

        # Split into paragraphs
        paragraphs = self.text.split("\\n\\n")

        return paragraphs
