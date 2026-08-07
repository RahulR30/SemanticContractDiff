import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main.chunking import chunk_text


class TestChunking(unittest.TestCase):
    def test_chunk_text_assigns_metadata(self):
        text = ("Indemnity. " * 40) + "\n\n" + ("Payment terms. " * 40)
        docs = chunk_text(text, version="A", source="demo.pdf")
        self.assertGreater(len(docs), 1)
        for i, doc in enumerate(docs):
            self.assertEqual(doc.metadata["version"], "A")
            self.assertEqual(doc.metadata["source"], "demo.pdf")
            self.assertEqual(doc.metadata["chunk_index"], i)
            self.assertTrue(doc.page_content.strip())

    def test_overlap_produces_multiple_chunks_on_long_text(self):
        text = "word " * 300
        docs = chunk_text(text, version="B", source="x.pdf", chunk_size=200, chunk_overlap=50)
        self.assertGreaterEqual(len(docs), 2)


if __name__ == "__main__":
    unittest.main()
