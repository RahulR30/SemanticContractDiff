"""Chunking: split contract text into overlapping pieces for embed/retrieve.

This is the 'chunking' step in a RAG pipeline. We use LangChain's
RecursiveCharacterTextSplitter so clauses are not hard-cut mid-sentence
when possible.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ~500 chars ≈ a short legal clause; overlap keeps boundary sentences in both chunks.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def make_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting on paragraph/sentence boundaries before raw characters.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(
    text: str,
    *,
    version: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split raw PDF text into LangChain Documents with version metadata.

    Metadata fields:
      - version: "A" (original) or "B" (revised)
      - source:  file path / label
      - chunk_index: order within that version
    """
    splitter = make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(
        [text],
        metadatas=[{"version": version, "source": source}],
    )
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i
    return docs
