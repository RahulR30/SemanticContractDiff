"""Vector DB layer: embed chunks and store/search them in Chroma.

Embeddings turn each chunk into a dense vector. Chroma indexes those vectors
so we can ask: 'which original chunks are most similar to this revised chunk?'
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Same family as the classic score_map MiniLM — keeps behavior comparable.
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load so importing the module does not download weights immediately."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def build_chroma_store(
    documents: List[Document],
    *,
    collection_name: str = "contract_chunks",
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Create a Chroma collection from Documents (cosine distance space).

    If persist_directory is set, the index is written to disk (real vector DB
    persistence). If None, Chroma runs ephemerally in-process for the session.
    """
    kwargs = {
        "documents": documents,
        "embedding": get_embeddings(),
        "collection_name": collection_name,
        # Cosine space → distance in ~[0, 2]; we convert to similarity below.
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    if persist_directory:
        kwargs["persist_directory"] = persist_directory
    return Chroma.from_documents(**kwargs)


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine *distance* to a [0, 1]-ish similarity score.

    With hnsw:space=cosine, Chroma returns distance ≈ 1 - cos_sim for unit
    vectors (often in [0, 2]). Higher similarity = more alike.
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


def retrieve_similar(
    store: Chroma,
    query: str,
    *,
    k: int = 3,
    version: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Return top-k (document, similarity) pairs. Higher similarity = closer match.

    Optional metadata filter restricts search to one contract version (A or B).
    """
    search_kwargs: dict = {"k": k}
    if version is not None:
        search_kwargs["filter"] = {"version": version}

    raw = store.similarity_search_with_score(query, **search_kwargs)
    return [(doc, _distance_to_similarity(dist)) for doc, dist in raw]
