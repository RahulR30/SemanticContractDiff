"""RAG pipeline: chunk → vector DB → retrieve → LangChain generate.

End-to-end flow for SemanticContractDiff using all four concepts:
  1. Chunking   (RecursiveCharacterTextSplitter)
  2. Vector DB  (Chroma + MiniLM embeddings)
  3. RAG        (retrieve changed spans, then generate)
  4. LangChain  (splitters, vector store, chat prompts/chains)
"""

from __future__ import annotations

import tempfile
from typing import List, Optional

from main.chunking import chunk_text
from main.extract_paragraphs import ExtractParagraphs
from main.rag_chain import analyze_change, answer_question, make_llm
from main.vector_store import build_chroma_store, retrieve_similar


def run_rag_pipeline(
    pdf_a_path: str,
    pdf_b_path: str,
    *,
    threshold: float = 0.95,
    top_k: int = 3,
    persist_directory: Optional[str] = None,
    call_llm: bool = True,
) -> List[dict]:
    """Compare two contracts via vector retrieval instead of index alignment.

    Steps:
      1. Extract raw text from both PDFs.
      2. Chunk each version (overlapping windows).
      3. Upsert all chunks into Chroma with metadata version=A|B.
      4. For each B chunk, retrieve top-k A chunks (vector search).
      5. If best relevance < threshold → treat as a substantive change.
      6. Optionally run LangChain RAG generation on that pair.

    Returns the same shape as the classic pipeline for UI compatibility:
      clause_index, original_text, revised_text, score, analysis
    """
    text_a = ExtractParagraphs(pdf_a_path).text
    text_b = ExtractParagraphs(pdf_b_path).text

    docs_a = chunk_text(text_a, version="A", source=pdf_a_path)
    docs_b = chunk_text(text_b, version="B", source=pdf_b_path)

    if not docs_a or not docs_b:
        return []

    # Persist under a temp dir when caller does not pass one — still a real
    # on-disk Chroma store for the duration of the run (interview-accurate).
    tmp_ctx = None
    store_dir = persist_directory
    if store_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="scd_chroma_")
        store_dir = tmp_ctx.name

    try:
        store = build_chroma_store(
            docs_a + docs_b,
            collection_name="contract_diff",
            persist_directory=store_dir,
        )

        llm = make_llm() if call_llm else None
        results: List[dict] = []

        for b_doc in docs_b:
            hits = retrieve_similar(
                store, b_doc.page_content, k=top_k, version="A"
            )
            if not hits:
                best_doc, best_score = None, 0.0
            else:
                best_doc, best_score = hits[0]

            # Low similarity ⇒ revised chunk does not closely match any original.
            if best_score >= threshold:
                continue

            original_text = best_doc.page_content if best_doc is not None else ""
            revised_text = b_doc.page_content

            if call_llm and llm is not None:
                analysis = analyze_change(
                    original_text, revised_text, float(best_score), llm=llm
                )
            else:
                analysis = (
                    "[LLM skipped] Low vector similarity to nearest original chunk."
                )

            results.append(
                {
                    "clause_index": int(b_doc.metadata.get("chunk_index", len(results))),
                    "original_text": original_text,
                    "revised_text": revised_text,
                    "score": round(float(best_score), 4),
                    "analysis": analysis,
                    # Extra fields for learning / debugging in the UI
                    "retrieval": "chroma_top1",
                    "matched_a_chunk": int(best_doc.metadata["chunk_index"])
                    if best_doc is not None
                    else None,
                }
            )

        return results
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def ask_contract_question(
    pdf_a_path: str,
    pdf_b_path: str,
    question: str,
    *,
    top_k: int = 4,
    persist_directory: Optional[str] = None,
) -> dict:
    """True Q&A RAG: embed the question, retrieve from both versions, generate."""
    text_a = ExtractParagraphs(pdf_a_path).text
    text_b = ExtractParagraphs(pdf_b_path).text
    docs = chunk_text(text_a, version="A", source=pdf_a_path) + chunk_text(
        text_b, version="B", source=pdf_b_path
    )

    tmp_ctx = None
    store_dir = persist_directory
    if store_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="scd_chroma_qa_")
        store_dir = tmp_ctx.name

    try:
        store = build_chroma_store(
            docs,
            collection_name="contract_qa",
            persist_directory=store_dir,
        )
        hits = retrieve_similar(store, question, k=top_k)
        context_parts = []
        for doc, score in hits:
            ver = doc.metadata.get("version", "?")
            context_parts.append(
                f"[version={ver} | score={score:.3f}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(no hits)"
        answer = answer_question(question, context)
        return {
            "question": question,
            "answer": answer,
            "retrieved": [
                {
                    "version": d.metadata.get("version"),
                    "score": round(float(s), 4),
                    "text": d.page_content,
                }
                for d, s in hits
            ],
        }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
