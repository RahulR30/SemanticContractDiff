"""Streamlit UI for SemanticContractDiff — classic + RAG modes."""

import os
import tempfile

import streamlit as st

from main.pipeline import run_pipeline
from main.rag_pipeline import ask_contract_question, run_rag_pipeline

st.set_page_config(page_title="Semantic Contract Diff", layout="wide")
st.title("Semantic Contract Diff")
st.caption(
    "Detects substantive legal changes between two contract versions. "
    "Classic mode = index-aligned embeddings. RAG mode = chunking + Chroma + LangChain."
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Pipeline mode",
        options=["RAG (chunk + Chroma + LangChain)", "Classic (index-aligned)"],
        index=0,
        help="RAG uses overlapping chunks, a Chroma vector DB, and LangChain generation.",
    )
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.95,
        step=0.01,
        help="Below this relevance/similarity score → flagged and sent to the LLM.",
    )
    top_k = st.slider("Retriever top-k (RAG only)", 1, 8, 3)
    st.markdown("---")
    st.markdown(
        "**Buzzword map**\n"
        "- **Chunking** → `main/chunking.py`\n"
        "- **Vector DB** → Chroma in `main/vector_store.py`\n"
        "- **LangChain** → splitters + prompts in `rag_chain.py`\n"
        "- **RAG** → `main/rag_pipeline.py`"
    )

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Original contract (Version A)", type="pdf")
with col_b:
    file_b = st.file_uploader("Revised contract (Version B)", type="pdf")

tab_diff, tab_qa = st.tabs(["Diff analysis", "Ask a question (RAG)"])

with tab_diff:
    if file_a and file_b:
        if st.button("Run analysis", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path_a = os.path.join(tmpdir, "version_a.pdf")
                path_b = os.path.join(tmpdir, "version_b.pdf")
                chroma_dir = os.path.join(tmpdir, "chroma")

                with open(path_a, "wb") as f:
                    f.write(file_a.read())
                with open(path_b, "wb") as f:
                    f.write(file_b.read())

                with st.spinner("Running pipeline..."):
                    try:
                        if mode.startswith("RAG"):
                            results = run_rag_pipeline(
                                path_a,
                                path_b,
                                threshold=threshold,
                                top_k=top_k,
                                persist_directory=chroma_dir,
                            )
                        else:
                            results = run_pipeline(
                                path_a, path_b, threshold=threshold
                            )
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                        st.stop()

            if not results:
                st.success("No substantive changes detected above the threshold.")
            else:
                st.subheader(f"{len(results)} clause(s) / chunk(s) flagged as changed")
                for item in results:
                    score = item.get("score", 0.0)
                    is_critical = score < 0.80
                    badge = ":red[CRITICAL]" if is_critical else ":grey[MINOR]"
                    header = (
                        f"Clause/chunk {item['clause_index']} — "
                        f"similarity {score:.2f}  {badge}"
                    )
                    with st.expander(header, expanded=is_critical):
                        left, right = st.columns(2)
                        highlight_color = "#ffd6d6" if is_critical else "#f0f0f0"
                        with left:
                            st.markdown("**Version A (original / nearest match)**")
                            st.markdown(
                                f"<div style='background:{highlight_color};padding:10px;"
                                f"border-radius:6px'>{item.get('original_text', '')}</div>",
                                unsafe_allow_html=True,
                            )
                        with right:
                            st.markdown("**Version B (revised)**")
                            st.markdown(
                                f"<div style='background:{highlight_color};padding:10px;"
                                f"border-radius:6px'>{item.get('revised_text', '')}</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown("**LLM analysis**")
                        st.info(item.get("analysis", "No analysis returned."))
                        if item.get("retrieval"):
                            st.caption(
                                f"retrieval={item['retrieval']} · "
                                f"matched A chunk={item.get('matched_a_chunk')}"
                            )
    else:
        st.info("Upload both contract PDFs to begin.")

with tab_qa:
    st.markdown(
        "Uses **RAG**: embed your question → Chroma top-k over both versions → LangChain LLM."
    )
    question = st.text_input(
        "Question",
        placeholder="e.g. Did the indemnity or payment terms get worse for us?",
    )
    if file_a and file_b and question:
        if st.button("Ask", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path_a = os.path.join(tmpdir, "version_a.pdf")
                path_b = os.path.join(tmpdir, "version_b.pdf")
                chroma_dir = os.path.join(tmpdir, "chroma_qa")
                with open(path_a, "wb") as f:
                    f.write(file_a.getbuffer())
                with open(path_b, "wb") as f:
                    f.write(file_b.getbuffer())
                with st.spinner("Retrieving + generating..."):
                    try:
                        out = ask_contract_question(
                            path_a,
                            path_b,
                            question,
                            top_k=top_k,
                            persist_directory=chroma_dir,
                        )
                    except Exception as e:
                        st.error(f"RAG Q&A failed: {e}")
                        st.stop()
            st.subheader("Answer")
            st.write(out["answer"])
            st.subheader("Retrieved chunks")
            for hit in out["retrieved"]:
                with st.expander(
                    f"version={hit['version']} · score={hit['score']}", expanded=False
                ):
                    st.write(hit["text"])
    elif not (file_a and file_b):
        st.info("Upload both PDFs first.")
