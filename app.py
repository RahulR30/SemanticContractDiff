"""Streamlit UI for SemanticContractDiff. Written by Rahul Rao."""

import os
import tempfile
import streamlit as st
from main.pipeline import run_pipeline

st.set_page_config(page_title="Semantic Contract Diff", layout="wide")
st.title("Semantic Contract Diff")
st.caption("Detects substantive legal changes between two contract versions using embeddings + LLM analysis.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.95,
        step=0.01,
        help="Paragraphs with cosine similarity below this value are flagged and sent to the LLM.",
    )

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Original contract (Version A)", type="pdf")
with col_b:
    file_b = st.file_uploader("Revised contract (Version B)", type="pdf")

if file_a and file_b:
    if st.button("Run analysis", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "version_a.pdf")
            path_b = os.path.join(tmpdir, "version_b.pdf")

            with open(path_a, "wb") as f:
                f.write(file_a.read())
            with open(path_b, "wb") as f:
                f.write(file_b.read())

            with st.spinner("Extracting text and computing embeddings..."):
                try:
                    results = run_pipeline(path_a, path_b, threshold=threshold)
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.stop()

        if not results:
            st.success("No substantive changes detected above the similarity threshold.")
        else:
            st.subheader(f"{len(results)} clause(s) flagged as changed")

            for item in results:
                score = item.get("score", 0.0)
                is_critical = score < 0.80

                badge = ":red[CRITICAL]" if is_critical else ":grey[MINOR]"
                header = f"Clause {item['clause_index']} — similarity {score:.2f}  {badge}"

                with st.expander(header, expanded=is_critical):
                    left, right = st.columns(2)

                    highlight_color = "#ffd6d6" if is_critical else "#f0f0f0"

                    with left:
                        st.markdown("**Version A (original)**")
                        st.markdown(
                            f"<div style='background:{highlight_color};padding:10px;border-radius:6px'>"
                            f"{item.get('original_text', '')}</div>",
                            unsafe_allow_html=True,
                        )

                    with right:
                        st.markdown("**Version B (revised)**")
                        st.markdown(
                            f"<div style='background:{highlight_color};padding:10px;border-radius:6px'>"
                            f"{item.get('revised_text', '')}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("**LLM analysis**")
                    st.info(item.get("analysis", "No analysis returned."))
else:
    st.info("Upload both contract PDFs to begin.")
