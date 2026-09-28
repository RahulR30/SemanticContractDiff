# SemanticContractDiff

Compare two contract PDFs with semantic similarity and retrieve relevant passages for follow-up questions. Built with Python, PyMuPDF, MiniLM embeddings, Chroma, LangChain, and Streamlit.

## Architecture

1. **Extract:** read PDF text, preserve boundary lines and page separation, and normalize Unicode ligatures.
2. **Compare:** classic mode scores paragraphs at corresponding positions; RAG mode chunks documents and retrieves the nearest original passage for each revised chunk.
3. **Explain:** send low-similarity pairs to an OpenRouter-hosted model for an explanation.
4. **Explore:** the Streamlit interface shows the paired text and provides retrieval-backed questions across both versions.

Classic mode retains unmatched trailing paragraphs as additions or deletions. The embedding model loads only when nonempty inputs require similarity computation, so importing the pipeline and comparing empty inputs does not download weights.

## Run locally

Use Python 3.11 or 3.12:

```bash
git clone https://github.com/RahulR30/SemanticContractDiff.git
cd SemanticContractDiff
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file containing `OPENROUTER_API_KEY=your-key`, then run:

```bash
streamlit run app.py
```

Upload `demo_contract_v1.pdf` and `demo_contract_v2.pdf` to explore the example. The first similarity computation downloads MiniLM weights; explanations and Q&A require an external model API.

## Automated verification

The default suite needs no API credentials or model weights. It generates real PDFs for extraction tests and uses deterministic embeddings and mocked model responses to verify processing logic.

```bash
pip install -r requirements-test.txt
python -m pytest test/ -q
```

Coverage includes single-line PDFs, first/last clause preservation, PDF cleanup after an extraction error, additions/deletions at document ends, empty inputs, cosine scoring, chunk metadata, and the threshold that decides whether to request an explanation. GitHub Actions runs this suite on Python 3.11 and 3.12.

The original model-backed similarity tests remain available separately after installing the application dependencies:

```bash
python -m pytest test/test_score_map.py -m integration -q
```

Those tests download model weights and assess the embedding model itself. They are separate from deterministic regression checks.

## Code map

- `main/extract_paragraphs.py`: PDF extraction and paragraph splitting.
- `main/pipeline.py`: classic comparison with unmatched-tail preservation.
- `main/score_map.py`: lazy model loading and pairwise cosine scoring.
- `main/chunking.py`: overlapping text chunks with source/version metadata.
- `main/vector_store.py`: Chroma storage and similarity retrieval.
- `main/rag_pipeline.py`: retrieval-based comparison and questions.
- `main/LLM_Orchestrator.py`: threshold-based explanation requests.
- `test/`: deterministic and opt-in model integration tests.
- `bench/latency_harness.py`: an experimental comparison of per-paragraph and batched embedding computation; no unmeasured speedup is claimed here.

## Interpretation and limits

Similarity is a text-comparison signal, not a measure of legal importance. Review the source clauses before drawing conclusions.

Classic mode aligns by paragraph position; insertions in the middle can shift later matches. RAG comparison searches from revised chunks to original chunks, so it does not comprehensively detect deletions. Scanned PDFs need OCR, which is not implemented. The extractor preserves all boundary lines because a first or last line is not reliably a header or footer. The default tests do not validate live model answers or persistent Chroma behavior.

## Contributing

Include a minimal document or mocked input that reproduces the issue, add a regression test, and describe the expected comparison behavior. Do not commit confidential documents or API keys.
