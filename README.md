# SemanticContractDiff

Compare two contract PDFs and flag substantive legal / financial changes.

## Pipelines

| Mode | Entry point | Stack |
|------|-------------|--------|
| **RAG** (default in UI) | `main/rag_pipeline.py` | Chunking + **Chroma** vector DB + **LangChain** RAG |
| **Classic** | `main/pipeline.py` | Index-aligned MiniLM cosine + OpenRouter LLM |

Read **[LEARNING.md](LEARNING.md)** for how chunking, vector DB, LangChain, and RAG map onto this repo (interview prep).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# put OPENROUTER_API_KEY in .env
streamlit run app.py
```

Demo PDFs: `demo_contract_v1.pdf`, `demo_contract_v2.pdf`.

## Tests

```bash
python -m pytest test/ -q
```
