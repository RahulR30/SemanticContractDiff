# SemanticContractDiff — how the four buzzwords work *here*

This project now has two pipelines:

| Mode | Path | Idea |
|------|------|------|
| **Classic** | `main/pipeline.py` | Align paragraphs by index → MiniLM cosine → LLM if low score |
| **RAG** | `main/rag_pipeline.py` | Chunk → Chroma vector DB → retrieve → LangChain LLM |

Use RAG mode when you want to honestly talk about **chunking, vector DB, LangChain, and RAG**.

---

## 1. Chunking (`main/chunking.py`)

**What it is:** Splitting long PDF text into smaller overlapping pieces.

**What we use:** LangChain `RecursiveCharacterTextSplitter`
- `chunk_size=500`
- `chunk_overlap=100`
- Prefers `\n\n`, then `\n`, then `. `, then spaces

**Why:** Embedding/search works better on clause-sized units than on a whole PDF. Overlap avoids cutting an important sentence in half.

**Interview line:**  
> “We chunk contracts with a recursive character splitter — about 500 characters with 100 overlap — and attach metadata like version A/B and chunk index.”

---

## 2. Vector DB (`main/vector_store.py`)

**What it is:** Store embeddings and run similarity search (“find nearest chunks”).

**What we use:** **Chroma** + HuggingFace MiniLM embeddings  
(`sentence-transformers/paraphrase-MiniLM-L6-v2`)

**What happens:**
1. Each chunk → embedding vector
2. Upsert into a Chroma collection with metadata `version=A|B`
3. Query with `similarity_search_with_relevance_scores` (optional filter on version)

**Interview line:**  
> “Chunks are embedded with MiniLM and indexed in Chroma. For each revised chunk I retrieve the top-k original chunks by cosine/relevance score, filtered to version A.”

---

## 3. LangChain (`main/chunking.py`, `vector_store.py`, `rag_chain.py`)

**What it is:** Glue library — not the model itself.

**Where it shows up in this repo:**
- Text splitting (`langchain_text_splitters`)
- Documents + Chroma / HuggingFace embeddings (`langchain_chroma`, `langchain_huggingface`)
- Chat prompts + OpenRouter LLM (`langchain_openai` + `ChatPromptTemplate`)

**Interview line:**  
> “LangChain orchestrates load/split/embed/retrieve/generate. The prompts and threshold logic are still ours.”

---

## 4. RAG (`main/rag_pipeline.py`, `main/rag_chain.py`)

**What it is:** Retrieve relevant text first, then generate with an LLM (don’t stuff the whole PDF).

**Two RAG uses in this project:**

### A) Diff RAG (`run_rag_pipeline`)
1. Chunk A and B  
2. Index both in Chroma  
3. For each B chunk, retrieve best A match  
4. If relevance **&lt; threshold** → flag as changed  
5. LangChain chain analyzes original vs revised excerpts only  

### B) Q&A RAG (`ask_contract_question`)
1. Index both versions  
2. Embed the user question  
3. Retrieve top-k chunks from either version  
4. LLM answers **only** from that context  

**Interview line:**  
> “It’s RAG: we retrieve the relevant contract spans from Chroma, then the LLM generates an analysis or answer grounded in those spans.”

---

## How to run

```bash
cd SemanticContractDiff
source .venv/bin/activate   # or create one
pip install -r requirements.txt
streamlit run app.py
```

In the UI:
1. Choose **RAG (chunk + Chroma + LangChain)**
2. Upload `demo_contract_v1.pdf` / `demo_contract_v2.pdf`
3. Run diff — and/or use the **Ask a question** tab

Requires `OPENROUTER_API_KEY` in `.env` for LLM steps.

---

## File map

```
main/chunking.py       ← Chunking
main/vector_store.py   ← Vector DB (Chroma)
main/rag_chain.py      ← LangChain prompts + LLM
main/rag_pipeline.py   ← Full RAG orchestration
main/pipeline.py       ← Classic (non-RAG) path still available
app.py                 ← UI toggle + Q&A tab
```
