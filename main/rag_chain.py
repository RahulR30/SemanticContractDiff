"""LangChain RAG generation: retrieved context → LLM analysis.

'RAG' = Retrieval-Augmented Generation. The LLM only sees the chunks we
retrieved (original + revised), not the entire PDF.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")


def make_llm(model: Optional[str] = None) -> ChatOpenAI:
    """OpenRouter via the OpenAI-compatible LangChain chat client."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment / .env")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        temperature=0,
    )


CHANGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze contract edits. Using ONLY the provided original and "
            "revised excerpts, say whether the change affects legal liability "
            "or financial obligations. Be concise (2-4 sentences).",
        ),
        (
            "human",
            "ORIGINAL CHUNK:\n{original}\n\nREVISED CHUNK:\n{revised}\n\n"
            "Retrieval note: similarity between these spans was {score:.3f}.",
        ),
    ]
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about contracts using ONLY the retrieved "
            "context. If the context is insufficient, say so. Cite which "
            "version (A=original, B=revised) a fact comes from when possible.",
        ),
        (
            "human",
            "Question: {question}\n\nRetrieved context:\n{context}",
        ),
    ]
)


def analyze_change(
    original: str,
    revised: str,
    score: float,
    *,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """RAG-style generation for a single retrieved change pair."""
    chain = CHANGE_PROMPT | (llm or make_llm())
    result = chain.invoke(
        {"original": original, "revised": revised, "score": score}
    )
    return result.content if hasattr(result, "content") else str(result)


def answer_question(
    question: str,
    context: str,
    *,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """Classic Q&A RAG over retrieved chunks."""
    chain = QA_PROMPT | (llm or make_llm())
    result = chain.invoke({"question": question, "context": context})
    return result.content if hasattr(result, "content") else str(result)
