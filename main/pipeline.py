"""Integration pipeline for SemanticContractDiff. Written by Rahul Rao."""

from typing import List
from main.extract_paragraphs import ExtractParagraphs
from main.similarity_scorer import SimilarityScorer
from main.llm_analyzer import LLMAnalyzer


def run_pipeline(
    pdf_a_path: str,
    pdf_b_path: str,
    threshold: float = 0.95,
) -> List[dict]:
    """
    Full pipeline: two PDF paths → list of JSON analysis objects for changed clauses.

    Args:
        pdf_a_path: Path to the original contract PDF.
        pdf_b_path: Path to the revised contract PDF.
        threshold:  Cosine similarity cutoff. Paragraphs below this are sent to the LLM.

    Returns:
        A list of dicts, each with keys:
            clause_index, original_text, score, analysis
    """
    paragraphs_a: List[str] = ExtractParagraphs(pdf_a_path).text_to_paragraph()
    paragraphs_b: List[str] = ExtractParagraphs(pdf_b_path).text_to_paragraph()

    # Align lists to the shorter document so indices stay consistent
    length = min(len(paragraphs_a), len(paragraphs_b))
    paragraphs_a = paragraphs_a[:length]
    paragraphs_b = paragraphs_b[:length]

    scores: List[float] = SimilarityScorer(paragraphs_a, paragraphs_b).compute_scores()

    results: List[dict] = LLMAnalyzer(
        paragraphs=paragraphs_b,
        scores=scores,
        threshold=threshold,
    ).analyze()

    return results
