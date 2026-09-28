"""Integration pipeline for SemanticContractDiff. Written by Rahul Rao."""

from typing import List
from main.extract_paragraphs import ExtractParagraphs
from main.score_map import compute_similarity
from main.LLM_Orchestrator import Orchestrator


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
            clause_index, original_text, revised_text, score, analysis
    """
    paragraphs_a: List[str] = ExtractParagraphs(pdf_a_path).text_to_paragraph()
    paragraphs_b: List[str] = ExtractParagraphs(pdf_b_path).text_to_paragraph()

    # Compare shared positions, then preserve additions/deletions as unmatched
    # clauses instead of silently truncating the longer document.
    shared = min(len(paragraphs_a), len(paragraphs_b))
    scores = compute_similarity(paragraphs_a[:shared], paragraphs_b[:shared])
    length = max(len(paragraphs_a), len(paragraphs_b))
    paragraphs_a += [""] * (length - len(paragraphs_a))
    paragraphs_b += [""] * (length - len(paragraphs_b))
    scores += [0.0] * (length - shared)

    results: List[dict] = Orchestrator(threshold=threshold).analyze(
        paragraphs_a, paragraphs_b, scores
    )

    return results
