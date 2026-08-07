"""Latency harness for the embedding + retrieval step of SemanticContractDiff.

Compares two implementations of the paragraph-similarity step:

    baseline : encode paragraphs one at a time (no batching) and score with a
               per-pair sklearn cosine_similarity loop. This mirrors the naive
               "encode-per-query" approach.
    tuned    : batch-encode all paragraphs in one call with normalized
               embeddings, then score every pair with a single vectorized
               dot product.

Both produce the same cosine scores (asserted), so the only thing that differs
is how the work is scheduled — which is exactly the latency being measured.

The real demo contracts are small, so we also replicate their paragraphs up to
several sizes to simulate "large document sets" and show how the gap scales.

Run:
    python -m bench.latency_harness
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from main.extract_paragraphs import ExtractParagraphs

MODEL_NAME = "paraphrase-MiniLM-L6-v2"
DEMO_A = "demo_contract_v1.pdf"
DEMO_B = "demo_contract_v2.pdf"

# Paragraph-pair counts to benchmark. The real contracts seed the text; we tile
# them to reach each size so "large document set" behavior is visible.
SIZES = [25, 100, 400]

WARMUP_RUNS = 1
TIMED_RUNS = 3

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------------------------------- #
# Implementations under test
# --------------------------------------------------------------------------- #
def score_baseline(list_a: list[str], list_b: list[str]) -> list[float]:
    """Naive: encode each paragraph on its own, score each pair in a loop."""
    scores: list[float] = []
    for a, b in zip(list_a, list_b):
        emb_a = model.encode(a)
        emb_b = model.encode(b)
        score = cosine_similarity(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0][0]
        scores.append(float(score))
    return scores


def score_tuned(list_a: list[str], list_b: list[str]) -> list[float]:
    """Batched encode + normalized dot product for the whole set at once."""
    emb_a = model.encode(
        list_a, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
    )
    emb_b = model.encode(
        list_b, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
    )
    # Row-wise cosine == row-wise dot product once vectors are unit-normalized.
    scores = np.einsum("ij,ij->i", emb_a, emb_b)
    return scores.astype(float).tolist()


# --------------------------------------------------------------------------- #
# Timing utilities
# --------------------------------------------------------------------------- #
def time_fn(
    fn: Callable[[list[str], list[str]], list[float]],
    list_a: list[str],
    list_b: list[str],
) -> list[float]:
    """Return per-run wall-clock times (seconds) after a warmup."""
    for _ in range(WARMUP_RUNS):
        fn(list_a, list_b)

    times: list[float] = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        fn(list_a, list_b)
        times.append(time.perf_counter() - start)
    return times


def build_pairs(size: int) -> tuple[list[str], list[str]]:
    """Tile the real contract paragraphs up to `size` aligned pairs."""
    paras_a = ExtractParagraphs(DEMO_A).text_to_paragraph()
    paras_b = ExtractParagraphs(DEMO_B).text_to_paragraph()
    n = min(len(paras_a), len(paras_b))
    paras_a, paras_b = paras_a[:n], paras_b[:n]

    reps = (size + n - 1) // n
    list_a = (paras_a * reps)[:size]
    list_b = (paras_b * reps)[:size]
    return list_a, list_b


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"Model: {MODEL_NAME}")
    print(f"Warmup runs: {WARMUP_RUNS}  Timed runs: {TIMED_RUNS}\n")

    header = (
        f"{'pairs':>6} | {'baseline p50':>13} | {'tuned p50':>10} | "
        f"{'speedup':>8} | {'latency cut':>11}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for size in SIZES:
        list_a, list_b = build_pairs(size)

        base_scores = score_baseline(list_a, list_b)
        tuned_scores = score_tuned(list_a, list_b)
        # Same math, so results must agree — proves the speedup isn't cheating.
        max_diff = max(abs(a - b) for a, b in zip(base_scores, tuned_scores))
        assert max_diff < 1e-4, f"score mismatch at size {size}: {max_diff}"

        base_times = time_fn(score_baseline, list_a, list_b)
        tuned_times = time_fn(score_tuned, list_a, list_b)

        base_p50 = statistics.median(base_times)
        tuned_p50 = statistics.median(tuned_times)
        speedup = base_p50 / tuned_p50
        cut_pct = (1 - tuned_p50 / base_p50) * 100

        print(
            f"{size:>6} | {base_p50 * 1000:>10.1f} ms | {tuned_p50 * 1000:>7.1f} ms | "
            f"{speedup:>7.1f}x | {cut_pct:>9.1f} %",
            flush=True,
        )

    print(
        "\nnote: p50 of "
        f"{TIMED_RUNS} timed runs (after {WARMUP_RUNS} warmup). "
        "Embedding + similarity only; LLM step excluded."
    )


if __name__ == "__main__":
    main()
