"""
metrics.py — Consistency metrics for English ↔ Hindi ↔ Hinglish evaluation.

Pure functions: no LLM calls and no model loading, so they are unit-testable offline.
Semantic similarity is optional: pass `embed` (a function list[str] → normalised vectors)
to use embeddings; without it the metrics fall back to the lexical stems used by validator.py.

All text compared here is English: Hindi and Hinglish answers are compared through their
back-translations, the same way validator.parity_report does.
"""

from typing import Callable, Sequence

from validator import (
    _extract_key_terms,
    _sentence_coverage,
    _split_sentences,
    _stems,
    _terms_present,
    extract_numbers,
)

Embed = Callable[[list[str]], Sequence[Sequence[float]]]

# Pass/fail bar for the report. Retrieval uses the existing parity threshold (validator.parity_report).
THRESHOLDS = {
    "retrieval_overlap": 0.6,     # Jaccard of retrieved chunk ids vs English
    "missing_evidence": 0,        # English chunks the other language did not retrieve
    "plan_consistency": 0.9,
    "answer_similarity": 0.85,
    "hallucination_rate": 0.2,    # max share of ungrounded answer sentences
}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def chunk_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """Jaccard overlap of two chunk-id lists (the formula of parity_report's retrieval_overlap)."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def rank_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """Average overlap of the top-1 … top-k prefixes: rewards the same chunks in the same order."""
    k = max(len(a), len(b))
    if k == 0:
        return 1.0
    return sum(len(set(a[:d]) & set(b[:d])) / d for d in range(1, k + 1)) / k


def missing_evidence(reference: Sequence[int], other: Sequence[int]) -> list[int]:
    """Chunk ids in the reference (English) retrieval that the other language did not retrieve."""
    return sorted(set(reference) - set(other))


def score_drift(ref_chunks: list[dict], other_chunks: list[dict]) -> float | None:
    """Mean |Δ retrieval score| on the chunks both retrievals share (None when they share none)."""
    ref = {c["id"]: c.get("score", 0.0) for c in ref_chunks}
    shared = [abs(ref[c["id"]] - c.get("score", 0.0)) for c in other_chunks if c["id"] in ref]
    return round(sum(shared) / len(shared), 4) if shared else None


def retrieval_comparison(ref_chunks: list[dict], other_chunks: list[dict]) -> dict:
    """All retrieval metrics of one language against the English reference."""
    ref_ids = [c["id"] for c in ref_chunks]
    other_ids = [c["id"] for c in other_chunks]
    missing = missing_evidence(ref_ids, other_ids)
    return {
        "chunk_ids": other_ids,
        "overlap": round(chunk_overlap(ref_ids, other_ids), 2),
        "rank_overlap": round(rank_overlap(ref_ids, other_ids), 2),
        "missing_evidence": missing,
        "missing_count": len(missing),
        "score_drift": score_drift(ref_chunks, other_chunks),
    }


# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------

def _cosine_matrix(embed: Embed, left: list[str], right: list[str]) -> list[list[float]]:
    vectors = embed(left + right)
    lv, rv = vectors[:len(left)], vectors[len(left):]
    return [[float(sum(x * y for x, y in zip(a, b))) for b in rv] for a in lv]


def _stem_jaccard(a: str, b: str) -> float:
    sa, sb = _stems(a), _stems(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def text_similarity(a: str, b: str, embed: Embed | None = None) -> float:
    """Cosine similarity with `embed`, else stem Jaccard."""
    if not a and not b:
        return 1.0
    if embed is None:
        return _stem_jaccard(a, b)
    return _cosine_matrix(embed, [a], [b])[0][0]


# ---------------------------------------------------------------------------
# Answer plan
# ---------------------------------------------------------------------------

PLAN_FIELDS = ("required_points", "important_conditions", "restrictions")


def _plan_points(plan: dict) -> list[str]:
    return [str(p) for field in PLAN_FIELDS for p in (plan or {}).get(field, []) if str(p).strip()]


def plan_consistency(plan_a: dict, plan_b: dict, embed: Embed | None = None) -> float:
    """How far two Answer Plans say the same thing (1.0 = identical).

    Identical plans (the normal case: one stored plan per canonical question) score 1.0.
    Otherwise each point is matched to its closest point in the other plan, in both
    directions, and the lower of the two mean scores is returned.
    """
    a, b = _plan_points(plan_a), _plan_points(plan_b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if embed is not None:
        m = _cosine_matrix(embed, a, b)
        forward = sum(max(row) for row in m) / len(a)
        backward = sum(max(m[i][j] for i in range(len(a))) for j in range(len(b))) / len(b)
    else:
        forward = sum(max(_stem_jaccard(x, y) for y in b) for x in a) / len(a)
        backward = sum(max(_stem_jaccard(y, x) for x in a) for y in b) / len(b)
    return round(min(forward, backward), 4)


# ---------------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------------

def answer_similarity(reference: str, other: str, embed: Embed | None = None) -> dict:
    """Similarity of an answer (or a back-translation) to the English reference answer.

    `value` is the embedding cosine when `embed` is given, else the sentence coverage;
    coverage is the min of both directions, as in parity_report's content_coverage.
    """
    coverage = min(_sentence_coverage(reference, other), _sentence_coverage(other, reference))
    cosine = text_similarity(reference, other, embed) if embed is not None else None
    return {
        "value": round(cosine if cosine is not None else coverage, 4),
        "cosine": None if cosine is None else round(cosine, 4),
        "content_coverage": round(coverage, 4),
    }


def hallucination_rate(answer_english: str, chunks: list[dict]) -> dict:
    """Share of answer sentences NOT supported by the retrieved evidence, plus unsupported numbers.

    Uses the evidence-grounding rule of validator.validate_answer (≥ 40% of a sentence's key
    terms appear in the evidence). For Hindi/Hinglish pass the back-translation.
    """
    evidence = " ".join(c.get("text", "") for c in chunks)
    evidence_lower = evidence.lower()
    sentences = _split_sentences(answer_english or "")
    ungrounded = [s for s in sentences
                  if not _terms_present(_extract_key_terms(s), evidence_lower, threshold=0.4)]
    numbers = sorted(extract_numbers(answer_english) - extract_numbers(evidence))
    return {
        "rate": round(len(ungrounded) / len(sentences), 4) if sentences else 0.0,
        "ungrounded_sentences": ungrounded,
        "unsupported_numbers": numbers,
    }
