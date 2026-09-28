"""
validator.py — Lightweight answer validation and English/Hindi parity checks.

validate_answer() checks that a canonical answer:
1. Covers the required_points from the Answer Plan.
2. Is grounded in the retrieved evidence.
3. Does not violate the plan's restrictions.
4. Keeps any safety caution the question calls for.

parity_report() checks that an English answer and a Hindi answer say the same thing.
All functions here are pure (no LLM calls) so they can be unit-tested offline.
"""

import re

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "if", "while", "about", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "which", "what", "who", "whom",
    "also", "any", "like", "may", "one", "get", "make", "help",
}


def _extract_key_terms(text: str) -> list[str]:
    """Extract meaningful terms from text, filtering stop words."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


_WORD_START = r"(?<![a-z])"  # a word boundary before the term


def _has_word(term: str, text: str) -> bool:
    """True if a word in `text` starts with `term` ("walk" matches "walking", "take" not "intake")."""
    return re.search(_WORD_START + re.escape(term), text) is not None


def _terms_present(terms: list[str], text: str, threshold: float = 0.5) -> bool:
    """Check if at least `threshold` fraction of terms appear in text."""
    if not terms:
        return True
    found = sum(1 for t in terms if _has_word(t, text))
    return (found / len(terms)) >= threshold


def _stems(text: str) -> set[str]:
    """Crude stems (first 5 letters) so 'recommended'/'recommends' match across paraphrases."""
    return {t[:5] for t in _extract_key_terms(text)}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•*") for p in parts if len(p.strip(" -•*")) > 20]


def _sentence_coverage(source: str, target: str, threshold: float = 0.5) -> float:
    """Fraction of sentences in `source` whose key stems mostly appear in `target`."""
    sentences = _split_sentences(source)
    if not sentences:
        return 1.0
    target_stems = _stems(target)
    covered = 0
    for s in sentences:
        s_stems = _stems(s)
        if not s_stems or len(s_stems & target_stems) / len(s_stems) >= threshold:
            covered += 1
    return covered / len(sentences)


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def extract_numbers(text: str) -> set[str]:
    """Extract numbers (incl. decimals) from text; Devanagari digits are normalised."""
    normalised = (text or "").translate(_DEVANAGARI_DIGITS)
    return {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", normalised)}


# ---------------------------------------------------------------------------
# Safety cautions (documentRAG.md Sections 15, 18, 28, 36)
# ---------------------------------------------------------------------------

# Regexes matched at a word start in lowercased text, so "kid" never matches "kidney";
# "insulin resistance/sensitivity" is a condition, not a medication.
CAUTION_CONCEPTS = {
    "diabetes_medication": [r"insulin(?!\s+(?:resistan|sensitiv))", "sulfonylurea",
                            "glimepiride", "medication", "medicine"],
    "pregnancy": ["pregnan", "breast-?feed", "postpartum"],
    "children": ["child", "adolescen", "kids?(?![a-z])", "pa?ediatric"],
    "eating_disorder": ["eating disorder", "disordered eating", "binge", "purging"],
}

REFERRAL_TERMS = ["doctor", "physician", "paediatrician", "pediatrician",
                  "obstetrician", "clinician", "gynaecologist", "endocrinologist"]


def detect_cautions(text: str) -> set[str]:
    """Return the caution concepts mentioned in the text."""
    lower = (text or "").lower()
    return {name for name, patterns in CAUTION_CONCEPTS.items()
            if any(re.search(_WORD_START + p, lower) for p in patterns)}


def has_referral(text: str) -> bool:
    lower = (text or "").lower()
    return any(_has_word(t, lower) for t in REFERRAL_TERMS)


def required_cautions(question: str, chunks: list[dict]) -> set[str]:
    """Cautions the answer must carry: raised by the question AND covered by the evidence."""
    evidence = " ".join(c.get("text", "") for c in chunks)
    return detect_cautions(question) & detect_cautions(evidence)


def safety_caution_check(answer: str, question: str, chunks: list[dict]) -> dict:
    required = required_cautions(question, chunks)
    present = detect_cautions(answer)
    missing = sorted(required - present)
    referral_ok = has_referral(answer) if required else True
    passed = not missing and referral_ok
    details = "No safety caution required"
    if required:
        details = f"Required: {sorted(required)}; missing: {missing}; doctor referral: {referral_ok}"
    return {
        "name": "safety_cautions",
        "passed": passed,
        "required": sorted(required),
        "missing": missing,
        "referral_present": referral_ok,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Answer validation
# ---------------------------------------------------------------------------

def validate_answer(answer: str, plan: dict, chunks: list[dict], question: str = "") -> dict:
    """Run lightweight validation checks on the canonical English answer.

    Returns {"passed": bool, "score": float, "checks": [...], "feedback": str}
    where `feedback` describes failures for a regeneration attempt.
    """
    checks = []
    answer_lower = answer.lower()

    # Check 1: Required points coverage
    required_points = plan.get("required_points", [])
    missing_points = [p for p in required_points
                      if not _terms_present(_extract_key_terms(p), answer_lower)]
    coverage = 1 - len(missing_points) / len(required_points) if required_points else 1.0
    checks.append({
        "name": "required_points_coverage",
        "passed": coverage >= 0.6,
        "coverage": round(coverage, 2),
        "missing": missing_points,
        "details": f"{len(required_points) - len(missing_points)}/{len(required_points)} required points covered",
    })

    # Check 2: Evidence grounding
    evidence_text = " ".join(c.get("text", "") for c in chunks).lower()
    answer_sentences = _split_sentences(answer)
    grounded = sum(1 for s in answer_sentences
                   if _terms_present(_extract_key_terms(s), evidence_text, threshold=0.4))
    grounding = grounded / max(len(answer_sentences), 1)
    checks.append({
        "name": "evidence_grounding",
        "passed": grounding >= 0.5,
        "grounding_ratio": round(grounding, 2),
        "details": f"{grounded}/{len(answer_sentences)} sentences grounded in evidence",
    })

    # Check 3: Restriction compliance (flags answers that echo a restriction's wording heavily)
    violations = [r for r in plan.get("restrictions", [])
                  if _extract_key_terms(r) and _terms_present(_extract_key_terms(r), answer_lower, threshold=0.7)]
    checks.append({
        "name": "restriction_compliance",
        "passed": not violations,
        "violations": violations,
        "details": f"{len(violations)} potential restriction violations",
    })

    # Check 4: Safety cautions
    checks.append(safety_caution_check(answer, question, chunks))

    all_passed = all(c["passed"] for c in checks)
    scores = [c.get("coverage", c.get("grounding_ratio", 1.0 if c["passed"] else 0.0)) for c in checks]

    feedback_lines = []
    if missing_points:
        feedback_lines.append("Include these required points: " + "; ".join(missing_points))
    safety = checks[-1]
    if not safety["passed"]:
        feedback_lines.append(
            f"State the safety caution clearly for: {', '.join(safety['required'])}, "
            "and tell the user to consult their doctor before changing their eating pattern."
        )
    if violations:
        feedback_lines.append("Do not do the following: " + "; ".join(violations))

    return {
        "passed": all_passed,
        "score": round(sum(scores) / len(scores), 2),
        "checks": checks,
        "feedback": "\n".join(feedback_lines),
    }


# ---------------------------------------------------------------------------
# English / Hindi parity
# ---------------------------------------------------------------------------

def _recommendation_terms(text: str) -> set[str]:
    lower = (text or "").lower()
    markers = {
        "consult", "doctor", "physician", "clinic", "specialist", "seek", "check",
        "monitor", "avoid", "limit", "reduce", "increase", "walk", "exercise",
        "stop", "start", "take", "follow", "talk", "contact"
    }
    return {marker for marker in markers if _has_word(marker, lower)}


def parity_report(en_answer: str, hi_answer: str, hi_back_translation: str,
                  en_chunk_ids: list[int] | None = None,
                  hi_chunk_ids: list[int] | None = None) -> dict:
    """Compare an English answer with a Hindi answer (via its back-translation).

    The same function scores both the canonical pipeline and the baseline, so their
    numbers are directly comparable. Question intent is not a check here: a pair that
    fails it never reaches this function.
    """
    checks = []

    if en_chunk_ids is not None and hi_chunk_ids is not None:
        a, b = set(en_chunk_ids), set(hi_chunk_ids)
        overlap = len(a & b) / len(a | b) if a | b else 1.0
        checks.append({
            "name": "retrieval_overlap",
            "passed": overlap >= 0.6,
            "value": round(overlap, 2),
            "details": f"{len(a & b)}/{len(a | b)} evidence chunks shared",
        })

    en_nums, hi_nums = extract_numbers(en_answer), extract_numbers(hi_answer)
    checks.append({
        "name": "numbers_match",
        "passed": en_nums == hi_nums,
        "value": 1.0 if en_nums == hi_nums else 0.0,
        "missing_in_hindi": sorted(en_nums - hi_nums),
        "extra_in_hindi": sorted(hi_nums - en_nums),
        "details": f"EN {sorted(en_nums)} vs HI {sorted(hi_nums)}",
    })

    en_cautions = detect_cautions(en_answer)
    hi_cautions = detect_cautions(hi_back_translation)
    en_ref, hi_ref = has_referral(en_answer), has_referral(hi_back_translation)
    cautions_ok = en_cautions <= hi_cautions and (hi_ref or not en_ref)
    checks.append({
        "name": "cautions_preserved",
        "passed": cautions_ok,
        "value": 1.0 if cautions_ok else 0.0,
        "details": f"EN {sorted(en_cautions)} referral={en_ref} | HI {sorted(hi_cautions)} referral={hi_ref}",
    })

    en_recommendations = _recommendation_terms(en_answer)
    hi_recommendations = _recommendation_terms(hi_back_translation)
    recommendation_ok = en_recommendations <= hi_recommendations
    checks.append({
        "name": "recommendation_match",
        "passed": recommendation_ok,
        "value": 1.0 if recommendation_ok else 0.0,
        "details": f"EN recs={sorted(en_recommendations)} | HI recs={sorted(hi_recommendations)}",
    })

    forward = _sentence_coverage(en_answer, hi_back_translation)
    backward = _sentence_coverage(hi_back_translation, en_answer)
    coverage = min(forward, backward)
    checks.append({
        "name": "content_coverage",
        "passed": coverage >= 0.8,
        "value": round(coverage, 2),
        "details": f"EN→HI {forward:.0%} of English content kept; HI→EN {backward:.0%} of Hindi content grounded in English",
    })

    return {
        "parity_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
