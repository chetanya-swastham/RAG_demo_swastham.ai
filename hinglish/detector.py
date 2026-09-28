"""
detector.py — Deterministic Hinglish detection (no LLM, unit-testable offline).

A question is Hinglish when it is written in Latin script and enough of its words are
Romanised Hindi. Words that are also English ("main", "to", "hi") count half, so an
English sentence is never classed as Hinglish because of them alone.
"""

import re

from hinglish.lexicon import SPELLING_VARIANTS, STRONG_MARKERS, WEAK_MARKERS

MIN_SCORE = 1.5        # weighted Hindi words needed (two strong words, or one strong + one weak)
MIN_HINDI_RATIO = 0.2  # share of words that are Hindi markers
MAX_DEVANAGARI = 0.3   # above this, llm_client.detect_language already says Hindi


def tokens(text: str) -> list[str]:
    """Lowercase Latin-script words (digits and punctuation dropped)."""
    return re.findall(r"[a-z]+", (text or "").lower())


def standard_spelling(token: str) -> str:
    return SPELLING_VARIANTS.get(token, token)


def _weight(token: str) -> float:
    """1 for an unambiguous Hindi word, 0.5 for one that is also English, else 0."""
    if token in WEAK_MARKERS:
        return 0.5
    standard = standard_spelling(token)
    if standard != token:
        return 1.0  # a Romanised-Hindi spelling variant ("mai", "hu", "nahin") is never English
    if standard in STRONG_MARKERS:
        return 1.0
    return 0.5 if standard in WEAK_MARKERS else 0.0


def detect_hinglish(text: str) -> dict:
    """Classify `text` as Hinglish or not.

    Returns {"is_hinglish", "score", "hindi_ratio", "hindi_tokens", "english_tokens"}.
    """
    letters = [ch for ch in (text or "") if ch.isalpha()]
    devanagari = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
    words = tokens(text)
    hindi = [w for w in words if _weight(w) > 0]
    score = sum(_weight(w) for w in words)
    ratio = len(hindi) / len(words) if words else 0.0
    mostly_latin = not letters or devanagari / len(letters) <= MAX_DEVANAGARI
    return {
        "is_hinglish": bool(words) and mostly_latin and score >= MIN_SCORE and ratio >= MIN_HINDI_RATIO,
        "score": round(score, 2),
        "hindi_ratio": round(ratio, 2),
        "hindi_tokens": hindi,
        "english_tokens": [w for w in words if _weight(w) == 0],
    }


def unconverted_hindi_words(text: str) -> list[str]:
    """Strong Romanised-Hindi words still written in Latin script (used by the conversion gate)."""
    return [w for w in tokens(text)
            if w not in WEAK_MARKERS and (standard_spelling(w) in STRONG_MARKERS or standard_spelling(w) != w)]
