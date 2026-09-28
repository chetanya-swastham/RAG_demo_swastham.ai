"""
normalizer.py — Hinglish question → standard Hindi (Devanagari) question.

    "kya mai fruit kha sakta hu"
      → normalise_hinglish   "kya main fruit kha sakta hoon"          (deterministic cache key)
      → mask_terms           protected medical terms → [[TA]], [[TB]] …
      → LLM conversion       Roman Hindi words → Devanagari, English words kept
      → gates                every placeholder kept, numbers unchanged, no Roman Hindi left
      → restore_terms        "क्या मैं fruit खा सकता हूँ"

The Hindi question then goes through the SAME Hindi → English translator as a Devanagari
user's question, so Hindi and Hinglish users converge on one canonical English question.
"""

import re

import llm_client  # module import (not names): llm_client imports hinglish.lexicon while loading
from hinglish.detector import standard_spelling, unconverted_hindi_words
from hinglish.lexicon import term_patterns
from validator import extract_numbers

_PLACEHOLDER = re.compile(r"\[\[T([A-Z])\]\]")
_TERM_SENTINEL = re.compile(r"\x00(\d+)\x00")


def _placeholder(i: int) -> str:
    # Letters, not digits, so a placeholder never counts as a number in the number gate
    return f"[[T{chr(ord('A') + i)}]]"


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace protected terms with sentinels; returns (text, standard spellings in order)."""
    found: list[str] = []

    for term, pattern in term_patterns():
        def repl(_match, term=term):
            found.append(term)
            return f" \x00{len(found) - 1}\x00 "
        text = pattern.sub(repl, text)
    return text, found


def normalise_hinglish(text: str) -> str:
    """Deterministic clean-up: lowercase, one spelling per Hindi word, punctuation dropped.

    Protected medical terms keep their standard spelling ("hba1c" → "HbA1c"). Two spellings
    of the same question ("kya mai fruit kha sakta hu" / "Kya main fruit kha sakta hoon?")
    give the same string, which is the cache key for the LLM conversion.
    """
    protected, terms = _protect((text or "").replace("’", "'"))
    parts = []
    for token in re.findall(r"\x00\d+\x00|\d+(?:\.\d+)?|[a-z]+|[ऀ-ॿ]+", protected.lower()):
        sentinel = _TERM_SENTINEL.fullmatch(token)
        if sentinel:
            parts.append(terms[int(sentinel.group(1))])
        elif token.isalpha() and token.isascii():
            parts.append(standard_spelling(token))
        else:
            parts.append(token)
    return " ".join(parts)


def mask_terms(text: str) -> tuple[str, dict[str, str]]:
    """Replace each protected term with a placeholder → (masked text, {placeholder: term})."""
    protected, terms = _protect(text)
    mapping: dict[str, str] = {}
    by_term: dict[str, str] = {}

    def repl(match):
        term = terms[int(match.group(1))]
        if term not in by_term:
            by_term[term] = _placeholder(len(by_term))
            mapping[by_term[term]] = term
        return by_term[term]

    masked = _TERM_SENTINEL.sub(repl, protected)
    return " ".join(masked.split()), mapping


def restore_terms(text: str, mapping: dict[str, str]) -> str:
    for placeholder, term in mapping.items():
        text = text.replace(placeholder, term)
    return text


def conversion_problems(masked_input: str, output: str, mapping: dict[str, str]) -> list[str]:
    """What a Roman→Devanagari conversion visibly got wrong (empty list = passed)."""
    problems = []
    for placeholder in mapping:
        count = output.count(placeholder)
        if count != 1 and masked_input.count(placeholder) == 1:
            problems.append(f"Keep the placeholder {placeholder} exactly once, unchanged (found {count}).")
    unknown = {m.group(0) for m in _PLACEHOLDER.finditer(output)} - set(mapping)
    if unknown:
        problems.append(f"Do not invent placeholders: {sorted(unknown)}.")
    plain_in, plain_out = _PLACEHOLDER.sub(" ", masked_input), _PLACEHOLDER.sub(" ", output)
    if extract_numbers(plain_in) != extract_numbers(plain_out):
        problems.append(f"Numbers must stay exactly the same: {sorted(extract_numbers(plain_in))}.")
    left = unconverted_hindi_words(plain_out)
    if left:
        problems.append(f"These Hindi words are still in Roman script; write them in Devanagari: {left}.")
    if unconverted_hindi_words(plain_in) and not re.search(r"[ऀ-ॿ]", plain_out):
        problems.append("The output has no Devanagari. Write the Hindi words in Devanagari script.")
    return problems


def to_standard_hindi(user_query: str) -> dict:
    """Convert a Hinglish question to standard Hindi, keeping medical terms verbatim.

    Returns {"normalised", "masked", "terms", "standard_hindi", "attempts"}.
    Without an API key the normalised Roman text is returned unchanged (like
    llm_client.canonicalize_query), so offline code paths never crash.
    Raises LLMError if the conversion fails its gates twice.
    """
    normalised = normalise_hinglish(user_query)
    masked, mapping = mask_terms(normalised)
    result = {"normalised": normalised, "masked": masked, "terms": sorted(mapping.values()),
              "standard_hindi": normalised, "attempts": 0}
    if not normalised or not llm_client.is_api_key_valid():
        return result

    feedback = None
    for attempt in (1, 2):
        output = llm_client.convert_roman_to_devanagari(masked, feedback=feedback)
        problems = conversion_problems(masked, output, mapping)
        if not problems:
            result.update(standard_hindi=restore_terms(output, mapping), attempts=attempt)
            return result
        feedback = "\n".join(problems)
    raise llm_client.LLMError("Hinglish → Hindi conversion failed its checks: " + " ".join(problems))
