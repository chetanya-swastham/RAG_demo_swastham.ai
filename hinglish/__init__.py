"""
hinglish — Hinglish (Romanised Hindi mixed with English) support for the parity pipeline.

Hinglish is not a new reasoning path. A Hinglish question is normalised, converted to
standard Hindi and then follows the existing Hindi path to the one canonical English
answer; the Hinglish answer is a checked translation of that same answer.

Import order matters: lexicon and detector are pure; normalizer imports llm_client, which
itself imports hinglish.lexicon.
"""

from hinglish.detector import detect_hinglish
from hinglish.lexicon import PROTECTED_TERMS, find_protected_terms
from hinglish.normalizer import mask_terms, normalise_hinglish, restore_terms, to_standard_hindi

__all__ = ["detect_hinglish", "PROTECTED_TERMS", "find_protected_terms", "mask_terms",
           "normalise_hinglish", "restore_terms", "to_standard_hindi"]
