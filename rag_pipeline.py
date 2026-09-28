"""
rag_pipeline.py — End-to-end Forward-Translation RAG orchestrator.

Swastham pipeline shape:
    English:  query → retrieve → plan → canonical English answer
    Hindi:    query → translate to English → retrieve → plan → canonical English answer
              → translate to Hindi → parity check (back-translation, one retranslation on failure)
    Hinglish: query → normalise + convert to standard Hindi → (the Hindi path above)
              → translate the canonical answer to Hinglish → parity check

Also provides the "before" baseline (LLM answers directly in the user's language)
and the English/Hindi comparison used by the UI and evaluate.py.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from hinglish import detect_hinglish, normalise_hinglish, to_standard_hindi
from ingest import get_embedding_model, load_artifacts, load_index_meta, retrieve
from llm_client import (
    LLMError,
    back_translate_hinglish,
    canonicalize_query,
    contextualize_question,
    detect_language,
    generate_answer_plan,
    generate_canonical_answer,
    generate_direct_answer,
    normalise_question,
    prompts_fingerprint,
    translate_to_english,
    translate_to_hindi,
    translate_to_hinglish,
)
from validator import parity_report, validate_answer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K = int(os.getenv("TOP_K", "4"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = Path(__file__).parent / "data"
INTENT_THRESHOLD = 0.75
INTENT_MISMATCH = "Question Intent Mismatch"

# A Hindi or Hinglish answer that fails one of these parity checks is never shown on the Ask tab
BLOCKING_PARITY_CHECKS = {"numbers_match", "cautions_preserved"}

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
# Output language → key of its checked translation in a stored answer entry (also the trace prefix)
TRANSLATED_LANGUAGES = {"hi": "hindi", "hinglish": "hinglish"}

# ---------------------------------------------------------------------------
# Lazy-load index (cached after first call)
# ---------------------------------------------------------------------------

_cache = {}
_lock = threading.RLock()  # the server handles requests in parallel threads


def _get_index_and_chunks():
    """Load and cache the FAISS index + chunks, refusing an index built with another embedding model."""
    with _lock:
        if "chunks" not in _cache:
            chunks, index = load_artifacts(DATA_DIR)
            built_with = load_index_meta(DATA_DIR).get("embedding_model")
            if built_with and built_with != EMBEDDING_MODEL:
                raise RuntimeError(f"The index was built with {built_with!r} but EMBEDDING_MODEL is "
                                   f"{EMBEDDING_MODEL!r}. Re-run ingest.py or fix EMBEDDING_MODEL.")
            dim = get_embedding_model(EMBEDDING_MODEL).encode(["dimension check"]).shape[1]
            if dim != index.d:
                raise RuntimeError(f"The index has dimension {index.d} but {EMBEDDING_MODEL!r} produces "
                                   f"{dim}. Re-run ingest.py with the same embedding model.")
            _cache["chunks"] = chunks
            _cache["index"] = index
        return _cache["chunks"], _cache["index"]


def _retrieve(english_query: str) -> list[dict]:
    chunks, index = _get_index_and_chunks()
    return retrieve(english_query, chunks, index, top_k=TOP_K, model_name=EMBEDDING_MODEL)


def _hinglish_to_hindi(user_query: str) -> dict:
    """Hinglish question → standard Hindi question, converted by the LLM once per normalised wording.

    Every spelling of the same question ("kya mai … hu" / "kya main … hoon") shares one
    cache key, so it always becomes the same Hindi question and therefore the same answer.
    """
    key = normalise_hinglish(user_query)
    cached = _store_get("hinglish_queries", key) if key else None
    if cached is not None:
        return {**cached, "reused": True}
    conversion = to_standard_hindi(user_query)
    if key and conversion["attempts"]:  # attempts == 0: offline, nothing was converted
        _store_put("hinglish_queries", key, conversion)
    return {**conversion, "reused": False}


def _prepare_query(user_query: str, lang: str) -> dict:
    """User question → the English question to canonicalise, plus the steps taken (for the trace).

    English passes through; Hindi is translated; Hinglish is converted to standard Hindi
    first and then uses the SAME Hindi → English translator, so Hindi and Hinglish users
    reach the knowledge base with the same wording.
    """
    steps = {}
    text = user_query
    if lang == "hinglish":
        conversion = _hinglish_to_hindi(user_query)
        steps.update(normalised_hinglish=conversion["normalised"],
                     protected_terms=conversion["terms"],
                     standard_hindi_query=conversion["standard_hindi"],
                     hinglish_conversion_reused=conversion["reused"])
        text = conversion["standard_hindi"]
    steps["translated_query"] = translate_to_english(text) if lang in TRANSLATED_LANGUAGES else text
    return steps


def _to_english(user_query: str, lang: str) -> str:
    """Hindi or Hinglish → English translation (English passes through unchanged)."""
    return _prepare_query(user_query, lang)["translated_query"]


def resolve_language(user_query: str, language_override: str | None = None) -> tuple[str, dict | None]:
    """(language, Hinglish detection details): Latin-script Romanised Hindi is Hinglish, not English."""
    if language_override:
        return language_override, None
    lang = detect_language(user_query)
    if lang != "en":
        return lang, None
    detection = detect_hinglish(user_query)
    return ("hinglish" if detection["is_hinglish"] else "en"), detection


def question_similarity(a: str, b: str) -> float:
    """Cosine similarity between two English questions using the retrieval embedding model."""
    model = get_embedding_model(EMBEDDING_MODEL)
    emb = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


# ---------------------------------------------------------------------------
# Store: canonical questions and one answer (+ its Hindi translation) per canonical question
#
# File layout: {"fingerprint": str, "queries": {normalised question: canonical question},
#               "answers": {canonical key: entry}}
# The fingerprint covers the model, prompts, glossaries, TOP_K, embedding model and chunks,
# so a change to any of them discards the stored answers automatically.
# ---------------------------------------------------------------------------

ANSWER_STORE_PATH = DATA_DIR / "answer_store.json"


def _store_key(canonical_query: str) -> str:
    return normalise_question(canonical_query)


def _fingerprint() -> str:
    chunks_path = DATA_DIR / "chunks.json"
    chunks_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()[:16] if chunks_path.exists() else ""
    parts = [prompts_fingerprint(), EMBEDDING_MODEL, str(TOP_K), chunks_hash]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _empty_store(fingerprint: str) -> dict:
    return {"fingerprint": fingerprint, "queries": {}, "hinglish_queries": {}, "answers": {}}


def _load_store() -> dict:
    with _lock:
        if "store" not in _cache:
            fingerprint = _fingerprint()
            try:
                store = json.loads(ANSWER_STORE_PATH.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                store = {}
            if not isinstance(store, dict) or store.get("fingerprint") != fingerprint:
                store = _empty_store(fingerprint)  # stale or old-format store: start fresh
            _cache["store"] = store
        return _cache["store"]


def _store_get(section: str, key: str):
    with _lock:
        return _load_store().get(section, {}).get(key)


def _store_put(section: str, key: str, value):
    """Save one item; the file is written atomically so a crash never leaves it half-written."""
    with _lock:
        store = _load_store()
        store.setdefault(section, {})[key] = value
        tmp = ANSWER_STORE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, ANSWER_STORE_PATH)


def _canonical_question(english_query: str) -> str:
    """Canonical form of an English question, canonicalised by the LLM once per wording."""
    cleaned = normalise_question(english_query)
    if not cleaned:
        return ""
    canonical = _store_get("queries", cleaned)
    if canonical is None:
        canonical = canonicalize_query(english_query)
        _store_put("queries", cleaned, canonical)
    return canonical


def _build_canonical_answer(english_query: str) -> dict:
    """Retrieve → Answer Plan → canonical English answer → validate (+1 retry)."""
    retrieved = _retrieve(english_query)
    plan = generate_answer_plan(english_query, retrieved)
    canonical = generate_canonical_answer(english_query, plan, retrieved)
    validation = validate_answer(canonical, plan, retrieved, question=english_query)
    attempts = 1
    if not validation["passed"] and validation["feedback"]:
        canonical = generate_canonical_answer(english_query, plan, retrieved,
                                              feedback=validation["feedback"])
        validation = validate_answer(canonical, plan, retrieved, question=english_query)
        attempts = 2
    return {
        "retrieved_chunks": retrieved,
        "answer_plan": plan,
        "canonical_answer": canonical,
        "validation": validation,
        "attempts": attempts,
    }


def _answer_for(canonical_question: str) -> tuple[dict, bool]:
    """Return (entry, reused) for a canonical question.

    Only answers that passed validation are stored; a failed one is returned for display
    but rebuilt next time instead of being reused forever.
    """
    key = _store_key(canonical_question)
    entry = _store_get("answers", key)
    if entry is not None:
        return entry, True
    entry = _build_canonical_answer(canonical_question)
    if entry["validation"]["passed"]:
        _store_put("answers", key, entry)
    return entry, False


# ---------------------------------------------------------------------------
# Hindi / Hinglish: translation of the canonical answer, checked by back-translation
# ---------------------------------------------------------------------------

def _parity_feedback(parity: dict) -> str:
    return "\n".join(f"- {c['name']}: {c['details']}" for c in parity["checks"] if not c["passed"])


def _translators(lang: str):
    """(forward, back) translators for an output language, looked up at call time (patchable in tests)."""
    if lang == "hinglish":
        return translate_to_hinglish, back_translate_hinglish
    return translate_to_hindi, translate_to_english


def _translate_checked(english_answer: str, lang: str = "hi") -> dict:
    """Translate the canonical answer, back-translate it and run the parity report.

    On a parity failure the SAME English answer is translated once more with the failed
    checks as feedback. Nothing is generated here; the source text never changes.
    """
    forward, back_translate = _translators(lang)
    feedback = None
    for attempt in (1, 2):
        translated = forward(english_answer, feedback=feedback)
        back = back_translate(translated)
        parity = parity_report(english_answer, translated, back)
        if parity["parity_passed"]:
            break
        feedback = _parity_feedback(parity)
    return {"answer": translated, "back_translation": back, "parity": parity, "attempts": attempt}


def _translation_for(canonical_question: str, entry: dict, lang: str) -> dict:
    """The Hindi or Hinglish version of a stored answer: reused when a checked translation is stored."""
    field = TRANSLATED_LANGUAGES[lang]
    if entry.get(field):
        return {**entry[field], "reused": True}
    translation = _translate_checked(entry["canonical_answer"], lang)
    key = _store_key(canonical_question)
    with _lock:  # entry is shared with the store; change it only while no one is writing the file
        if translation["parity"]["parity_passed"] and _store_get("answers", key) is entry:
            entry[field] = translation
            _store_put("answers", key, entry)
    return {**translation, "reused": False}


def _hindi_for(canonical_question: str, entry: dict) -> dict:
    return _translation_for(canonical_question, entry, "hi")


def _hinglish_for(canonical_question: str, entry: dict) -> dict:
    return _translation_for(canonical_question, entry, "hinglish")


def _trace_from(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in TRANSLATED_LANGUAGES.values()}


# ---------------------------------------------------------------------------
# Main query function — the single answer path
# ---------------------------------------------------------------------------

def query(user_query: str, language_override: str | None = None,
          history: list[dict] | None = None, reply_language: str | None = None) -> dict:
    """Run the full forward-translation RAG pipeline and return an execution trace.

    history (voice agent): earlier turns as {"question": English question, "answer": English answer};
    a follow-up is rewritten into a standalone question before canonicalisation.
    reply_language: answer in this language instead of the question's (e.g. a Hindi transcript
    of Hinglish speech answered in Hinglish). The answer is still a checked translation.
    """
    # Step 1: Detect language (Latin-script Romanised Hindi is Hinglish)
    lang, detection = resolve_language(user_query, language_override)
    trace = {"user_query": user_query, "detected_language": lang}
    if detection is not None:
        trace["hinglish_detection"] = detection

    # Step 2: English query for retrieval (Hinglish → standard Hindi → English, then canonicalised)
    trace.update(_prepare_query(user_query, lang))
    question = trace["translated_query"]
    if history:
        question = trace["standalone_query"] = contextualize_question(question, history)
    english_query = _canonical_question(question)
    trace["english_query"] = english_query

    # Steps 3–6: reuse the stored canonical answer for this canonical question if there is one,
    # so an English and a Hindi user asking the same thing get the SAME chunks and SAME answer.
    entry, reused = _answer_for(english_query)
    trace["answer_reused"] = reused
    trace.update(_trace_from(entry))

    # Step 7: Output language — Hindi/Hinglish is a translation of the canonical answer, never a new answer
    lang = reply_language or lang
    trace["output_language"] = lang
    if lang not in TRANSLATED_LANGUAGES:
        trace["final_answer"] = entry["canonical_answer"]
        return trace

    translation = _hindi_for(english_query, entry) if lang == "hi" else _hinglish_for(english_query, entry)
    failed = {c["name"] for c in translation["parity"]["checks"] if not c["passed"]}
    if failed & BLOCKING_PARITY_CHECKS:
        # Fail closed: never show a translated answer that lost a number or a safety caution
        raise LLMError(f"The {LANGUAGE_NAMES[lang]} translation did not pass the parity check "
                       f"({', '.join(sorted(failed))}) after a retry, so it is not shown.")
    prefix = TRANSLATED_LANGUAGES[lang]
    trace["final_answer"] = translation["answer"]
    trace[f"{prefix}_back_translation"] = translation["back_translation"]
    trace[f"{prefix}_parity"] = translation["parity"]
    trace["translation_reused"] = translation["reused"]
    return trace


def query_baseline(user_query: str, language_override: str | None = None) -> dict:
    """The "before" path: same retrieval, but the LLM answers directly in the user's language."""
    lang, _ = resolve_language(user_query, language_override)
    # Baseline = current behaviour: plain translation, no canonicalisation
    english_query = _to_english(user_query, lang)
    retrieved = _retrieve(english_query)
    answer = generate_direct_answer(user_query, retrieved, lang)
    return {
        "user_query": user_query,
        "detected_language": lang,
        "english_query": english_query,
        "retrieved_chunks": retrieved,
        "final_answer": answer,
        "output_language": lang,
    }


# ---------------------------------------------------------------------------
# English / Hindi comparison
# ---------------------------------------------------------------------------

def _chunk_ids(chunks: list[dict]) -> list[int]:
    return [c["id"] for c in chunks]


def compare_multilingual_questions(english_query: str, hindi_query: str,
                                   include_baseline: bool = False,
                                   hinglish_query: str | None = None) -> dict:
    """Use one canonical English answer and translate it for Hindi (and Hinglish when given).

    This enforces the required architecture:
      user question -> canonical English intent -> one RAG pass -> one answer -> translations
    """
    if not english_query or not hindi_query:
        raise ValueError("Both english_query and hindi_query are required")

    hindi_as_english = translate_to_english(hindi_query)
    canonical_english = _canonical_question(english_query)
    canonical_hindi = _canonical_question(hindi_as_english)
    same_canonical = _store_key(canonical_english) == _store_key(canonical_hindi)
    similarity = question_similarity(english_query, hindi_as_english)
    if not same_canonical and similarity < INTENT_THRESHOLD:
        return {
            "status": INTENT_MISMATCH,
            "query_similarity": round(similarity, 2),
            "hindi_translated": hindi_as_english,
            "reason": "The English and Hindi questions do not express the same intent.",
        }

    hinglish_steps = None
    if hinglish_query:
        hinglish_steps = _prepare_query(hinglish_query, "hinglish")
        hinglish_as_english = hinglish_steps["translated_query"]
        canonical_hinglish = _canonical_question(hinglish_as_english)
        same_canonical_hg = _store_key(canonical_english) == _store_key(canonical_hinglish)
        similarity_hg = question_similarity(english_query, hinglish_as_english)
        if not same_canonical_hg and similarity_hg < INTENT_THRESHOLD:
            return {
                "status": INTENT_MISMATCH,
                "query_similarity": round(similarity_hg, 2),
                "hinglish_translated": hinglish_as_english,
                "reason": "The English and Hinglish questions do not express the same intent.",
            }

    # IMPORTANT: one shared canonical answer for both languages.
    entry, reused = _answer_for(canonical_english)
    canonical_answer = entry["canonical_answer"]
    hindi = _hindi_for(canonical_english, entry)

    # Would a Hindi user asking on the Ask tab reach the same evidence? Their question is
    # retrieved on its own canonical form (a check only; the answer is not regenerated).
    en_chunks = entry["retrieved_chunks"]
    hi_chunks = en_chunks if same_canonical else _retrieve(canonical_hindi)

    en_trace = _trace_from(entry)
    en_trace.update(user_query=english_query, detected_language="en", english_query=canonical_english,
                    answer_reused=reused, final_answer=canonical_answer, output_language="en")

    hi_trace = _trace_from(entry)
    hi_trace.update(user_query=hindi_query, detected_language="hi", translated_query=hindi_as_english,
                    english_query=canonical_hindi, answer_reused=reused,
                    final_answer=hindi["answer"], output_language="hi",
                    translated_from=canonical_answer, source_is_canonical=True,
                    translation_attempts=hindi["attempts"], translation_reused=hindi["reused"],
                    own_retrieval_chunk_ids=_chunk_ids(hi_chunks))

    result = {
        "status": "ok",
        "query_similarity": round(similarity, 2),
        "same_canonical_question": same_canonical,
        "english": en_trace,
        "hindi": hi_trace,
        "hindi_back_translation": hindi["back_translation"],
        "parity": parity_report(canonical_answer, hindi["answer"], hindi["back_translation"],
                                _chunk_ids(en_chunks), _chunk_ids(hi_chunks)),
    }

    if hinglish_steps is not None:
        hinglish = _hinglish_for(canonical_english, entry)
        hg_chunks = en_chunks if same_canonical_hg else _retrieve(canonical_hinglish)
        hg_trace = _trace_from(entry)
        hg_trace.update(hinglish_steps)
        hg_trace.update(user_query=hinglish_query, detected_language="hinglish",
                        english_query=canonical_hinglish, answer_reused=reused,
                        final_answer=hinglish["answer"], output_language="hinglish",
                        translated_from=canonical_answer, source_is_canonical=True,
                        translation_attempts=hinglish["attempts"], translation_reused=hinglish["reused"],
                        own_retrieval_chunk_ids=_chunk_ids(hg_chunks))
        result.update(
            hinglish=hg_trace,
            hinglish_query_similarity=round(similarity_hg, 2),
            same_canonical_question_hinglish=same_canonical_hg,
            hinglish_back_translation=hinglish["back_translation"],
            hinglish_parity=parity_report(canonical_answer, hinglish["answer"], hinglish["back_translation"],
                                          _chunk_ids(en_chunks), _chunk_ids(hg_chunks)),
        )

    if include_baseline:
        base_en = query_baseline(english_query, language_override="en")
        base_hi = query_baseline(hindi_query, language_override="hi")
        base_back = translate_to_english(base_hi["final_answer"])
        result["baseline"] = {
            "english": base_en,
            "hindi": base_hi,
            "hindi_back_translation": base_back,
            "parity": parity_report(
                base_en["final_answer"], base_hi["final_answer"], base_back,
                _chunk_ids(base_en["retrieved_chunks"]), _chunk_ids(base_hi["retrieved_chunks"]),
            ),
        }
        if hinglish_steps is not None:
            base_hg = query_baseline(hinglish_query, language_override="hinglish")
            base_hg_back = back_translate_hinglish(base_hg["final_answer"])
            result["baseline"].update(
                hinglish=base_hg,
                hinglish_back_translation=base_hg_back,
                hinglish_parity=parity_report(
                    base_en["final_answer"], base_hg["final_answer"], base_hg_back,
                    _chunk_ids(base_en["retrieved_chunks"]), _chunk_ids(base_hg["retrieved_chunks"]),
                ),
            )

    return result


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = " ".join(sys.argv[1:]) or input("Query > ").strip()
    if user_query:
        result = query(user_query)
        print(f"Detected Language : {result['detected_language'].upper()}")
        if "standard_hindi_query" in result:
            print(f"Normalised        : {result['normalised_hinglish']}")
            print(f"Standard Hindi    : {result['standard_hindi_query']}")
        print(f"English Query     : {result['english_query']}")
        print(f"Retrieved Chunks  : {[c['id'] for c in result['retrieved_chunks']]}")
        print(f"Validation        : passed={result['validation']['passed']} "
              f"score={result['validation']['score']} attempts={result['attempts']}")
        if "hindi_parity" in result:
            print(f"Hindi parity      : passed={result['hindi_parity']['parity_passed']}")
        if "hinglish_parity" in result:
            print(f"Hinglish parity   : passed={result['hinglish_parity']['parity_passed']}")
        print("\nANSWER:\n")
        print(result["final_answer"])
