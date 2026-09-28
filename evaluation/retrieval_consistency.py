"""
retrieval_consistency.py — Do English, Hindi and Hinglish versions of a question reach the same evidence?

For one test item it runs each language through the pipeline's own query preparation
(rag_pipeline._prepare_query → _canonical_question → _retrieve), with no answer generation,
and compares every retrieval with the English one.

Two "normaliser off" baselines show what the Hinglish layer buys:
  - hinglish_raw:    the raw Hinglish string embedded directly (no LLM at all)
  - hinglish_legacy: the pipeline before Hinglish support (Hinglish treated as English,
                     canonicalised as-is, then retrieved)
"""

import rag_pipeline
from evaluation.metrics import chunk_overlap, retrieval_comparison
from hinglish import detect_hinglish


def embedder():
    """list[str] → normalised vectors, using the retrieval embedding model."""
    from ingest import get_embedding_model

    model = get_embedding_model(rag_pipeline.EMBEDDING_MODEL)
    return lambda texts: model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True).tolist()


def _slim(chunks: list[dict]) -> list[dict]:
    return [{"id": c["id"], "heading": c.get("heading", ""), "score": round(c.get("score", 0.0), 4)}
            for c in chunks]


def retrieve_for(user_query: str, lang: str) -> dict:
    """The pipeline's retrieval for one question: steps, canonical English question and chunks."""
    steps = rag_pipeline._prepare_query(user_query, lang)
    canonical = rag_pipeline._canonical_question(steps["translated_query"])
    return {**steps, "english_query": canonical, "chunks": rag_pipeline._retrieve(canonical)}


def _same(a: str, b: str) -> bool:
    return rag_pipeline._store_key(a) == rag_pipeline._store_key(b)


def compare_item(item: dict, legacy: bool = True) -> dict:
    """Retrieval consistency of one {en, hi, hinglish, variants} item against English."""
    runs = {"en": retrieve_for(item["en"], "en"),
            "hi": retrieve_for(item["hi"], "hi"),
            "hinglish": retrieve_for(item["hinglish"], "hinglish")}
    en = runs["en"]
    row = {
        "id": item["id"],
        "safety": item.get("safety", False),
        "detection": {"hinglish_detected": detect_hinglish(item["hinglish"])["is_hinglish"],
                      "english_misdetected": detect_hinglish(item["en"])["is_hinglish"]},
        "canonical": {lang: run["english_query"] for lang, run in runs.items()},
        "standard_hindi_query": runs["hinglish"].get("standard_hindi_query"),
        "chunks": {lang: _slim(run["chunks"]) for lang, run in runs.items()},
        "retrieval": {},
    }
    for lang in ("hi", "hinglish"):
        comparison = retrieval_comparison(en["chunks"], runs[lang]["chunks"])
        comparison["same_canonical"] = _same(en["english_query"], runs[lang]["english_query"])
        comparison["query_similarity"] = round(
            rag_pipeline.question_similarity(en["english_query"], runs[lang]["english_query"]), 4)
        row["retrieval"][lang] = comparison

    # Normaliser off: what Hinglish retrieval looked like without this layer
    raw = rag_pipeline._retrieve(item["hinglish"])
    row["retrieval"]["hinglish_raw"] = retrieval_comparison(en["chunks"], raw)
    if legacy:
        legacy_canonical = rag_pipeline._canonical_question(item["hinglish"])
        legacy_chunks = rag_pipeline._retrieve(legacy_canonical)
        row["retrieval"]["hinglish_legacy"] = {**retrieval_comparison(en["chunks"], legacy_chunks),
                                               "english_query": legacy_canonical,
                                               "same_canonical": _same(en["english_query"], legacy_canonical)}

    # Spelling variants of the Hinglish question must all reach the same canonical question
    en_ids = [c["id"] for c in en["chunks"]]
    row["variants"] = []
    for text in item.get("variants", []):
        run = retrieve_for(text, "hinglish")
        row["variants"].append({
            "text": text,
            "normalised": run.get("normalised_hinglish"),
            "english_query": run["english_query"],
            "same_canonical": _same(en["english_query"], run["english_query"]),
            "overlap": round(chunk_overlap(en_ids, [c["id"] for c in run["chunks"]]), 2),
        })
    return row
