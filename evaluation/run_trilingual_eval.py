"""
run_trilingual_eval.py — Automated English ↔ Hindi ↔ Hinglish consistency evaluation.

Usage:
    python -m evaluation.run_trilingual_eval --retrieval-only     # retrieval consistency only (cheap)
    python -m evaluation.run_trilingual_eval                      # full: retrieval + plan + answer
    python -m evaluation.run_trilingual_eval --ids T1 T3          # selected items
    python -m evaluation.run_trilingual_eval --no-legacy          # skip the "old pipeline" baseline

Full mode runs rag_pipeline.query() SEPARATELY for each language, so convergence on one
answer is measured, not assumed (the Parity Compare path forces one shared answer).

Metrics (see IMPLEMENTATION.md §9): Retrieval Consistency, Evidence Overlap, Missing Evidence,
Plan Consistency, Final Answer Similarity, Hallucination Rate, and HI/Hinglish parity.
Writes reports/trilingual_report.{md,json,csv}.
"""

import argparse
import json
import sys
from pathlib import Path

import rag_pipeline
from evaluation.metrics import answer_similarity, hallucination_rate, plan_consistency, retrieval_comparison
from evaluation.report import write_reports
from evaluation.retrieval_consistency import compare_item, embedder

ROOT = Path(__file__).parent
DATASET = ROOT / "trilingual_eval_set.json"
REPORT_DIR = ROOT.parent / "reports"


def evaluate_answers(item: dict, embed) -> dict:
    """Run each language through the full pipeline independently and compare with English."""
    traces = {"en": rag_pipeline.query(item["en"], language_override="en"),
              "hi": rag_pipeline.query(item["hi"], language_override="hi"),
              "hinglish": rag_pipeline.query(item["hinglish"], language_override="hinglish")}
    en = traces["en"]
    result = {
        "canonical_answer": en["canonical_answer"],
        "validation_passed": en["validation"]["passed"],
        "en": {"hallucination": hallucination_rate(en["final_answer"], en["retrieved_chunks"])},
    }
    for lang in ("hi", "hinglish"):
        trace = traces[lang]
        prefix = rag_pipeline.TRANSLATED_LANGUAGES[lang]
        back = trace[f"{prefix}_back_translation"]
        result[lang] = {
            "english_query": trace["english_query"],
            "same_answer_entry": rag_pipeline._store_key(trace["english_query"])
            == rag_pipeline._store_key(en["english_query"]),
            "retrieval": retrieval_comparison(en["retrieved_chunks"], trace["retrieved_chunks"]),
            "plan_consistency": plan_consistency(en["answer_plan"], trace["answer_plan"], embed),
            "answer_similarity": answer_similarity(en["final_answer"], back, embed),
            "hallucination": hallucination_rate(back, trace["retrieved_chunks"]),
            "parity_passed": trace[f"{prefix}_parity"]["parity_passed"],
            "failed_parity_checks": [c["name"] for c in trace[f"{prefix}_parity"]["checks"] if not c["passed"]],
            "final_answer": trace["final_answer"],
            "back_translation": back,
        }
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", help="Only evaluate these item ids")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answer generation")
    parser.add_argument("--no-legacy", action="store_true", help="Skip the pre-Hinglish pipeline baseline")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]
    embed = embedder()

    rows = []
    for item in items:
        print(f"Evaluating {item['id']} ...", flush=True)
        try:
            row = compare_item(item, legacy=not args.no_legacy)
            if not args.retrieval_only:
                row["answers"] = evaluate_answers(item, embed)
            row["status"] = "ok"
        except Exception as e:  # keep going; report the failure
            row = {"id": item["id"], "safety": item.get("safety", False), "status": f"error: {e}"}
        rows.append(row)

    mode = "retrieval-only" if args.retrieval_only else "full"
    paths = write_reports(rows, mode, args.out)
    print("\n" + paths["summary_text"])
    print(f"\nReports: {paths['md']}\n         {paths['json']}\n         {paths['csv']}")


if __name__ == "__main__":
    main()
