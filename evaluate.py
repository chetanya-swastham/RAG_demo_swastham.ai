"""
evaluate.py — English/Hindi parity evaluation: canonical pipeline vs baseline.

For every EN/HI pair in eval_set.json, runs:
  - NEW:      the canonical forward-translation pipeline (rag_pipeline.query)
  - BASELINE: the LLM answering directly in each language (rag_pipeline.query_baseline)
and scores both with the same parity checks (validator.parity_report).

Usage:
    python evaluate.py                 # all pairs
    python evaluate.py --ids Q3 Q14    # selected pairs
Writes eval_results.csv and prints a summary table.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from rag_pipeline import INTENT_MISMATCH, compare_multilingual_questions

ROOT = Path(__file__).parent
CHECKS = ["retrieval_overlap", "numbers_match", "cautions_preserved", "recommendation_match",
          "content_coverage"]


def _check(parity: dict, name: str) -> dict:
    return next((c for c in parity["checks"] if c["name"] == name), {"passed": False})


def evaluate_pair(item: dict) -> dict:
    result = compare_multilingual_questions(item["en"], item["hi"], include_baseline=True)
    row = {"id": item["id"], "safety": item["safety"], "status": result["status"],
           "query_similarity": result.get("query_similarity")}
    if result["status"] == INTENT_MISMATCH:
        return row

    for label, parity in (("new", result["parity"]), ("base", result["baseline"]["parity"])):
        row[f"{label}_parity"] = parity["parity_passed"]
        for name in CHECKS:
            row[f"{label}_{name}"] = _check(parity, name)["passed"]
        row[f"{label}_coverage"] = _check(parity, "content_coverage").get("value")
    row["new_validation"] = result["english"]["validation"]["passed"]
    row["new_attempts"] = result["english"]["attempts"]
    return row


def _mark(value) -> str:
    return "✓" if value else "✗"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="Only evaluate these pair ids")
    args = parser.parse_args()

    items = json.loads((ROOT / "eval_set.json").read_text(encoding="utf-8"))
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]

    rows = []
    for item in items:
        print(f"Evaluating {item['id']} ...", flush=True)
        try:
            rows.append(evaluate_pair(item))
        except Exception as e:  # keep going; report the failure in the table
            rows.append({"id": item["id"], "safety": item["safety"], "status": f"error: {e}"})

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "id", k))
    with open(ROOT / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print(f"{'ID':<5} {'Safety':<7} {'NEW parity':<11} {'BASE parity':<12} "
          f"{'NEW cautions':<13} {'BASE cautions':<14} coverage new/base")
    print("-" * 78)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['id']:<5} {r['status']}")
            continue
        print(f"{r['id']:<5} {_mark(r['safety']):<7} {_mark(r['new_parity']):<11} "
              f"{_mark(r['base_parity']):<12} {_mark(r['new_cautions_preserved']):<13} "
              f"{_mark(r['base_cautions_preserved']):<14} {r['new_coverage']} / {r['base_coverage']}")

    ok = [r for r in rows if r["status"] == "ok"]
    safety = [r for r in ok if r["safety"]]
    count = lambda rs, key: sum(1 for r in rs if r.get(key))
    print("=" * 78)
    print(f"Parity passed         : NEW {count(ok, 'new_parity')}/{len(ok)}"
          f"   vs   BASELINE {count(ok, 'base_parity')}/{len(ok)}")
    print(f"Numbers preserved     : NEW {count(ok, 'new_numbers_match')}/{len(ok)}"
          f"   vs   BASELINE {count(ok, 'base_numbers_match')}/{len(ok)}")
    print(f"Recommendations kept  : NEW {count(ok, 'new_recommendation_match')}/{len(ok)}"
          f"   vs   BASELINE {count(ok, 'base_recommendation_match')}/{len(ok)}")
    print(f"Safety cautions kept  : NEW {count(safety, 'new_cautions_preserved')}/{len(safety)}"
          f"   vs   BASELINE {count(safety, 'base_cautions_preserved')}/{len(safety)}")
    print(f"Validation passed     : NEW {count(ok, 'new_validation')}/{len(ok)}")
    if len(ok) < len(rows):
        print(f"Errors / mismatches   : {len(rows) - len(ok)}")
    print("Details written to eval_results.csv")


if __name__ == "__main__":
    main()
