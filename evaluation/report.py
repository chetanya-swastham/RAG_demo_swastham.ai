"""
report.py — Summarise trilingual evaluation rows into Markdown, JSON and CSV reports.

summarise() is pure (unit-testable); write_reports() writes the files.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from evaluation.metrics import THRESHOLDS

LANGS = {"hi": "Hindi", "hinglish": "Hinglish"}
RETRIEVAL_VARIANTS = {"hi": "Hindi", "hinglish": "Hinglish (normalised)",
                      "hinglish_legacy": "Hinglish, old pipeline", "hinglish_raw": "Hinglish, raw embedding"}


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _mark(ok) -> str:
    return "—" if ok is None else ("✓" if ok else "✗")


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def summarise(rows: list[dict]) -> dict:
    """Aggregate metrics over the successful rows."""
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "items": len(rows),
        "evaluated": len(ok),
        "errors": [{"id": r["id"], "status": r["status"]} for r in rows if r.get("status") != "ok"],
        "detection": {
            "hinglish_detected": sum(r["detection"]["hinglish_detected"] for r in ok),
            "english_misdetected": sum(r["detection"]["english_misdetected"] for r in ok),
        },
        "retrieval": {},
        "variants": {},
        "answers": {},
    }

    for key in RETRIEVAL_VARIANTS:
        comps = [r["retrieval"][key] for r in ok if key in r.get("retrieval", {})]
        if not comps:
            continue
        summary["retrieval"][key] = {
            "mean_overlap": _mean(c["overlap"] for c in comps),
            "mean_rank_overlap": _mean(c["rank_overlap"] for c in comps),
            "passed": sum(c["overlap"] >= THRESHOLDS["retrieval_overlap"] for c in comps),
            "total_missing_evidence": sum(c["missing_count"] for c in comps),
            "same_canonical": sum(bool(c.get("same_canonical")) for c in comps),
            "mean_query_similarity": _mean(c.get("query_similarity") for c in comps),
            "mean_score_drift": _mean(c.get("score_drift") for c in comps),
            "n": len(comps),
        }

    variants = [v for r in ok for v in r.get("variants", [])]
    summary["variants"] = {"total": len(variants),
                           "same_canonical": sum(v["same_canonical"] for v in variants),
                           "mean_overlap": _mean(v["overlap"] for v in variants)}

    answered = [r for r in ok if "answers" in r]
    if answered:
        summary["answers"]["en_hallucination"] = _mean(r["answers"]["en"]["hallucination"]["rate"] for r in answered)
        for lang in LANGS:
            items = [r["answers"][lang] for r in answered]
            summary["answers"][lang] = {
                "n": len(items),
                "same_answer_entry": sum(i["same_answer_entry"] for i in items),
                "mean_overlap": _mean(i["retrieval"]["overlap"] for i in items),
                "total_missing_evidence": sum(i["retrieval"]["missing_count"] for i in items),
                "mean_plan_consistency": _mean(i["plan_consistency"] for i in items),
                "plan_passed": sum(i["plan_consistency"] >= THRESHOLDS["plan_consistency"] for i in items),
                "mean_answer_similarity": _mean(i["answer_similarity"]["value"] for i in items),
                "answer_passed": sum(i["answer_similarity"]["value"] >= THRESHOLDS["answer_similarity"]
                                     for i in items),
                "mean_hallucination": _mean(i["hallucination"]["rate"] for i in items),
                "hallucination_passed": sum(i["hallucination"]["rate"] <= THRESHOLDS["hallucination_rate"]
                                            for i in items),
                "parity_passed": sum(i["parity_passed"] for i in items),
            }
    return summary


def _summary_lines(summary: dict, mode: str) -> list[str]:
    n = summary["evaluated"]
    lines = [f"Mode: {mode}   Items: {n}/{summary['items']} evaluated",
             f"Hinglish detected: {summary['detection']['hinglish_detected']}/{n}   "
             f"English misdetected as Hinglish: {summary['detection']['english_misdetected']}/{n}"]
    for key, label in RETRIEVAL_VARIANTS.items():
        s = summary["retrieval"].get(key)
        if s:
            lines.append(f"Retrieval {label:<24}: overlap {_fmt(s['mean_overlap'])}, "
                         f"passed {s['passed']}/{s['n']}, missing evidence {s['total_missing_evidence']}, "
                         f"same canonical {s['same_canonical']}/{s['n']}")
    v = summary["variants"]
    if v["total"]:
        lines.append(f"Spelling variants → same canonical question: {v['same_canonical']}/{v['total']}")
    for lang, label in LANGS.items():
        a = summary["answers"].get(lang)
        if a:
            lines.append(f"{label:<9} plan {_fmt(a['mean_plan_consistency'])} ({a['plan_passed']}/{a['n']}), "
                         f"answer sim {_fmt(a['mean_answer_similarity'])} ({a['answer_passed']}/{a['n']}), "
                         f"hallucination {_fmt(a['mean_hallucination'])}, parity {a['parity_passed']}/{a['n']}, "
                         f"same answer {a['same_answer_entry']}/{a['n']}")
    if summary["errors"]:
        lines.append(f"Errors: {len(summary['errors'])}")
    return lines


def _markdown(rows: list[dict], summary: dict, mode: str) -> str:
    t = THRESHOLDS
    md = ["# Trilingual consistency report (English ↔ Hindi ↔ Hinglish)",
          "", f"Generated {datetime.now():%Y-%m-%d %H:%M} · mode **{mode}** · "
              f"{summary['evaluated']}/{summary['items']} items evaluated", "",
          "## Thresholds", "",
          f"Evidence overlap ≥ {t['retrieval_overlap']} · missing evidence = {t['missing_evidence']} · "
          f"plan consistency ≥ {t['plan_consistency']} · answer similarity ≥ {t['answer_similarity']} · "
          f"hallucination rate ≤ {t['hallucination_rate']}", "",
          "## Summary", "", "```", *_summary_lines(summary, mode), "```", "",
          "## Retrieval: normaliser on vs off", "",
          "| Retrieval of the Hinglish question | Mean overlap vs English | Passed | Missing evidence | Same canonical |",
          "|---|---|---|---|---|"]
    for key, label in RETRIEVAL_VARIANTS.items():
        s = summary["retrieval"].get(key)
        if s:
            md.append(f"| {label} | {_fmt(s['mean_overlap'])} | {s['passed']}/{s['n']} | "
                      f"{s['total_missing_evidence']} | {s['same_canonical']}/{s['n']} |")

    md += ["", "## Per item: retrieval", "",
           "| ID | Safety | HI overlap | HG overlap | HG missing | HG raw overlap | Same canonical HI/HG | Variants same |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") != "ok":
            md.append(f"| {r['id']} | {_mark(r.get('safety'))} | {r['status']} ||||||")
            continue
        ret = r["retrieval"]
        variants = r.get("variants", [])
        md.append(f"| {r['id']} | {_mark(r['safety'])} | {_fmt(ret['hi']['overlap'])} | "
                  f"{_fmt(ret['hinglish']['overlap'])} | {ret['hinglish']['missing_count']} | "
                  f"{_fmt(ret['hinglish_raw']['overlap'])} | "
                  f"{_mark(ret['hi']['same_canonical'])}/{_mark(ret['hinglish']['same_canonical'])} | "
                  f"{sum(v['same_canonical'] for v in variants)}/{len(variants)} |")

    answered = [r for r in rows if r.get("status") == "ok" and "answers" in r]
    if answered:
        md += ["", "## Per item: plan and answer", "",
               "| ID | Lang | Same answer | Plan | Answer sim | Hallucination | Parity | Failed checks |",
               "|---|---|---|---|---|---|---|---|"]
        for r in answered:
            for lang, label in LANGS.items():
                a = r["answers"][lang]
                md.append(f"| {r['id']} | {label} | {_mark(a['same_answer_entry'])} | {_fmt(a['plan_consistency'])} | "
                          f"{_fmt(a['answer_similarity']['value'])} | {_fmt(a['hallucination']['rate'])} | "
                          f"{_mark(a['parity_passed'])} | {', '.join(a['failed_parity_checks']) or '—'} |")

    failures = [r for r in rows if r.get("status") == "ok" and
                (not r["retrieval"]["hinglish"]["same_canonical"] or not r["retrieval"]["hi"]["same_canonical"])]
    if failures:
        md += ["", "## Canonical-question mismatches", "",
               "Questions that did not converge on one canonical question (each one is a potential answer drift):", ""]
        for r in failures:
            c = r["canonical"]
            md += [f"- **{r['id']}**", f"  - EN: {c['en']}", f"  - HI: {c['hi']}", f"  - HG: {c['hinglish']}"]
    if summary["errors"]:
        md += ["", "## Errors", ""] + [f"- **{e['id']}**: {e['status']}" for e in summary["errors"]]
    return "\n".join(md) + "\n"


def _csv_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        flat = {"id": r["id"], "safety": r.get("safety"), "status": r.get("status")}
        if r.get("status") == "ok":
            flat["hinglish_detected"] = r["detection"]["hinglish_detected"]
            for key, comp in r["retrieval"].items():
                flat[f"{key}_overlap"] = comp["overlap"]
                flat[f"{key}_missing"] = comp["missing_count"]
                if "same_canonical" in comp:
                    flat[f"{key}_same_canonical"] = comp["same_canonical"]
            variants = r.get("variants", [])
            flat["variants_same_canonical"] = f"{sum(v['same_canonical'] for v in variants)}/{len(variants)}"
            if "answers" in r:
                flat["en_hallucination"] = r["answers"]["en"]["hallucination"]["rate"]
                for lang in LANGS:
                    a = r["answers"][lang]
                    flat.update({f"{lang}_plan_consistency": a["plan_consistency"],
                                 f"{lang}_answer_similarity": a["answer_similarity"]["value"],
                                 f"{lang}_hallucination": a["hallucination"]["rate"],
                                 f"{lang}_parity": a["parity_passed"]})
        out.append(flat)
    return out


def write_reports(rows: list[dict], mode: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    paths = {"md": out_dir / "trilingual_report.md", "json": out_dir / "trilingual_report.json",
             "csv": out_dir / "trilingual_report.csv"}
    paths["md"].write_text(_markdown(rows, summary, mode), encoding="utf-8")
    paths["json"].write_text(json.dumps({"mode": mode, "thresholds": THRESHOLDS, "summary": summary, "items": rows},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    flat = _csv_rows(rows)
    fields = list(dict.fromkeys(k for r in flat for k in r))
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return {**paths, "summary": summary, "summary_text": "\n".join(_summary_lines(summary, mode))}
