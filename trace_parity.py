"""Runtime trace of compare_multilingual_questions: logs every retrieval/LLM call in order.

Usage: python trace_parity.py [insulin|bmi|pregnancy ...]   (default: all three UI samples)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

import rag_pipeline as rp

CALLS = []


def wrap(name):
    orig = getattr(rp, name)

    def inner(*args, **kwargs):
        out = orig(*args, **kwargs)
        CALLS.append({"fn": name, "args": args, "out": out})
        n = len(CALLS)
        if name == "retrieve":
            print(f"  [{n}] retrieve(query={args[0]!r}) -> ids {[c['id'] for c in out]}")
        else:
            print(f"  [{n}] {name}  IN={json.dumps(args[0], ensure_ascii=False)[:160]}")
        return out

    setattr(rp, name, inner)


for fn in ("retrieve", "translate_to_hindi", "translate_to_english", "generate_answer_plan",
           "generate_canonical_answer", "generate_direct_answer", "canonicalize_query"):
    wrap(fn)

SAMPLES = {
    "insulin": ("I take insulin for type 2 diabetes. Can I follow the Dixit two-meal plan?",
                "मैं टाइप 2 डायबिटीज के लिए इंसुलिन लेता हूँ। क्या मैं दीक्षित की दिन में दो बार भोजन वाली योजना अपना सकता हूँ?"),
    "bmi": ("What are the revised BMI cutoffs for Indian adults and why do they differ from WHO guidelines?",
            "भारतीय वयस्कों के लिए संशोधित बीएमआई कटऑफ क्या हैं और वे डब्ल्यूएचओ दिशानिर्देशों से अलग क्यों हैं?"),
    "pregnancy": ("Is it okay to follow this plan while pregnant?",
                  "क्या गर्भावस्था के दौरान यह योजना अपनाना ठीक है?"),
    "hunger": ("Why do I feel hungry all the time?",
               "मुझे हर समय भूख क्यों लगती है?"),
}

names = sys.argv[1:] or list(SAMPLES)
for name in names:
    en, hi = SAMPLES[name]
    CALLS.clear()
    print(f"\n==================== {name} ====================")
    r = rp.compare_multilingual_questions(en, hi)
    if r["status"] != "ok":
        print("STATUS:", r)
        continue
    canonical = r["english"]["canonical_answer"]
    th = [c for c in CALLS if c["fn"] == "translate_to_hindi"]
    print("\n(1) CANONICAL ENGLISH ANSWER:\n" + canonical)
    print("\n(2) exact string passed to translate_to_hindi == canonical:",
          [c["args"][0] == canonical for c in th])
    print("    english final_answer == canonical:", r["english"]["final_answer"] == canonical)
    print("(3) retrieve calls:", [c["args"][0] for c in CALLS if c["fn"] == "retrieve"],
          "(a 2nd call is the Hindi question's own retrieval, used only for the retrieval_overlap check)")
    print("(4) generation calls:", [c["fn"] for c in CALLS if c["fn"].startswith("generate")])
    print("(5) hindi final_answer:", "stored checked translation reused" if r["hindi"]["translation_reused"]
          else f"raw translator output = {any(r['hindi']['final_answer'] is c['out'] for c in th)}",
          f"(translation attempts: {r['hindi']['translation_attempts']})")
    print("\n(7) HINDI RETURNED:\n" + r["hindi"]["final_answer"])
    print("\nBACK-TRANSLATION:\n" + r["hindi_back_translation"])
    print("\nPARITY:", "PASS" if r["parity"]["parity_passed"] else "FAIL")
    for c in r["parity"]["checks"]:
        print(f"  {'OK ' if c['passed'] else 'BAD'} {c['name']}: {c['details']}")
