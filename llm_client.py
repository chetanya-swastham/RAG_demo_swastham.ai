"""
llm_client.py — Unified LLM interface for translation, planning, and generation.

Uses Groq API (openai/gpt-oss-120b by default) via httpx.
All LLM interactions are routed through this module for consistency.

There is deliberately NO offline fallback: if the LLM is unavailable, calls raise
LLMError so the demo never silently shows fake output.
"""

import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from hinglish.lexicon import PROTECTED_TERMS, find_protected_terms

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
SEED = 42  # fixed seed + temperature 0 → same input gives same output


class LLMError(RuntimeError):
    """Raised when the LLM is unavailable or returns unusable output."""


def get_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def is_api_key_valid() -> bool:
    key = get_api_key()
    return bool(key and not key.startswith("your_"))


# ---------------------------------------------------------------------------
# Low-level Groq call
# ---------------------------------------------------------------------------

def _call_groq(messages: list[dict], max_tokens: int = 2048,
               json_mode: bool = False) -> str:
    """Send a deterministic chat completion request to Groq."""
    if not is_api_key_valid():
        raise LLMError("GROQ_API_KEY is not set in .env — add a valid Groq key and restart.")

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if "gpt-oss" in LLM_MODEL:
        # Reasoning model: keep thinking short so it doesn't consume the output budget
        payload["reasoning_effort"] = "low"
        payload["max_tokens"] = max_tokens + 2000

    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = _post_with_retries(client, GROQ_BASE_URL, headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"Groq request failed ({e.response.status_code}): {e.response.text[:300]}") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Groq request failed: {e}") from e

    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        # A cut-off answer must never be validated, stored or translated as if it were complete
        raise LLMError(f"LLM output was cut off at the token limit (max_tokens={payload['max_tokens']}).")
    content = (choice.get("message") or {}).get("content")
    if not content or not content.strip():
        raise LLMError("LLM returned an empty response.")
    return content.strip()


_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 8


def _post_with_retries(client: httpx.Client, url: str, headers: dict, **request_kwargs) -> httpx.Response:
    """POST to Groq, retrying rate limits, 5xx errors, timeouts and connection errors.

    request_kwargs go to httpx as-is: json= for chat completions, files=/data= for audio.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        last = attempt == _MAX_ATTEMPTS
        try:
            resp = client.post(url, headers=headers, **request_kwargs)
        except httpx.TransportError as e:  # timeouts and connection errors
            if last:
                raise
            reason, wait = type(e).__name__, min(2 ** attempt, 30)
        else:
            if resp.status_code not in _RETRY_STATUSES or last:
                return resp
            if resp.status_code == 429:
                # Free-tier tokens-per-minute limit: wait for the window Groq tells us
                match = re.search(r"try again in ([\d.]+)s", resp.text)
                wait = float(match.group(1)) + 1 if match else float(resp.headers.get("retry-after", 15))
                reason, wait = "rate limited", min(wait, 65)
            else:
                reason, wait = f"server error {resp.status_code}", min(2 ** attempt, 30)
        print(f"[LLM] {reason}, waiting {wait:.0f}s (attempt {attempt}/{_MAX_ATTEMPTS})", flush=True)
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Return 'hi' if the text is mostly Devanagari, else 'en'."""
    devanagari_count = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    latin_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = devanagari_count + latin_count

    if total == 0:
        return "en"
    return "hi" if devanagari_count / total > 0.3 else "en"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

# From documentRAG.md Section 34 — keeps Hindi terminology consistent across answers.
HINDI_GLOSSARY = {
    "Obesity": "मोटापा",
    "Overweight": "अधिक वजन",
    "BMI": "बीएमआई (बॉडी मास इंडेक्स)",
    "Waist circumference": "कमर का घेरा",
    "Insulin resistance": "इंसुलिन प्रतिरोध",
    "Blood sugar": "ब्लड शुगर / रक्त शर्करा",
    "Type 2 diabetes": "टाइप 2 डायबिटीज (मधुमेह)",
    "Diabetes (when the English does not say type 2)": "डायबिटीज (मधुमेह)",
    "Fasting": "उपवास",
    "Two meals a day": "दिन में दो बार भोजन",
    "Snacking": "बीच-बीच में खाना (never use नाश्ता alone — it means breakfast)",
    "Brisk walk": "तेज गति से टहलना",
    "Physician / doctor": "डॉक्टर",
    # Referral roles keep their specific meaning (never collapse them into plain डॉक्टर)
    "Clinician": "क्लिनिशियन (डॉक्टर)",
    "Obstetrician": "प्रसूति रोग विशेषज्ञ डॉक्टर (ऑब्स्टेट्रिशियन)",
    "Paediatrician": "बाल रोग विशेषज्ञ डॉक्टर (पीडियाट्रिशियन)",
    "Endocrinologist": "एंडोक्राइनोलॉजिस्ट (हार्मोन विशेषज्ञ डॉक्टर)",
    "Diabetes-care provider": "डायबिटीज़-केयर डॉक्टर",
    "Healthcare professional": "हेल्थकेयर प्रोफेशनल (स्वास्थ्य विशेषज्ञ)",
    "Healthcare team": "हेल्थकेयर टीम",
    "Medication": "दवा",
    "Take (a medicine / insulin)": "(दवा / इंसुलिन) लेते हैं — simple present, never ले रहे हैं",
    "Weight loss": "वजन कम करना",
    "Blood pressure": "ब्लड प्रेशर (रक्तचाप)",
    "Thyroid": "थायरॉइड",
    "Family history": "पारिवारिक इतिहास",
}

_GLOSSARY_TEXT = "\n".join(f"- {en} → {hi}" for en, hi in HINDI_GLOSSARY.items())

TO_HINDI_SYSTEM = f"""You are a medical translator for Swastham.ai. Translate the English answer into natural, simple Hindi (Devanagari).

STRICT RULES — this is a patient-safety requirement:
1. Translate EVERYTHING. Do not add, remove, summarise, soften, or reinterpret any statement.
2. Keep every number, unit, range, percentage, and drug name exactly (e.g. 90 cm, 5.7–6.4%, glimepiride, HbA1c).
3. Every caution, warning, and "consult your doctor" instruction must appear with the same strength as in English.
4. Keep the certainty of every statement: "must", "should", "may", "never", "almost certainly" keep their exact strength.
   Keep the tense of every action: English simple present stays simple present (e.g. "if you take medicines" → "यदि आप दवाएं लेते हैं", never "ले रहे हैं").
5. Keep every specific health professional (clinician, obstetrician, paediatrician…) as that role — use the glossary.
6. Keep the same structure: same paragraphs, same bullet points, same bold markers (**). Translate headings and bold labels too.
7. Use this terminology, but only where the English uses that exact term. Never make a term more specific or more general (e.g. "diabetes" is never "type 2 diabetes"; "healthcare professional" is never "healthcare team"):
{_GLOSSARY_TEXT}
8. The output must be written in Hindi (Devanagari), even for headings, lists and tables — never return the English text.
9. Output ONLY the Hindi translation."""

# Hindi → English terms, so a translated query uses the SAME wording as an English user
# (and as the knowledge base). A paraphrase here changes retrieval and therefore the answer.
ENGLISH_TERMS = {
    "दीक्षित / दीक्षित लाइफस्टाइल": "Dixit / Dixit Lifestyle",
    "दिन में दो बार भोजन / दो बार भोजन वाली योजना / दो-भोजन योजना": "two-meal plan (two meals a day)",
    "बीच-बीच में खाना": "snacking",
    "कमर का घेरा / कमर की परिधि": "waist circumference",
    "इंसुलिन प्रतिरोध": "insulin resistance",
    "मोटापा": "obesity",
    "गर्भावस्था": "pregnancy",
    "स्तनपान": "breastfeeding",
    "मधुमेह / डायबिटीज": "diabetes",
    "पीसीओएस": "PCOS",
    "क्लिनिशियन": "clinician",
    "प्रसूति रोग विशेषज्ञ / ऑब्स्टेट्रिशियन": "obstetrician",
    "बाल रोग विशेषज्ञ / पीडियाट्रिशियन": "paediatrician",
    "एंडोक्राइनोलॉजिस्ट": "endocrinologist",
    "हेल्थकेयर प्रोफेशनल / स्वास्थ्य विशेषज्ञ": "healthcare professional",
    "हेल्थकेयर टीम": "healthcare team",
    "दवा / दवाएं / इंसुलिन लेते हैं (or ले रहे हैं)": "take medicine(s) / insulin",
}
_ENGLISH_TERMS_TEXT = "\n".join(f"- {hi} → {en}" for hi, en in ENGLISH_TERMS.items())

TO_ENGLISH_SYSTEM = f"""You are a precise Hindi-to-English translator for a medical/health AI system.
Translate the Hindi text to English accurately and completely.
Preserve all medical terms, numbers, drug names, and cautions.
Keep proper names and named plans exactly (never drop or generalise them).
Translate literally, sentence by sentence: render each Hindi action verb as its plain English verb in the same tense (लेते हैं → take, परामर्श करें → consult, बचें → avoid, सीमित करें → limit, कम करें → reduce, शुरू करें → start, बंद करें → stop, जांच करें → check, निगरानी करें → monitor, पालन करें → follow).
Use this terminology:
{_ENGLISH_TERMS_TEXT}
Output ONLY the English translation, nothing else."""


def translate_to_english(hindi_text: str) -> str:
    """Translate Hindi text to English (used for retrieval and back-translation checks)."""
    messages = [
        {"role": "system", "content": TO_ENGLISH_SYSTEM},
        {"role": "user", "content": hindi_text},
    ]
    return _call_groq(messages)


def _translation_problems(english_text: str, hindi_text: str) -> list[str]:
    """What a Hindi translation visibly got wrong: not Hindi at all, or numbers changed."""
    from validator import extract_numbers

    problems = []
    letters = [ch for ch in hindi_text if ch.isalpha()]
    devanagari = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
    if not letters or devanagari / len(letters) < 0.5:
        problems.append("The output was not in Hindi. Write the whole translation in Hindi (Devanagari).")
    en_nums, hi_nums = extract_numbers(english_text), extract_numbers(hindi_text)
    if en_nums != hi_nums:
        problems.append(f"Numbers must match the English exactly. Missing: {sorted(en_nums - hi_nums)}; "
                        f"not in the English: {sorted(hi_nums - en_nums)}.")
    return problems


def translate_to_hindi(english_text: str, feedback: str | None = None) -> str:
    """Faithfully translate the canonical English answer into Hindi.

    The answer is passed as delimited source text, so the model translates it rather than
    echoing it. If the result is not Hindi or changes a number, the SAME English text is
    translated once more with that feedback; nothing is ever generated or rewritten here.
    `feedback` lists what a previous translation lost (from the parity check), for a retranslation.
    """
    if not english_text or not english_text.strip():
        return ""
    retry_note = ""
    if feedback:
        retry_note = ("A previous translation of this answer lost or changed the following when it was "
                      f"back-translated to English:\n{feedback}\nKeep every one of these exactly as in the English.\n\n")
    messages = [
        {"role": "system", "content": TO_HINDI_SYSTEM},
        # The instruction is repeated AFTER the text: with it only before, the model echoed
        # heading/list-heavy answers back in English.
        {"role": "user", "content": (
            f"Translate this English answer into Hindi:\n\n<answer>\n{english_text}\n</answer>\n\n"
            f"{retry_note}"
            "Now write the complete Hindi (Devanagari) translation of the answer above. "
            "Every sentence, heading and list label must be in Hindi; keep numbers and units unchanged.")},
    ]
    hindi = _call_groq(messages, max_tokens=3000)
    problems = _translation_problems(english_text, hindi)
    if problems:
        messages += [
            {"role": "assistant", "content": hindi},
            {"role": "user", "content": "Fix the translation of the same English answer:\n" + "\n".join(problems)},
        ]
        hindi = _call_groq(messages, max_tokens=3000)
        problems = _translation_problems(english_text, hindi)
        if problems:
            raise LLMError("Hindi translation is not faithful: " + " ".join(problems))
    return hindi.replace("<answer>", "").replace("</answer>", "").strip()


# ---------------------------------------------------------------------------
# Hinglish (Romanised Hindi mixed with English)
#
# Input:  Hinglish question → standard Hindi question (hinglish.normalizer, prompt below),
#         then the SAME Hindi → English translator as a Devanagari question.
# Output: Hinglish translation of the canonical English answer, checked like Hindi.
# ---------------------------------------------------------------------------

_PROTECTED_TEXT = ", ".join(PROTECTED_TERMS)

ROMAN_TO_HINDI_SYSTEM = """You convert a health question written in Hinglish (Romanised Hindi mixed with English) into standard Hindi written in Devanagari.

Rules:
1. Write every Hindi word in Devanagari with standard spelling (e.g. "kya main kha sakta hoon" → "क्या मैं खा सकता हूँ").
2. Keep English words as they are, in Latin script (e.g. "fruits", "snack", "exercise", "can I eat").
3. Keep every placeholder like [[TA]] exactly as written, once each. They stand for medical terms.
4. Keep every number exactly.
5. Do not answer, explain, translate into English, add or remove any meaning. Keep it a question if it is a question.
Output ONLY the converted question."""

TO_HINGLISH_SYSTEM = f"""You are a medical translator for Swastham.ai. Translate the English answer into natural Hinglish: Hindi grammar and everyday Hindi words, written in Roman (Latin) script, mixed with the common English words Indians use (fruits, sugar, doctor, diet, exercise, weight, walk).

STRICT RULES — this is a patient-safety requirement:
1. Translate EVERYTHING. Do not add, remove, summarise, soften, or reinterpret any statement.
2. Keep every number, unit, range, percentage, and drug name exactly (e.g. 90 cm, 5.7–6.4%, glimepiride).
3. Keep these medical terms in English, spelled exactly like this, wherever the English uses them: {_PROTECTED_TEXT}.
4. Every caution, warning, and "consult your doctor" instruction must appear with the same strength as in English.
5. Keep the certainty of every statement ("must" → "zaroor … chahiye", "never" → "kabhi nahi", "may" → "ho sakta hai") and the tense of every action.
6. Keep every specific health professional as that role (clinician, obstetrician, paediatrician, endocrinologist — in English).
7. Keep the same structure: same paragraphs, same bullet points, same bold markers (**). Translate headings too.
8. Write in Roman script only — never Devanagari. It must read as Hinglish (e.g. "Abhi fruits allow nahi hain. Apne doctor se baat karein."), never as plain English and never as pure Hindi.
9. Output ONLY the Hinglish translation."""

HINGLISH_TO_ENGLISH_SYSTEM = f"""You are a precise Hinglish-to-English translator for a medical/health AI system.
The text is Hinglish: Hindi written in Roman script, mixed with English words.
Translate it to English accurately and completely, sentence by sentence.
Preserve all medical terms, numbers, drug names, and cautions. Keep proper names and named plans exactly.
Render each Hindi action verb as its plain English verb in the same tense (lete hain → take, salah lein / consult karein → consult, bachein / avoid karein → avoid, seemit karein → limit, kam karein → reduce, shuru karein → start, band karein → stop, check karein → check, monitor karein → monitor, follow karein → follow).
Use this terminology (Devanagari or its Roman spelling):
{_ENGLISH_TERMS_TEXT}
Output ONLY the English translation, nothing else."""


def convert_roman_to_devanagari(masked_question: str, feedback: str | None = None) -> str:
    """One LLM call: Hinglish question (medical terms masked) → Devanagari Hindi question.

    The gates, retry and term restoring live in hinglish.normalizer.to_standard_hindi.
    """
    content = f"Convert this question:\n\n<question>\n{masked_question}\n</question>"
    if feedback:
        content += f"\n\nA previous conversion had these problems; fix them:\n{feedback}"
    messages = [
        {"role": "system", "content": ROMAN_TO_HINDI_SYSTEM},
        {"role": "user", "content": content},
    ]
    out = _call_groq(messages, max_tokens=300)
    return out.replace("<question>", "").replace("</question>", "").strip()


def _hinglish_problems(english_text: str, hinglish_text: str) -> list[str]:
    """What a Hinglish translation visibly got wrong: Devanagari, plain English, lost numbers or terms."""
    from hinglish.detector import detect_hinglish
    from validator import extract_numbers

    problems = []
    letters = [ch for ch in hinglish_text if ch.isalpha()]
    devanagari = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
    if letters and devanagari / len(letters) > 0.05:
        problems.append("The output used Devanagari. Write the whole translation in Roman (Latin) script.")
    elif not detect_hinglish(hinglish_text)["is_hinglish"]:
        problems.append("The output reads as plain English. Write natural Hinglish: Hindi grammar and words "
                        "(hai, ke, ko, karein, chahiye, nahi…) in Roman script, with common English words mixed in.")
    en_nums, hg_nums = extract_numbers(english_text), extract_numbers(hinglish_text)
    if en_nums != hg_nums:
        problems.append(f"Numbers must match the English exactly. Missing: {sorted(en_nums - hg_nums)}; "
                        f"not in the English: {sorted(hg_nums - en_nums)}.")
    lost = sorted(find_protected_terms(english_text) - find_protected_terms(hinglish_text))
    if lost:
        problems.append(f"Keep these medical terms in English, spelled exactly: {lost}.")
    return problems


def translate_to_hinglish(english_text: str, feedback: str | None = None) -> str:
    """Faithfully translate the canonical English answer into Hinglish (Roman script).

    Same contract as translate_to_hindi: translation only, one retranslation of the SAME
    English text when a gate fails, then LLMError. `feedback` comes from the parity check.
    """
    if not english_text or not english_text.strip():
        return ""
    retry_note = ""
    if feedback:
        retry_note = ("A previous translation of this answer lost or changed the following when it was "
                      f"back-translated to English:\n{feedback}\nKeep every one of these exactly as in the English.\n\n")
    messages = [
        {"role": "system", "content": TO_HINGLISH_SYSTEM},
        {"role": "user", "content": (
            f"Translate this English answer into Hinglish:\n\n<answer>\n{english_text}\n</answer>\n\n"
            f"{retry_note}"
            "Now write the complete Hinglish translation (Roman script, Hindi grammar, common English words "
            "mixed in) of the answer above. Keep numbers, units and the listed medical terms unchanged.")},
    ]
    hinglish = _call_groq(messages, max_tokens=3000)
    problems = _hinglish_problems(english_text, hinglish)
    if problems:
        messages += [
            {"role": "assistant", "content": hinglish},
            {"role": "user", "content": "Fix the translation of the same English answer:\n" + "\n".join(problems)},
        ]
        hinglish = _call_groq(messages, max_tokens=3000)
        problems = _hinglish_problems(english_text, hinglish)
        if problems:
            raise LLMError("Hinglish translation is not faithful: " + " ".join(problems))
    return hinglish.replace("<answer>", "").replace("</answer>", "").strip()


def back_translate_hinglish(hinglish_text: str) -> str:
    """Literal Hinglish → English translation, used only by the parity check."""
    messages = [
        {"role": "system", "content": HINGLISH_TO_ENGLISH_SYSTEM},
        {"role": "user", "content": hinglish_text},
    ]
    return _call_groq(messages)


# ---------------------------------------------------------------------------
# Query canonicalisation — makes EN and HI questions converge on ONE retrieval query
# ---------------------------------------------------------------------------

CANONICAL_QUERY_SYSTEM = """You normalise health questions so that the same question, however it was worded or translated, becomes the SAME standalone English question.

Rules:
1. Output one short, neutral English question (max ~20 words) in the simplest standard phrasing.
2. Keep the topic and every specific detail: named plans (e.g. "Dixit Lifestyle two-meal plan"), medicines, conditions, numbers, who is asking (child, pregnant, on insulin).
3. Use these standard names: "Dixit Lifestyle two-meal plan", "waist circumference", "insulin resistance", "type 2 diabetes", "HbA1c", "BMI".
4. Remove filler and politeness; do not answer the question; do not add information.
5. Same meaning in → same wording out. Prefer the form "What is X?", "How does X work?", "Can I do X if Y?", "Why ...?".
Output ONLY the normalised question."""


_CONTRACTIONS = [
    (r"\bwhat's\b", "what is"), (r"\bit's\b", "it is"), (r"\bi'm\b", "i am"),
    (r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"), (r"n't\b", " not"),
    (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"),
]


def normalise_question(text: str) -> str:
    """Cheap, deterministic clean-up of a question's wording.

    Lowercases, expands contractions and drops punctuation and filler words, so trivially
    different spellings ("What's BMI" / "what is BMI?") share one cache key. It does not
    decide meaning; that is the canonicaliser's job.
    """
    q = (text or "").lower().replace("’", "'")
    for pattern, repl in _CONTRACTIONS:
        q = re.sub(pattern, repl, q)
    q = re.sub(r"\bkeep\s+feeling\b", "feel", q)
    q = re.sub(r"\bkeep\s+getting\b", "get", q)
    q = re.sub(r"\bkeep\s+being\b", "be", q)
    q = re.sub(r"\b(?:just|really|constantly|please)\b", " ", q)
    q = re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", q)
    return " ".join(q.split())


def canonicalize_query(english_query: str) -> str:
    """Rewrite an English question into ONE standard question, used for retrieval and as the answer-store key.

    Hindi questions are translated to English first; this function only ever sees English.
    Callers cache the result per normalised question (rag_pipeline), so a question is only
    canonicalised by the LLM once. Without an API key the normalised wording is returned.
    """
    cleaned = normalise_question(english_query)
    if not cleaned:
        return ""
    if not is_api_key_valid():
        return cleaned
    messages = [
        {"role": "system", "content": CANONICAL_QUERY_SYSTEM},
        {"role": "user", "content": english_query.strip()},
    ]
    return _call_groq(messages, max_tokens=100).strip().strip('"')


# ---------------------------------------------------------------------------
# Follow-up resolution (voice agent conversations)
# ---------------------------------------------------------------------------

CONTEXTUALIZE_SYSTEM = """You rewrite a follow-up question from a health conversation into ONE standalone English question.

Rules:
- Only resolve references to earlier turns ("it", "this plan", "that", "what about ...", "and if I am ...").
- Keep every condition the user mentioned earlier that the follow-up still depends on (e.g. "I take insulin").
- NEVER add facts, advice or conditions the user did not state.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question, no quotes or explanation."""

HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 400


def contextualize_question(question_en: str, history: list[dict]) -> str:
    """Follow-up + earlier turns → a standalone English question. Callers skip this when there is no history."""
    turns = "\n\n".join(
        f"User: {t.get('question', '')}\nAssistant: {t.get('answer', '')[:HISTORY_ANSWER_CHARS]}"
        for t in history[-HISTORY_TURNS:]
    )
    messages = [
        {"role": "system", "content": CONTEXTUALIZE_SYSTEM},
        {"role": "user", "content": f"Conversation so far:\n{turns}\n\nFollow-up question: {question_en.strip()}"},
    ]
    return _call_groq(messages, max_tokens=150).strip().strip('"')


# ---------------------------------------------------------------------------
# Answer Plan Generation
# ---------------------------------------------------------------------------

ANSWER_PLAN_SYSTEM = """You are a medical-knowledge AI assistant for Swastham.ai.
Given a user question and retrieved evidence chunks, create a structured Answer Plan.

The Answer Plan must be a valid JSON object with these exact keys:
{
  "required_points": ["short key facts that MUST appear in the final answer"],
  "important_conditions": ["conditions, caveats, or safety cautions from the evidence"],
  "restrictions": ["things the answer must NOT claim or recommend"],
  "evidence": ["[Chunk N] short quote or paraphrase supporting a point"]
}

Rules:
- Base the plan ONLY on the provided evidence chunks. Do NOT add external knowledge.
- If the question involves medication, pregnancy, breastfeeding, children, or eating disorders,
  the relevant safety caution from the evidence MUST be in important_conditions.
- If the evidence does not address the question, say so in required_points.
- Output ONLY the JSON object."""


def _format_evidence(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Chunk {c['id']}] ({c['heading']})\n{c['text']}" for c in chunks)


def generate_answer_plan(question: str, chunks: list[dict]) -> dict:
    """Generate a structured answer plan from the question and retrieved chunks."""
    messages = [
        {"role": "system", "content": ANSWER_PLAN_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nRetrieved Evidence:\n{_format_evidence(chunks)}"},
    ]
    last_error = None
    for _ in range(2):  # one retry on malformed JSON
        raw = _call_groq(messages, max_tokens=1500, json_mode=True)
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        for key in ("required_points", "important_conditions", "restrictions", "evidence"):
            plan.setdefault(key, [])
        return plan
    raise LLMError(f"Answer plan was not valid JSON: {last_error}")


# ---------------------------------------------------------------------------
# Canonical English Answer Generation
# ---------------------------------------------------------------------------

CANONICAL_ANSWER_SYSTEM = """You are a medical-knowledge AI assistant for Swastham.ai.
Generate a clear, accurate, and helpful answer to the user's health question.

Rules:
1. Follow the Answer Plan exactly — include ALL required_points and ALL important_conditions.
2. Respect ALL restrictions listed in the plan.
3. Use ONLY the provided evidence. Do NOT add facts from outside the evidence.
4. If evidence is insufficient, say so honestly.
5. Be concise but thorough. Use simple language suitable for a general audience.
6. Safety cautions (medication, pregnancy, children, eating disorders) must be stated clearly, never softened.
7. Do NOT include disclaimers like "I'm an AI" — just answer the question directly."""


def generate_canonical_answer(question: str, plan: dict, chunks: list[dict],
                              feedback: str | None = None) -> str:
    """Generate the canonical English answer following the plan.

    `feedback` lists what a previous attempt missed; used for the single validation retry.
    """
    plan_text = json.dumps(plan, indent=2, ensure_ascii=False)
    content = f"Question: {question}\n\nAnswer Plan:\n{plan_text}\n\nEvidence:\n{_format_evidence(chunks)}"
    if feedback:
        content += f"\n\nYour previous answer failed validation. Fix this:\n{feedback}"

    messages = [
        {"role": "system", "content": CANONICAL_ANSWER_SYSTEM},
        {"role": "user", "content": content},
    ]
    return _call_groq(messages, max_tokens=1500)


# ---------------------------------------------------------------------------
# Baseline ("before") — answers directly in the user's language
# ---------------------------------------------------------------------------

DIRECT_ANSWER_SYSTEM = """You are a health assistant. Answer the user's question using the evidence provided.
Answer in {language_name}."""


def generate_direct_answer(question: str, chunks: list[dict], language: str) -> str:
    """Baseline: the LLM reasons and answers directly in the user's language.

    This mirrors the inconsistent behaviour being fixed — each language gets its
    own independent generation instead of one canonical answer.
    """
    language_name = {"hi": "Hindi (Devanagari script)",
                     "hinglish": "Hinglish (Romanised Hindi mixed with English)"}.get(language, "English")
    messages = [
        {"role": "system", "content": DIRECT_ANSWER_SYSTEM.format(language_name=language_name)},
        {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{_format_evidence(chunks)}"},
    ]
    return _call_groq(messages, max_tokens=1500)


# ---------------------------------------------------------------------------
# Fingerprint of everything that shapes an answer (used to invalidate stored answers)
# ---------------------------------------------------------------------------

def prompts_fingerprint() -> str:
    """Hash of the model, every prompt and both glossaries. Any change → stored answers are stale."""
    import hashlib

    parts = [LLM_MODEL, str(SEED), CANONICAL_QUERY_SYSTEM, ANSWER_PLAN_SYSTEM, CANONICAL_ANSWER_SYSTEM,
             TO_HINDI_SYSTEM, TO_ENGLISH_SYSTEM,
             ROMAN_TO_HINDI_SYSTEM, TO_HINGLISH_SYSTEM, HINGLISH_TO_ENGLISH_SYSTEM]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]
