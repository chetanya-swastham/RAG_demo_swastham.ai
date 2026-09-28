# Implementation Guide

How the Swastham English / Hindi / Hinglish parity RAG, with its voice assistant and voice agent, is built: the design, every module, the checks and their thresholds, the evaluation, the bugs found along the way, and how to extend it.

For setup and day-to-day use, see [`README.md`](README.md).

---

## Contents

1. [Goal and original brief](#1-goal-and-original-brief)
2. [Design principles](#2-design-principles)
3. [Architecture](#3-architecture)
4. [Module reference](#4-module-reference)
5. [Ingestion](#5-ingestion)
6. [Query pipeline, step by step](#6-query-pipeline-step-by-step)
7. [Checks: validation, translation gates, parity](#7-checks-validation-translation-gates-parity)
8. [Hinglish](#8-hinglish)
9. [Evaluation](#9-evaluation)
10. [Voice assistant and voice agent](#10-voice-assistant-and-voice-agent)
11. [Server, HTTP API and web UI](#11-server-http-api-and-web-ui)
12. [LLM layer](#12-llm-layer)
13. [Answer store](#13-answer-store)
14. [Tests](#14-tests)
15. [Engineering history: parity bugs found and fixed](#15-engineering-history-parity-bugs-found-and-fixed)
16. [Known limitations and open risks](#16-known-limitations-and-open-risks)
17. [Extending the system](#17-extending-the-system)

---

## 1. Goal and original brief

**Problem.** When an LLM answers an English question in English, and the same question in Hindi directly in Hindi, it produces two independent answers. They drift apart: different facts, less detail, and sometimes a missing safety caution (*"adjust your insulin dose with your doctor"*). For a health product this is not acceptable.

**Brief (Swastham.ai proof of concept).** Take an English knowledge document on obesity and diabetes, and ask the same question in English and Hindi. Both answers must rest on the **same evidence and the same underlying answer**, and Hindi must never reason on its own. The brief set these constraints:
- **Ingestion:** English document → extract → chunk → English embeddings → FAISS.
- **Query:** detect language; Hindi is translated to English; both languages converge on *retrieved evidence → Answer Plan → canonical English answer*; Hindi output is a translation of that answer.
- **Answer Plan** (`required_points`, `important_conditions`, `restrictions`, `evidence`) before generation. The answer uses retrieved evidence only.
- **Lightweight validation:** required points present, supported by evidence, no obvious contradiction.
- **Keep it simple:** Python, PyMuPDF, sentence-transformers, FAISS, one LLM. No separate Hindi vector database, no agent frameworks, no graph or hybrid search, no reranking, no external database.
- **A small UI** to ask questions and to compare English and Hindi answers side by side.

**Built beyond the brief, in order:**
1. Parity checks and fail-closed translation gates.
2. Question canonicalisation and an answer store.
3. Hinglish (Romanised Hindi) input and output.
4. A trilingual evaluation suite.
5. A voice assistant and a hands-free voice agent.

Each addition keeps the core rule: **one canonical English answer per question; every other language is a checked translation of it.**

---

## 2. Design principles

| Principle | Where it shows |
|---|---|
| **Generate once, translate everywhere** | Only `generate_canonical_answer` produces content. Hindi, Hinglish and speech are all derived from that one English answer |
| **Same meaning → same answer** | Questions are canonicalised, and answers are stored per canonical question (`rag_pipeline._canonical_question`, `_answer_for`) |
| **Validate translations like any model output** | Script, number and medical-term gates on every translation, then a back-translation parity check |
| **Fail closed on safety** | A translation that loses a number or a safety caution after one retry is never shown or spoken (`BLOCKING_PARITY_CHECKS`) |
| **Fix causes, not thresholds** | Every failure found was fixed in the prompt, glossary or pipeline. Thresholds were never lowered (§15) |
| **Deterministic where possible** | Temperature 0 and seed 42 on every call. Stored answers and translations give repeatable output |
| **Checks are pure functions** | `validator.py` makes no LLM calls, so every check is unit-tested offline |
| **Voice adds I/O only** | Speech wraps `rag_pipeline.query()`; the spoken text is its `final_answer` |

---

## 3. Architecture

```
                         ┌──────────────────────── Browser ─────────────────────────┐
                         │ Ask tab (typed / 🎤)   Parity Compare    Voice Agent tab  │
                         └──────┬───────────────────────┬─────────────────┬─────────┘
                    /api/transcribe (Whisper)     /api/compare     /api/query (+history)
                                │                        │          /api/speak (edge-tts)
                                ▼                        ▼                 ▲
 English ──────────────────────────────────────────┐                        │
 Hindi ─────────► translate_to_english ────────────┤                        │
 Hinglish ─► detect ─► normalise ─► mask terms ─► LLM → Devanagari ─► restore ┘ (then as Hindi)
                                                    ▼
                     Follow-up? (voice agent history) → contextualize_question
                                                    ▼
                     Canonicalise question (one per meaning, cached per wording)
                                                    ▼
                     Answer store lookup ── hit ──► reuse stored entry
                                                    │ miss
                                                    ▼
                     FAISS retrieval (MiniLM, top-K English chunks)
                                                    ▼
                     Answer Plan (JSON) → Canonical English answer → Validator (+1 regeneration)
                                                    ▼  (stored only if validation passed)
             ┌──────────────────────────────────────┼──────────────────────────────────────┐
             ▼                                      ▼                                      ▼
      English output                 Hindi translation of THAT answer        Hinglish translation of THAT answer
                                     gates → back-translate → parity         gates → back-translate → parity
                                     (1 retranslation on failure)            (1 retranslation on failure)
                                                    ▼
                          final_answer ─► speech_sentences ─► voice output
```

---

## 4. Module reference

| Path | Responsibility | Key functions |
|---|---|---|
| `ingest.py` | Document → chunks → embeddings → FAISS; retrieval | `chunk_document`, `build_index`, `save_artifacts`, `load_artifacts`, `retrieve`, `get_embedding_model` |
| `llm_client.py` | Every LLM call (Groq), prompts, glossaries, translation gates | `_call_groq`, `_post_with_retries`, `detect_language`, `translate_to_english`, `translate_to_hindi`, `translate_to_hinglish`, `back_translate_hinglish`, `convert_roman_to_devanagari`, `normalise_question`, `canonicalize_query`, `contextualize_question`, `generate_answer_plan`, `generate_canonical_answer`, `generate_direct_answer`, `prompts_fingerprint` |
| `rag_pipeline.py` | Orchestration: the single answer path, answer store, comparison | `query`, `query_baseline`, `compare_multilingual_questions`, `resolve_language`, `question_similarity` |
| `validator.py` | Pure-function checks | `validate_answer`, `parity_report`, `safety_caution_check`, `extract_numbers`, `detect_cautions` |
| `hinglish/` | Hinglish detection, spelling normalisation, protected terms, Roman → Devanagari with gates | `detect_hinglish`, `normalise_hinglish`, `mask_terms`, `restore_terms`, `to_standard_hindi`, `PROTECTED_TERMS` |
| `voice.py` | Speech → text (Groq Whisper), text → speech (edge-tts), speech text cleaning | `transcribe`, `speak`, `clean_for_speech`, `split_sentences`, `speech_sentences`, `is_available` |
| `server.py` | Threaded HTTP server: static UI plus JSON/audio API, request validation | `RAGHandler`, `parse_query_request`, `parse_voice_options`, `parse_compare_request`, `parse_speak_request`, `parse_transcribe_language` |
| `web/index.html`, `web/style.css` | UI: Ask, Parity Compare and Voice Agent tabs | |
| `web/app.js` | Ask and Compare tabs, voice assistant (mic, Listen, auto-read) | `handleAsk`, `renderAskResult`, `renderCompareResult`, `readAnswerAloud` |
| `web/voice.js` | Browser voice engine | `Voice.Mic`, `Voice.capture`, `Voice.watchForBargeIn`, `Voice.Speaker`, `Voice.transcribe`, `Voice.synthesise` |
| `web/agent.js` | Voice agent conversation loop and chat UI | `start`, `loop`, `stop`, `speakInterruptible` |
| `evaluate.py` | 15 EN/HI pairs: canonical pipeline vs baseline → `eval_results.csv` | |
| `evaluation/` | Trilingual evaluation: metrics, retrieval consistency, runner, report writer, dataset | `run_trilingual_eval.main`, `metrics.*`, `retrieval_consistency.compare_item`, `report.write_reports` |
| `trace_parity.py` | Runtime trace: logs every retrieval and LLM call for the sample questions | |
| `test_parity.py`, `test_hinglish.py`, `test_voice.py` | 81 offline unit tests | |
| `documentRAG.md` | The English knowledge base (obesity, diabetes, the Dixit Lifestyle) | |
| `eval_set.json` | 15 EN/HI question pairs (6 safety-critical) | |
| `data/` | Generated: `chunks.json`, `faiss_index.bin`, `index_meta.json` (newer ingests), `answer_store.json` | |
| `reports/` | Generated trilingual evaluation reports | |

---

## 5. Ingestion

`python ingest.py --source documentRAG.md [--model NAME] [--output-dir DIR]`

1. **Extract.** Markdown is read as-is, and PDFs go through PyMuPDF (`extract_text`).
2. **Section split** (`_split_by_sections`). Markdown headings (`#` to `####`) mark section boundaries. A document without them falls back to numbered headings (`12. Title`). Each chunk carries its heading path (`Part > Section`); the level-1 document title is left out because it is identical everywhere.
3. **Size split** (`_split_long_text`). Sections longer than `CHUNK_MAX_SIZE = 1400` characters are split on sentence boundaries with `CHUNK_OVERLAP_CHARS = 150` of overlap.
4. **Safety tagging** (`_is_safety_critical`). A chunk is tagged `safety_critical` when it comes from sections 15, 18 or 28 (which the corpus marks as safety-critical), or contains keywords such as *sulfonylurea*, *glimepiride*, *on insulin*, *pregnan*, *breastfeed*, *eating disorder* or *hypoglycaemia*.
5. **Embed.** `"heading: text"` is embedded with `all-MiniLM-L6-v2`, so the heading context influences retrieval. Vectors are L2-normalised and stored in `faiss.IndexFlatIP`, which gives exact cosine search.
6. **Persist.** `data/chunks.json`, `data/faiss_index.bin`, and `data/index_meta.json` (the embedding model name). The pipeline refuses to start if `EMBEDDING_MODEL` differs from the model that built the index.

The current index has **83 chunks, 23 of them safety-critical**. There is one English index and no Hindi index.

**Retrieval** (`ingest.retrieve`): embed the canonical English question, take the top `TOP_K` (default 4) by cosine similarity, and return the chunks with their `score`.

---

## 6. Query pipeline, step by step

`rag_pipeline.query(user_query, language_override=None, history=None, reply_language=None)` is the **single answer path**, used by the Ask tab, the voice assistant, the voice agent and the CLI.

| # | Step | Code | Notes |
|---|---|---|---|
| 1 | **Resolve language** | `resolve_language` → `llm_client.detect_language`, `hinglish.detect_hinglish` | More than 30% Devanagari → `hi`. Latin script with enough Roman-Hindi marker words → `hinglish`. Otherwise `en`. An override skips detection |
| 2 | **Hinglish → standard Hindi** | `_hinglish_to_hindi` → `hinglish.to_standard_hindi` | Cached per normalised wording in `answer_store.json → hinglish_queries` (§8) |
| 3 | **Translate question to English** | `_prepare_query` → `translate_to_english` | Hindi and Hinglish use the same translator, with a fixed Hindi → English term list (`ENGLISH_TERMS`), so questions arrive in the knowledge base's own wording |
| 4 | **Resolve follow-up** (voice agent only) | `contextualize_question(question, history)` | Rewrites *"and if I'm pregnant?"* into a standalone question using the last 3 turns. Only resolves references and never adds facts. **Skipped when there is no history**, so single questions are unaffected. Shown as `standalone_query` |
| 5 | **Canonicalise** | `_canonical_question` → `canonicalize_query` | The LLM rewrites the question into one standard English question. Cached per `normalise_question` wording (contractions expanded, filler words dropped, punctuation stripped), so each wording costs one LLM call ever |
| 6 | **Answer store** | `_answer_for` | Hit → reuse the stored chunks, plan and answer (`answer_reused: true`). Miss → steps 7–10 |
| 7 | **Retrieve** | `_retrieve` → `ingest.retrieve` | Top-K English chunks for the canonical question |
| 8 | **Answer Plan** | `generate_answer_plan` | JSON with `required_points`, `important_conditions`, `restrictions`, `evidence`. Built from evidence only; safety cautions become required conditions |
| 9 | **Canonical English answer** | `generate_canonical_answer` | Follows the plan and uses only the evidence |
| 10 | **Validate** | `validator.validate_answer` | On failure, **one** regeneration with the failure as feedback. Stored only if validation passes (§7) |
| 11 | **Output language** | `reply_language or lang` | English → the canonical answer as-is. Hindi or Hinglish → step 12 |
| 12 | **Checked translation** | `_hindi_for` / `_hinglish_for` → `_translate_checked` | Translate the same English answer, back-translate, run `parity_report`, retry once with the failed checks as feedback. A passing translation is stored with the answer entry and reused |
| 13 | **Fail closed** | `BLOCKING_PARITY_CHECKS = {numbers_match, cautions_preserved}` | If either still fails, `LLMError` is raised and nothing is shown or spoken. A failure of only another check is shown, reported in the trace, and not stored |

The returned **trace** includes the user query, detected language, translated, standalone and canonical questions, retrieved chunks, Answer Plan, validation, attempts, reuse flags, `final_answer`, plus back-translation and parity for Hindi or Hinglish, and (from the server) `speech_sentences`.

**Baseline** (`query_baseline`): the pre-project behaviour for comparison. Same retrieval, but the LLM answers directly in the user's language (`generate_direct_answer`).

**Parity Compare** (`compare_multilingual_questions`):
1. **Intent gate.** The English and Hindi (and Hinglish) questions must canonicalise to the same question, or have cosine similarity ≥ 0.75. Otherwise the result is `Question Intent Mismatch`.
2. **One shared answer.** A single canonical answer is built and translated for every language.
3. **Evidence check.** Each language's own canonical question is retrieved separately, only to check that it would reach the same evidence.
4. **Baseline (optional).** The baseline answers are scored with the same checks.

---

## 7. Checks: validation, translation gates, parity

Thresholds are the project's quality bar. When a check fails, the fix goes into the cause, never into the threshold. All keyword checks match at the **start of a word** (`validator._has_word`): *kid* does not match *kidney*, *take* does not match *intake*, and *insulin resistance* is not a medication caution.

### 7.1 Canonical answer validation (`validator.validate_answer`)

| Check | Passes when |
|---|---|
| `required_points_coverage` | ≥ 60% of the plan's required points appear (key-term match, ≥ 50% of each point's terms) |
| `evidence_grounding` | ≥ 50% of answer sentences are supported by the evidence (≥ 40% of each sentence's key terms appear in the chunks) |
| `restriction_compliance` | No answer sentence echoes ≥ 70% of a restriction's key terms |
| `safety_cautions` | For each caution raised by **both** the question and the evidence (diabetes medication, pregnancy, children, eating disorders), the answer carries that caution **and** a doctor referral (`REFERRAL_TERMS`) |

On failure, `feedback` lists the missing points, the missing cautions and the restrictions to avoid, and it is passed to one regeneration.

### 7.2 Translation gates (before parity)

| Output | Gate (`llm_client`) | Rejects |
|---|---|---|
| Hindi | `_translation_problems` | Mostly non-Devanagari output (the model echoing English); any changed number |
| Hinglish | `_hinglish_problems` | Devanagari output (> 5% of letters); plain English (must pass `detect_hinglish`); changed numbers; a protected medical term that is missing |
| Hinglish → Hindi (question) | `hinglish.normalizer.conversion_problems` | Lost or invented placeholders; changed numbers; strong Roman-Hindi words left in Latin script; no Devanagari |

A failed gate triggers one retry of the **same source text** with the problem as feedback. A second failure raises `LLMError`.

### 7.3 Parity (`validator.parity_report`)

This compares the English answer with the translation's **back-translation**:

| Check | Passes when |
|---|---|
| `retrieval_overlap` (Compare only) | Jaccard overlap ≥ 0.6 between the answer's chunks and the chunks the other language's own canonical question retrieves |
| `numbers_match` **(blocking)** | Exactly the same set of numbers (Devanagari digits normalised) |
| `cautions_preserved` **(blocking)** | Every English caution concept and the doctor referral survive |
| `recommendation_match` | Every recommendation verb in the English (`consult`, `avoid`, `take`, `monitor`, …) appears in the back-translation |
| `content_coverage` | ≥ 80% sentence coverage in **both** directions (EN→back and back→EN) |

---

## 8. Hinglish

Hinglish adds no knowledge base, retrieval path or reasoning path. It joins the Hindi path before retrieval, and leaves as a checked translation of the canonical English answer.

### 8.1 Detection (`hinglish/detector.py`)

Detection is deterministic and makes no LLM call. It runs only when the text is not already Hindi.
- Words are lowercased and mapped to a standard spelling.
- **Strong markers** (words that are not English: `kya, hai, mujhe, nahi, sakta, chahiye, karne, ke, ko, se, pehle, bhook…`) score 1, as do Romanised spelling variants (`mai`, `hu`, `nahin`).
- **Weak markers** (words that are also English: `main, me, to, hi, ki, so, par…`) score 0.5, so they never make an English sentence Hinglish on their own.
- The text is Hinglish when the **score ≥ 1.5** and **≥ 20% of its words** are markers.

| Input | Result |
|---|---|
| kya mai fruit kha sakta hu | Hinglish (score 5.0) |
| mujhe diabetes hai can i eat fruit | Hinglish (score 2.0, ratio 0.29) |
| What is the main cause of obesity? | English (score 0.5) |

### 8.2 Normalisation and conversion (`hinglish/normalizer.py`)

1. **`normalise_hinglish`** (deterministic): lowercase, one spelling per Hindi word (`mai→main`, `hu→hoon`, `kyu→kyun`, `bhookh→bhook`…), punctuation dropped, numbers kept, medical terms in standard spelling. The result is the **cache key**, so every spelling of a question is converted once and always to the same Hindi.
2. **`mask_terms`**: protected terms become the placeholders `[[TA]]`, `[[TB]]`… These use letters, never digits, so they can't disturb the number gate.
3. **LLM conversion** (`ROMAN_TO_HINDI_SYSTEM`): Roman Hindi → Devanagari. English words and placeholders are left unchanged, and nothing is answered or paraphrased. The §7.2 gates apply.
4. **`restore_terms`**: placeholders → the original terms in Latin script.

The standard Hindi question then goes through the same `translate_to_english` that Hindi users get.

**Protected medical terms** (`hinglish/lexicon.py → PROTECTED_TERMS`): type 2 diabetes, type 1 diabetes, insulin resistance, waist circumference, Dixit Lifestyle, prediabetes, diabetes, diabetic, HbA1c, insulin, obesity, remission, BMI, PCOS, glimepiride, sulfonylurea, metformin, hypoglycaemia, hypoglycemia. They are masked on input, and required in any Hinglish answer whose English source uses them.

### 8.3 Hinglish answers

`translate_to_hinglish` translates the same canonical English answer with the same strict rules as Hindi: translate everything; keep certainty, tense, numbers, drug names and referral roles. The §7.2 gates apply, then `back_translate_hinglish` → `parity_report`, one retry, and fail-closed exactly as for Hindi. The checked translation is stored as `entry["hinglish"]`.

Example (live run): all four of these reach the canonical question *"Can I eat fruit if I have diabetes?"* and the same chunks (49, 24, 62, 54):
- EN: *Can I eat fruits if I have diabetes?*
- HI: *अगर मुझे डायबिटीज है तो क्या मैं फल खा सकता हूँ?*
- HG: *Mujhe diabetes hai, kya main fruits kha sakta hu?*
- HG: *kya mai fruit kha sakta hu diabetes me*

---

## 9. Evaluation

### 9.1 English/Hindi: `evaluate.py`

`python evaluate.py [--ids Q3 Q14]` runs the 15 pairs in `eval_set.json` (6 safety-critical) through `compare_multilingual_questions` with the baseline, and scores both with `parity_report`. It prints a summary and writes `eval_results.csv`.

### 9.2 Trilingual: `evaluation/`

```powershell
python -m evaluation.run_trilingual_eval --retrieval-only   # retrieval consistency only (cheap)
python -m evaluation.run_trilingual_eval                    # + plans, answers, hallucination, parity
python -m evaluation.run_trilingual_eval --ids T1 T2 --no-legacy
```

The dataset `evaluation/trilingual_eval_set.json` has 14 EN/HI/Hinglish triples (5 safety-critical) plus Hinglish spelling variants. Reports go to `reports/trilingual_report.{md,json,csv}`. Full mode runs `query()` **separately per language** (not Compare, which forces a shared answer), so convergence is measured rather than assumed.

| Metric (`evaluation/metrics.py`) | Definition | Pass |
|---|---|---|
| Retrieval consistency | Same canonical question as English | same question |
| Evidence overlap | Jaccard of chunk ids vs English (+ rank overlap, score drift) | ≥ 0.6 |
| Missing evidence | English chunks the other language missed | 0 |
| Plan consistency | 1.0 if identical, else mean best-match cosine of plan points (min of both directions) | ≥ 0.9 |
| Final answer similarity | MiniLM cosine of English answer vs back-translation | ≥ 0.85 |
| Hallucination rate | Share of answer sentences not grounded in evidence, plus numbers not in evidence | ≤ 0.2 |
| Parity | `parity_report` pass for Hindi and Hinglish | pass |

The report also compares Hinglish retrieval **with the normaliser** against two baselines: **raw embedding** (Hinglish text embedded directly) and the **old pipeline** (Hinglish treated as English).

### 9.3 Results

**Retrieval-only, 14 items** (2026-09-26, `reports/trilingual_report.md`):

| Hinglish retrieval | Mean overlap vs English | Passed | Missing evidence | Same canonical question |
|---|---|---|---|---|
| **With normaliser** | **1.00** | **14/14** | **0** | 10/14 |
| Old pipeline (as English) | 0.97 | 14/14 | 1 | 7/14 |
| Raw embedding | 0.21 | 4/14 | 42 | 0/14 |
| *Hindi, for reference* | 0.94 | 14/14 | 2 | 8/14 |

Hinglish was detected in 14 of 14 questions, and 0 of 14 English questions were misdetected as Hinglish.

**Full mode, T1/T2/T9** (`reports/full_sample/`): parity passed 6 of 6, and answer similarity was 0.89–1.00.

**Parity trace** (`trace_parity.py`, after the §15 fixes): insulin, BMI, pregnancy and hunger each passed 7 of 7 parity checks.

**Voice, live** (2026-09-28):
- Hindi TTS → Whisper round trip: the transcript was exactly *"मुझे हर समय भूख क्यों लगती है?"*.
- Two-turn agent conversation: the follow-up *"और अगर मैं प्रेग्नेंट हूँ तो?"* was rewritten to *"Can I follow the Dixit Lifestyle two-meal plan while taking insulin for type 2 diabetes if I am pregnant?"*, answered in Hinglish, and passed validation and parity.

Hallucination rate is a lexical check on back-translations, and it flags paraphrases. For example, T1 Hinglish scored 0.40 while being 1.00 similar to English. Treat it as a pointer for review.

---

## 10. Voice assistant and voice agent

Voice only adds input and output around `query()`. What is spoken is the validated, parity-checked `final_answer`. There is no voice-specific generation.

### 10.1 Server side (`voice.py`)

**Speech → text** (`transcribe`):
- A multipart POST to Groq's `/audio/transcriptions` with `STT_MODEL` (default `whisper-large-v3`), `verbose_json` and temperature 0, using the shared `_post_with_retries` (which retries 429, 5xx and timeouts).
- `language`: `en` → `en`; `hi`/`hinglish` → `hi`; none → auto-detect.
- Whisper often labels spoken Hindustani as **Urdu** in Arabic script. In auto mode that triggers one re-transcription forced to Hindi.
- Segments with `no_speech_prob ≥ 0.6` are dropped (Whisper invents "Thank you." for silence). An empty result raises `ValueError("No speech detected")`.
- Audio is capped at 10 MB.
- Returns `{"text", "language": "en"|"hi"}`.

**Text → speech** (`speak`):
- Uses edge-tts (Microsoft neural voices, no key). Hindi uses `hi-IN-SwaraNeural`. English **and Hinglish** use `en-IN-NeerjaNeural`, because Hinglish is Roman script and an Indian-English voice reads it naturally.
- Runs `asyncio.run` inside the request thread, which is safe with `ThreadingHTTPServer`.
- At most 1,500 characters per call. Returns MP3.

**Speech text** (`speech_sentences = split_sentences(clean_for_speech(answer))`):
- `clean_for_speech` strips markdown (bold, headings, bullets, `[Chunk N]`) and ends each line with `.`, or `।` for Devanagari lines.
- `split_sentences` splits after `. ? ! ।` followed by whitespace, so decimals such as 24.9 survive. It merges pieces shorter than 25 characters and caps each piece at 1,500.
- `/api/query` returns this as `speech_sentences`, so the browser never re-implements it.

### 10.2 Browser engine (`web/voice.js`)

| Part | Behaviour |
|---|---|
| `Mic` | `getUserMedia` with echo cancellation, noise suppression and auto gain; an `AnalyserNode` for RMS level. **Calibrates** on 0.5 s of room noise: `threshold = max(3 × noise, 0.015)`, `bargeThreshold = 2 × threshold` |
| `capture` | `MediaRecorder` (webm/opus where supported). Speech starts after 200 ms above the threshold and ends after 1.2 s below it. Blips under 400 ms are discarded and recording restarts. There is a 30 s maximum, and `startTimeoutMs` returns `null` if nobody speaks. `finish()` / `cancel()` stop it early |
| `watchForBargeIn` | Fires once when the level stays above `bargeThreshold` for 300 ms |
| `Speaker` | Plays sentences in order, synthesising sentence *n+1* while *n* plays. `stop()` is instant. Optional caching for fixed phrases (fillers) |

### 10.3 Voice assistant (Ask tab, `web/app.js`)

1. 🎤 → the mic opens and calibrates → `capture` (8 s start timeout; a second click finishes early).
2. `/api/transcribe` (forced to the selected pill's language, or auto) → the transcript fills the box → `handleAsk({spoken: true})`.
3. For spoken questions the pill sets **`reply_language`**, not the input language. A transcript of Hinglish speech is written in Devanagari, so the pipeline detects the input from the text and replies in the chosen language.
4. The answer is read aloud when **Auto-read** is on (switched on automatically for spoken questions). 🔊 Listen / ⏹ Stop toggles playback.

### 10.4 Voice agent (Voice Agent tab, `web/agent.js`)

State machine: **idle → listening → hearing → thinking → speaking → listening…**

1. **Listening:** `capture` with a 30 s timeout. No speech → the conversation ends.
2. **Thinking:** `/api/transcribe` (auto) → user bubble. A short filler ("One moment, let me check that." / "एक क्षण, मैं जाँच कर रही हूँ।" / "Ek second, main check kar rahi hoon.") plays while `/api/query` runs with `{query, history, reply_language?}`.
3. **Speaking:** the agent bubble shows the answer, a **✓ verified** badge (validation and parity passed) or ⚠, the rewritten follow-up, and a trace. `speech_sentences` are played with barge-in watching. If you talk over it, playback stops and the next capture starts already in "speaking" mode, so your sentence is kept.
4. **Memory:** the last 3 turns (`{question: english_query, answer: canonical_answer}`) are kept in the browser and sent with each question, so the server stays stateless. **Clear memory** resets it.
5. **Errors:** if `/api/query` fails (including fail-closed parity), the agent speaks a fixed line ("I couldn't give a verified answer… check with your doctor") in the reply language. It never reads an unverified answer.
6. It ends on **End**, after 30 s of silence, or when you leave the tab (which releases the mic).

---

## 11. Server, HTTP API and web UI

`server.py` is a standard-library `ThreadingHTTPServer` on `127.0.0.1:$PORT` (default 8000). Each request runs in its own thread, so a slow comparison doesn't block others. It serves `web/` and the API below. Bad input → 400 `{"error"}`, and LLM or server failure → 500 `{"error"}`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/status` | – | `{"has_key", "model", "voice"}` |
| POST | `/api/query` | `{"query", "language"?, "history"? (≤ 6 × {question, answer}), "reply_language"?}` | Full trace + `speech_sentences` |
| POST | `/api/compare` | `{"english_query", "hindi_query", "hinglish_query"?, "baseline"?}` | Traces, back-translations, parity reports, optional baseline, or `status: "Question Intent Mismatch"` |
| POST | `/api/transcribe?language=auto\|en\|hi\|hinglish` | Raw audio bytes, `Content-Type` = audio MIME, ≤ 10 MB | `{"text", "language"}` |
| POST | `/api/speak` | `{"text" (≤ 1500), "language"}` | `audio/mpeg` |

Request validation lives in pure `parse_*` functions (unit-tested). The **UI** is plain HTML, CSS and JS with no build step:
- **Ask:** typed or spoken question, pipeline trace, evidence, Listen.
- **Parity Compare:** EN/HI/(HG) questions, per-language scorecards, back-translations, optional baseline column.
- **Voice Agent:** orb, reply-language pills, chat transcript with traces.

---

## 12. LLM layer

All calls go through `llm_client._call_groq` to Groq (`LLM_MODEL`, default `openai/gpt-oss-120b`):
- **Determinism:** temperature 0 and seed 42 on every call. For gpt-oss models, `reasoning_effort: low` and an extra 2,000 output tokens, so reasoning doesn't consume the answer budget.
- **Retries:** `_post_with_retries` retries 429 (waiting the time Groq suggests, up to 65 s), 5xx errors, timeouts and connection errors, up to 8 attempts. Whisper uses the same helper.
- **No silent fallback:** a cut-off (`finish_reason == "length"`) or empty output raises `LLMError`. The demo never shows fake or partial output.

| Prompt | Purpose |
|---|---|
| `TO_ENGLISH_SYSTEM` | Hindi → English (questions and back-translation), with `ENGLISH_TERMS`; literal verb mapping |
| `TO_HINDI_SYSTEM` | English → Hindi, with `HINDI_GLOSSARY`. Translate everything; keep certainty, tense and specific professionals; glossary only where the English uses that exact term; instruction repeated after the source text |
| `ROMAN_TO_HINDI_SYSTEM` | Hinglish question → Devanagari (placeholders kept) |
| `TO_HINGLISH_SYSTEM`, `HINGLISH_TO_ENGLISH_SYSTEM` | Answer → Hinglish, and back-translation |
| `CANONICAL_QUERY_SYSTEM` | One standard English question per meaning |
| `CONTEXTUALIZE_SYSTEM` | Follow-up → standalone question (voice agent) |
| `ANSWER_PLAN_SYSTEM` | Evidence → JSON Answer Plan |
| `CANONICAL_ANSWER_SYSTEM` | Plan + evidence → English answer |
| `DIRECT_ANSWER_SYSTEM` | Baseline only: answer directly in the user's language |

The **glossaries** (`HINDI_GLOSSARY`, `ENGLISH_TERMS`) are bidirectional and a clinical asset. Specific terms decide whether advice survives translation (e.g. नाश्ता means breakfast, not snacking).

---

## 13. Answer store

`data/answer_store.json`, written atomically (temp file + `os.replace`) under a process-wide lock:

```json
{
  "fingerprint": "e37441d3f1821883",
  "queries":          { "<normalised English wording>": "<canonical question>" },
  "hinglish_queries": { "<normalised Hinglish>": { "normalised", "terms", "standard_hindi", "attempts" } },
  "answers": {
    "<canonical key>": {
      "retrieved_chunks": [...], "answer_plan": {...}, "canonical_answer": "...",
      "validation": {...}, "attempts": 1,
      "hindi":    { "answer", "back_translation", "parity", "attempts" },
      "hinglish": { "answer", "back_translation", "parity", "attempts" }
    }
  }
}
```

- **Fingerprint** = hash of the model, seed, every prompt, both glossaries, `EMBEDDING_MODEL`, `TOP_K` and `chunks.json`. Any change discards the store automatically. `validator.py` is **not** part of it, so delete the file by hand after changing a check.
- Only answers that **passed validation**, and translations that **passed parity**, are stored. A failed one is shown but rebuilt on the next request.
- Voice-agent follow-ups use the same store: a rewritten question that canonicalises to a known question reuses its answer.

---

## 14. Tests

`python -m unittest test_parity.py test_hinglish.py test_voice.py`: **81 offline tests** that need no key, microphone or network (LLM, Whisper and edge-tts are mocked, and the answer store is a temp file).

| File | Tests | Covers |
|---|---|---|
| `test_parity.py` | 32 | Numbers and cautions, parity report, whole-word matching, canonicalisation, answer store (fingerprint, stored only if passed), retranslation with feedback, fail-closed Hindi, Compare intent gate, Groq retries and truncation, server validation, chunking |
| `test_hinglish.py` | 28 | Detector, normaliser and masking, Hinglish output gates, Hinglish pipeline (same answer, cached conversion, fail closed), evaluation metrics, report writer, server |
| `test_voice.py` | 21 | Speech text cleaning and splitting (decimals, `।`), voice selection, Urdu → Hindi retry, silence rejection, audio cap, multipart request, TTS input limits, voice request validation, follow-up contextualisation (skipped without history), `reply_language`, fail closed with `reply_language` |

---

## 15. Engineering history: parity bugs found and fixed

These fixes were found with a **runtime trace** (`trace_parity.py`, which logs every retrieval and LLM call), not by reading code. The architecture was already correct: one retrieval, one generation, and the exact canonical string passed to the translator. **The translation itself was unfaithful.**

### Round 1 (2026-09-24)

| Sample | What the translator did | Parity effect |
|---|---|---|
| BMI | Returned the English text untouched | "Passed" only because English was compared with English, which hid the bug |
| Pregnancy | "obstetrician" → प्रसूति विशेषज्ञ ("obstetric specialist") | Doctor referral lost → safety FAIL |
| Insulin | "clinician" collapsed to "doctor"; "almost certainly need to be adjusted" → "adjusted fairly precisely" | Recommendation FAIL; certainty weakened |

**Fixes:**
- Referral roles added to both glossaries (clinician, obstetrician, paediatrician, endocrinologist, healthcare team).
- Prompt rules to keep certainty words and specific professionals, and to always output Devanagari.
- The answer is sent inside `<answer>…</answer>`.
- New `_translation_problems` gates (Devanagari share, exact numbers) with one retranslation of the same text, then fail closed.

### Round 2: hunger sample failed `recommendation_match`

| English | Hindi output | Problem |
|---|---|---|
| "or **take** medicines" | ले रहे हैं ("are taking") | Tense changed; recommendation verb lost |
| "If you have **diabetes**" | **टाइप 2** डायबिटीज | Translator added "type 2" from the glossary |
| "a **healthcare professional**" | हेल्थकेयर टीम | Meaning changed by a round-1 glossary entry |
| Heading | left in English | Not translated |

**Fixes:**
- Separate glossary entries (diabetes vs type 2 diabetes; healthcare professional vs team vs diabetes-care provider; take → लेते हैं).
- Rule: use a glossary term **only where the English uses that exact term**.
- Rule: keep tense; translate headings and bold labels.
- Literal back-translation of action verbs.
- The instruction is repeated **after** the source text, because heading-heavy answers were otherwise echoed in English, and the Devanagari gate refused them.

**Result:** insulin, BMI, pregnancy and hunger passed 7 of 7 checks. No threshold or scoring logic was changed.

### Later fixes (2026-09-25)

- **Answer-store fingerprint:** stale answers are discarded automatically.
- **LLM canonicalisation:** English and Hindi converge on one question.
- **Parity on every Hindi answer** (Ask tab too), with fail closed.

### Lessons for production

1. Generate once, translate everywhere.
2. Validate translations like any model output.
3. Fail closed on safety.
4. A passing check can hide a bug: give every check a sanity condition on its inputs.
5. The glossary is a clinical asset: version it, have a clinician review it, and rerun all samples after each change.
6. Normalise questions before retrieval.
7. Trace the runtime before changing code.
8. Fix causes, not thresholds.
9. Determinism is a setting, not a guarantee.

---

## 16. Known limitations and open risks

- **LLM output is not fully deterministic,** even at temperature 0 with a seed. Consistency comes from stored answers plus checks with a retry.
- **Keyword checks have false negatives and positives.** "Consult a healthcare professional" was marked missing when the answer said "talk with your doctor". `restriction_compliance` flags an answer that correctly *states* a restriction ("the plan is not a cure"), so that answer is not stored and is regenerated on every request (the voice agent shows ⚠). Production should add meaning-level checks (an entailment model or an LLM judge).
- **Canonical wording can still differ slightly** ("BMI cut-offs" vs "BMI cut-off"). The evidence is the same, but each wording gets its own stored answer. Suggested fix: reuse a stored answer when the chunk set is identical and the canonical questions have cosine ≥ ~0.9.
- **Hinglish:**
  - Roman → Devanagari conversion is an LLM call, so gates catch lost terms and numbers but not subtle meaning changes.
  - The spelling lexicon is finite.
  - Short, mostly-English Hinglish may be treated as English.
  - The knowledge base has no rule on fruit, so the fruit question gets a general evidence-grounded answer.
- **Voice:**
  - Hinglish speech is transcribed in Devanagari, so it gets a Hindi reply unless Hinglish is chosen.
  - The energy detector can cut in early in a noisy room, or trigger on the speaker's own voice without headphones.
  - The mic works only on `localhost` or HTTPS.
  - edge-tts needs internet access.
- **Latency and rate limits:** a new question makes about 6–10 Groq calls (50–80 s on the free tier). Stored questions take seconds.
- **Glossary coverage:** new roles, drugs or topics need entries in both directions.

---

## 17. Extending the system

| Task | What to do |
|---|---|
| **New or changed knowledge document** | `python ingest.py --source file.md\|file.pdf`. The store fingerprint includes `chunks.json`, so stored answers reset automatically |
| **New glossary term** | Add to `HINDI_GLOSSARY` **and** `ENGLISH_TERMS` in `llm_client.py`. For Hinglish, add terms that must stay verbatim to `PROTECTED_TERMS`. Rerun `python trace_parity.py` and the evaluations |
| **New Hinglish spelling or marker** | `hinglish/lexicon.py` (`SPELLING_VARIANTS`, `STRONG_MARKERS`, `WEAK_MARKERS`); add a test in `test_hinglish.py` |
| **New output language** (e.g. Marathi) | Add forward and back translators with gates in `llm_client.py`, register them in `rag_pipeline._translators` and `TRANSLATED_LANGUAGES`, add the code to `server.LANGUAGES`, a voice in `voice.VOICES`, and a pill in the UI. Retrieval, planning, answering and validation stay unchanged |
| **Changed check** | Edit `validator.py`, add a unit test, and delete `data/answer_store.json` (checks are not fingerprinted) |
| **Different voices or STT model** | `.env`: `TTS_VOICE_EN`, `TTS_VOICE_HI`, `STT_MODEL` (list edge-tts voices with `edge-tts --list-voices`) |
| **Different LLM** | `.env`: `LLM_MODEL` (any Groq chat model). The fingerprint change resets stored answers |
