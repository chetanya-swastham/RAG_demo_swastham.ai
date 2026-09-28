# Swastham Multilingual Health RAG

**English · हिन्दी · Hinglish, by text or by voice: one verified answer for every language.**

A proof of concept for Swastham.ai. The same health question asked in **English**, **Hindi** or **Hinglish** (Romanised Hindi mixed with English) gets the **same answer**, with the same facts, numbers and safety warnings, and the system checks this automatically. You can type your question, speak it to a **voice assistant**, or hold a hands-free conversation with a **voice agent** that understands follow-up questions.

> How it is built, every check and threshold, the evaluation results and the bug history are in **[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

---

## Contents

1. [Why this exists](#1-why-this-exists)
2. [How it works](#2-how-it-works)
3. [Features](#3-features)
4. [Quick start](#4-quick-start)
5. [Using the app](#5-using-the-app)
6. [Command line](#6-command-line)
7. [HTTP API](#7-http-api)
8. [Configuration](#8-configuration)
9. [Testing and evaluation](#9-testing-and-evaluation)
10. [Project structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations](#12-limitations)

---

## 1. Why this exists

If an LLM answers an English question in English and the same question in Hindi directly in Hindi, it writes **two independent answers**. They drift apart: different facts, less detail, and sometimes a missing safety caution such as *"adjust your insulin dose with your doctor"*. A health product cannot allow that.

## 2. How it works

**Hindi and Hinglish never reason on their own.**

```
 English question ─────────────────────────────┐
 Hindi question ─► translate to English ───────┤
 Hinglish question ─► standard Hindi ─► English┤
 Spoken question ─► Whisper transcript ────────┘ (then as above)
                                               ▼
               canonical question ─► FAISS retrieval (English knowledge base)
                                               ▼
               Answer Plan ─► ONE canonical English answer ─► validation
                                               ▼
        English: as-is  ·  Hindi / Hinglish: checked translation of THAT answer
                                               ▼
                       shown on screen  ·  read aloud (voice)
```

1. Every question becomes **one canonical English question**, answered once from evidence retrieved from an English knowledge base (`documentRAG.md`).
2. The answer is **validated**: required points covered, grounded in evidence, safety cautions present.
3. A Hindi or Hinglish user gets a **faithful translation** of that exact answer. It is back-translated and must keep every **number**, **safety caution**, **recommendation** and its **content**. If a number or caution is lost, the answer is **not shown or spoken** (fail closed).
4. Answers are stored per canonical question, so the same question always gets the same answer, in any language.

## 3. Features

| Feature | What you get |
|---|---|
| **Trilingual Q&A** | Ask in English, Hindi or Hinglish. The language is auto-detected, or you can pick it |
| **Pipeline trace** | See the canonical question, retrieved evidence, Answer Plan, validation checks, the English source of a translation, and the parity check |
| **Parity Compare** | Ask the same question in all three languages and get a scorecard per language. Optionally compare against the old "answer directly in Hindi" baseline |
| **Voice assistant** | 🎤 on the Ask tab: speak your question and hear the answer read aloud (🔊 Listen / ⏹ Stop, Auto-read) |
| **Voice agent** | Hands-free conversation on the **Voice Agent** tab. It notices when you stop talking, remembers the last 3 turns for follow-ups ("and if I'm pregnant?"), stops when you talk over it, and replies in English, Hindi or Hinglish |
| **Safety first** | Doctor-referral and medication, pregnancy, children and eating-disorder cautions are checked in every answer and every translation |

## 4. Quick start

Requirements: **Python 3.10+**, internet access, a free **[Groq API key](https://console.groq.com/keys)**, and **Chrome or Edge** for voice.

```powershell
pip install -r requirements.txt
copy .env.example .env              # then set GROQ_API_KEY in .env

python ingest.py --source documentRAG.md    # build the search index (once)
python server.py                            # → http://localhost:8000
```

- The first run downloads the MiniLM embedding model (~90 MB).
- The same Groq key powers the LLM and speech recognition. Speech output (edge-tts) needs no key.

## 5. Using the app

### Ask Question tab
- Type a question, or click **🎤** and speak it; recording stops when you pause or click again.
- Pick **Auto-Detect**, **English**, **हिन्दी** or **Hinglish**. For typed questions this sets the question language. For spoken questions it sets the reply language.
- **🔊 Listen** reads the answer aloud. **Auto-read** turns on by itself after a spoken question.
- Open **Pipeline trace** and **Retrieved evidence** to see exactly how the answer was built and checked.

### Parity Compare tab
- Enter the question in English and Hindi (Hinglish optional), or pick a sample: *Insulin (safety)*, *BMI Cutoffs*, *Pregnancy (safety)*, *Fruits (diabetes)*.
- The scorecards show each check. Tick **Show baseline** to see the old approach scored the same way.
- Different questions are refused (*Question Intent Mismatch*).

### Voice Agent tab
1. Choose **Reply in**: Same as me, English, हिन्दी or Hinglish.
2. Press the orb or **Start conversation** and allow the microphone.
3. Ask, then pause. The orb shows **Listening → Thinking → Speaking**. It says "one moment" while it checks.
4. Ask follow-ups naturally. The bubble shows how a follow-up was understood, and a **✓ verified** badge when validation and parity passed.
5. Talk over it to interrupt. **Clear memory** starts a fresh topic. **End**, 30 s of silence, or leaving the tab ends the conversation.

### Suggested 5-minute demo
1. **Ask tab:** ask the Hindi insulin question and open the pipeline trace.
2. **Parity Compare:** run *Insulin (safety)* with **Show baseline**. The canonical pipeline passes, and the baseline is scored next to it.
3. **Voice Agent:** set Reply in: Hinglish. Say *"I take insulin for type 2 diabetes, can I follow the Dixit two-meal plan?"*, then *"aur agar main pregnant hoon to?"*. Interrupt the answer mid-sentence.

## 6. Command line

```powershell
python rag_pipeline.py "मुझे हर समय भूख क्यों लगती है?"      # ask one question
python ingest.py --source file.pdf [--model NAME] [--output-dir DIR]   # index another document
python trace_parity.py [insulin|bmi|pregnancy|hunger]       # log every retrieval/LLM call + parity
```

## 7. HTTP API

The server listens on `127.0.0.1:$PORT`. Errors return `{"error": "..."}` with 400 (bad input) or 500 (LLM or server failure).

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/status` | – | `{"has_key", "model", "voice"}` |
| `POST` | `/api/query` | `{"query", "language"?: "en"\|"hi"\|"hinglish", "history"?: [{"question", "answer"}] (≤ 6), "reply_language"?}` | Full trace: detected language, canonical question, chunks, Answer Plan, validation, `final_answer`, parity and back-translation for Hindi/Hinglish, `standalone_query` for follow-ups, `speech_sentences` |
| `POST` | `/api/compare` | `{"english_query", "hindi_query", "hinglish_query"?, "baseline"?: bool}` | Per-language traces, back-translations, parity reports, optional baseline, or `status: "Question Intent Mismatch"` |
| `POST` | `/api/transcribe?language=auto\|en\|hi\|hinglish` | Raw audio (webm/ogg/wav/mp3, ≤ 10 MB) | `{"text", "language": "en"\|"hi"}` |
| `POST` | `/api/speak` | `{"text" (≤ 1500 chars), "language"}` | `audio/mpeg` |

## 8. Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – | **Required.** LLM and speech recognition |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model for every LLM call |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Must match the model the index was built with |
| `TOP_K` | `4` | Chunks retrieved per question |
| `PORT` | `8000` | Web server port |
| `STT_MODEL` | `whisper-large-v3` | Groq speech-to-text model (`whisper-large-v3-turbo` is faster) |
| `TTS_VOICE_EN` | `en-IN-NeerjaNeural` | Voice for English and Hinglish |
| `TTS_VOICE_HI` | `hi-IN-SwaraNeural` | Voice for Hindi |

Fixed in code: temperature 0 and seed 42 on every LLM call, and automatic retries on rate limits, 5xx errors and timeouts.

Stored answers (`data/answer_store.json`) reset automatically when the model, prompts, glossaries, `TOP_K`, embedding model or knowledge base change. After editing `validator.py`, delete that file by hand.

## 9. Testing and evaluation

```powershell
python -m unittest test_parity.py test_hinglish.py test_voice.py   # 81 offline tests, no key needed
python evaluate.py [--ids Q3 Q14]                                  # 15 EN/HI pairs vs baseline → eval_results.csv
python -m evaluation.run_trilingual_eval --retrieval-only          # EN/HI/Hinglish retrieval consistency → reports/
python -m evaluation.run_trilingual_eval                           # + plans, answers, hallucination, parity
```

**Latest results:**
- 81/81 unit tests pass.
- Hinglish retrieval matches English on 14/14 questions (mean overlap 1.00).
- The insulin, BMI, pregnancy and hunger samples pass 7/7 parity checks.
- Voice was tested live: the Hindi speech round trip was transcribed exactly, and a follow-up was rewritten correctly and passed parity in Hinglish.

Details are in [IMPLEMENTATION.md §9](IMPLEMENTATION.md#9-evaluation).

## 10. Project structure

```
RAG_demo/
├── README.md                  this file
├── IMPLEMENTATION.md          design, modules, checks, evaluation, history
├── requirements.txt · .env.example
├── documentRAG.md             English knowledge base (obesity, diabetes, Dixit Lifestyle)
├── ingest.py                  chunk → embed (MiniLM) → FAISS; retrieval
├── rag_pipeline.py            the single answer path, answer store, Parity Compare, baseline
├── llm_client.py              every LLM call (Groq), prompts, glossaries, translation gates
├── validator.py               answer validation + parity checks (pure functions)
├── voice.py                   speech → text (Whisper) and text → speech (edge-tts)
├── server.py                  web server + JSON/audio API
├── hinglish/                  Hinglish detector, spelling normaliser, protected medical terms
├── web/                       index.html · style.css · app.js (Ask, Compare, voice assistant)
│                              voice.js (mic, pause detection, playback) · agent.js (voice agent)
├── evaluate.py · eval_set.json            EN/HI evaluation vs baseline
├── evaluation/                trilingual evaluation (metrics, runner, report, dataset)
├── trace_parity.py            runtime trace of every retrieval/LLM call
├── test_parity.py · test_hinglish.py · test_voice.py   offline unit tests
├── data/                      generated: chunks, FAISS index, answer store
├── reports/                   generated trilingual evaluation reports
└── Swastham_EN-HI_Parity_RAG_Presentation.pdf
```

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Add the key to `.env` and restart `server.py` |
| `FileNotFoundError` for `data/…` | Run `python ingest.py --source documentRAG.md` |
| `The index was built with … but EMBEDDING_MODEL is …` | Re-run `ingest.py`, or set `EMBEDDING_MODEL` to the model that built the index |
| `[LLM] rate limited, waiting …s` | Groq free-tier limit; it retries automatically. Space out heavy testing |
| `LLM output was cut off at the token limit` | Raise `max_tokens` for that call in `llm_client.py` |
| `The Hindi translation did not pass the parity check …` | The translation lost a number or safety caution twice, so it was blocked by design. Check it on Parity Compare |
| Mic button disabled / "Voice needs …" | Set `GROQ_API_KEY`, run `pip install -r requirements.txt`, restart the server |
| "Microphone access was blocked" | Allow the mic (lock icon in the address bar); open the app on `localhost`, not a LAN IP |
| Voice agent interrupts itself | Use headphones or lower the volume, because it hears its own voice |
| Changes don't appear | Restart `server.py`. After editing `validator.py`, also delete `data/answer_store.json` |
| Hindi shows as boxes in the terminal | Use Windows Terminal or set `PYTHONIOENCODING=utf-8` |

## 12. Limitations

- **A new question is slow** on the free Groq tier: about 6–10 LLM calls, or 50–80 s. A question it has seen before answers in seconds.
- **The checks are keyword-based,** so they have some false alarms. An answer that correctly states a restriction can fail `restriction_compliance`; it is then shown marked as failed (⚠ in the voice agent) and not stored.
- **Hinglish speech is transcribed in Devanagari,** so choose *Reply in: Hinglish* to get a Hinglish reply.
- **Pause detection is energy-based,** so noisy rooms can cut in early.
- **The system answers only from `documentRAG.md`.** Topics the document doesn't cover get a cautious, general answer.

Full list and suggested fixes: [IMPLEMENTATION.md §16](IMPLEMENTATION.md#16-known-limitations-and-open-risks).
