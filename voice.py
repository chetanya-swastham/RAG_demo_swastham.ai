"""
voice.py — Speech in and speech out around the existing answer path.

    transcribe(): audio → text with Groq Whisper (same GROQ_API_KEY as the LLM calls)
    speak():      text → MP3 with edge-tts (Microsoft neural voices, no key needed)

Voice never generates content: the text that is spoken is the pipeline's validated,
parity-checked final_answer, cleaned of markdown and split into sentences so the browser
can start playing the first sentence while the next one is synthesised.
"""

import asyncio
import os
import re

import httpx

from llm_client import LLMError, _post_with_retries, get_api_key, is_api_key_valid

try:
    import edge_tts
except ImportError:  # voice output is optional; the text app still works without it
    edge_tts = None

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

VOICES = {
    "hi": os.getenv("TTS_VOICE_HI", "hi-IN-SwaraNeural"),
    # Hinglish is Roman script: an Indian-English voice reads it far better than a Hindi voice
    "en": os.getenv("TTS_VOICE_EN", "en-IN-NeerjaNeural"),
}

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # Groq accepts 25 MB; a spoken question is far smaller
MAX_SPEAK_CHARS = 1500

# Whisper language names → pipeline language codes
_WHISPER_LANGUAGES = {"english": "en", "en": "en", "hindi": "hi", "hi": "hi"}
# Whisper often labels spoken Hindustani as Urdu and writes it in Arabic script
_URDU = {"urdu", "ur"}
_ARABIC_SCRIPT = re.compile(r"[؀-ۿ]")
# Whisper invents text ("Thank you.") for silence; segments it rates as probably silent are dropped
NO_SPEECH_THRESHOLD = 0.6


def is_available() -> bool:
    """True when both speech directions can work: a Groq key for Whisper and edge-tts installed."""
    return edge_tts is not None and is_api_key_valid()


def voice_for(language: str) -> str:
    return VOICES["hi"] if language == "hi" else VOICES["en"]


# ---------------------------------------------------------------------------
# Speech → text
# ---------------------------------------------------------------------------

def _whisper(audio: bytes, mime: str, language: str | None) -> dict:
    if not is_api_key_valid():
        raise LLMError("GROQ_API_KEY is not set in .env — add a valid Groq key and restart.")
    extension = (mime.split("/")[-1].split(";")[0] or "webm").strip()
    data = {"model": STT_MODEL, "response_format": "verbose_json", "temperature": "0"}
    if language:
        data["language"] = language
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = _post_with_retries(client, GROQ_STT_URL, {"Authorization": f"Bearer {get_api_key()}"},
                                      files={"file": (f"speech.{extension}", audio, mime)}, data=data)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"Speech recognition failed ({e.response.status_code}): {e.response.text[:300]}") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Speech recognition failed: {e}") from e


def _spoken_text(result: dict) -> str:
    segments = result.get("segments")
    if segments:
        return " ".join(s.get("text", "").strip() for s in segments
                        if s.get("no_speech_prob", 0) < NO_SPEECH_THRESHOLD).strip()
    return (result.get("text") or "").strip()


def transcribe(audio: bytes, mime: str = "audio/webm", language: str | None = None) -> dict:
    """Audio → {"text", "language"} where language is "en" or "hi".

    language: "en", "hi" or "hinglish" forces the recognition language (Hinglish speech is
    transcribed as Devanagari Hindi, which the pipeline already handles); None auto-detects.
    """
    if not audio:
        raise ValueError("No audio received")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError(f"Audio is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB")

    forced = {"en": "en", "hi": "hi", "hinglish": "hi"}.get(language or "")
    result = _whisper(audio, mime, forced)
    detected = (result.get("language") or forced or "").lower()
    text = _spoken_text(result)
    if not forced and (detected in _URDU or _ARABIC_SCRIPT.search(text)):
        result = _whisper(audio, mime, "hi")
        detected, text = "hi", _spoken_text(result)

    if not text:
        raise ValueError("No speech detected")
    return {"text": text, "language": _WHISPER_LANGUAGES.get(detected, "en")}


# ---------------------------------------------------------------------------
# Text → speech
# ---------------------------------------------------------------------------

_MARKDOWN_LINE = re.compile(r"^\s*(?:#{1,6}\s+|[•\-*]\s+|\d+[.)]\s+)")
_SENTENCE_END = re.compile(r"(?<=[.?!।])\s+")
MIN_SENTENCE_CHARS = 25  # shorter pieces are merged with the next, so there are fewer TTS requests


def clean_for_speech(text: str) -> str:
    """Markdown answer → plain sentences a voice can read (headings and bullets become sentences)."""
    lines = []
    for line in (text or "").splitlines():
        line = _MARKDOWN_LINE.sub("", line)
        line = re.sub(r"\[Chunk \d+\]", "", line)
        line = re.sub(r"[*_`#>|]+", "", line).strip()
        if not line:
            continue
        if line[-1] not in ".?!।:;,":
            line += "।" if re.search(r"[ऀ-ॿ]", line) else "."
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_sentences(text: str) -> list[str]:
    """Plain text → speakable pieces of at least MIN_SENTENCE_CHARS and at most MAX_SPEAK_CHARS."""
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(text.strip()):
        current = f"{current} {sentence}".strip()
        if len(current) >= MIN_SENTENCE_CHARS:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)
    return [p[i:i + MAX_SPEAK_CHARS] for p in pieces for i in range(0, len(p), MAX_SPEAK_CHARS)]


def speech_sentences(answer: str) -> list[str]:
    return split_sentences(clean_for_speech(answer))


async def _synthesise(text: str, voice: str) -> bytes:
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def speak(text: str, language: str) -> bytes:
    """Text → MP3 bytes in the voice for `language` ("en", "hi" or "hinglish")."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > MAX_SPEAK_CHARS:
        raise ValueError(f"text must be at most {MAX_SPEAK_CHARS} characters; send one sentence at a time")
    if edge_tts is None:
        raise LLMError("edge-tts is not installed: run pip install -r requirements.txt")
    try:
        audio = asyncio.run(_synthesise(text, voice_for(language)))
    except Exception as e:  # network or service errors from edge-tts
        raise LLMError(f"Speech synthesis failed: {e}") from e
    if not audio:
        raise LLMError("Speech synthesis returned no audio.")
    return audio
