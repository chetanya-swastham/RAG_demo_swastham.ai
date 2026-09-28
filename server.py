"""
server.py — Lightweight local HTTP server for the Swastham RAG web interface.

Serves static files from web/ and exposes:
    GET  /api/status   — Whether the LLM key is configured
    POST /api/query    — Single question → answer trace
    POST /api/compare  — English + Hindi (+ optional Hinglish) question → traces + parity reports
                         (body: {"english_query", "hindi_query", "hinglish_query"?, "baseline": bool})
    POST /api/transcribe?language=auto|en|hi|hinglish — raw audio body → {"text", "language"}
    POST /api/speak    — {"text", "language"} → audio/mpeg (one sentence at a time)

Usage:
    python server.py
    → Open http://localhost:8000
"""

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLM_MODEL, is_api_key_valid
from rag_pipeline import compare_multilingual_questions, query as run_query
import voice

WEB_DIR = Path(__file__).parent / "web"
HOST = os.getenv("HOST", "127.0.0.1")  # the deployed container sets 0.0.0.0
PORT = int(os.getenv("PORT", "8000"))
LANGUAGES = {"en", "hi", "hinglish"}
MAX_HISTORY_TURNS = 6


def _text_field(body: dict, name: str) -> str:
    value = body.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def parse_query_request(body) -> tuple[str, str | None]:
    """Validate a /api/query body → (query, language or None)."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    user_query = _text_field(body, "query")
    if not user_query:
        raise ValueError("query is required")
    language = body.get("language")
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
    return user_query, language


def parse_voice_options(body: dict) -> tuple[list[dict] | None, str | None]:
    """Validate the optional voice-agent fields of a /api/query body → (history, reply_language)."""
    history = body.get("history")
    if history is not None:
        if not isinstance(history, list) or len(history) > MAX_HISTORY_TURNS:
            raise ValueError(f"history must be a list of at most {MAX_HISTORY_TURNS} turns")
        for turn in history:
            if not (isinstance(turn, dict) and isinstance(turn.get("question"), str)
                    and isinstance(turn.get("answer"), str)):
                raise ValueError('each history turn must be {"question": str, "answer": str}')
    reply_language = body.get("reply_language")
    if reply_language is not None and reply_language not in LANGUAGES:
        raise ValueError(f"reply_language must be one of {sorted(LANGUAGES)}")
    return history or None, reply_language


def parse_transcribe_language(path: str) -> str | None:
    """?language=auto|en|hi|hinglish → forced language or None (auto-detect)."""
    language = parse_qs(urlparse(path).query).get("language", ["auto"])[0]
    if language != "auto" and language not in LANGUAGES:
        raise ValueError(f"language must be auto or one of {sorted(LANGUAGES)}")
    return None if language == "auto" else language


def parse_speak_request(body) -> tuple[str, str]:
    """Validate a /api/speak body → (text, language)."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    text = _text_field(body, "text")
    if not text:
        raise ValueError("text is required")
    if len(text) > voice.MAX_SPEAK_CHARS:
        raise ValueError(f"text must be at most {voice.MAX_SPEAK_CHARS} characters")
    language = body.get("language", "en")
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
    return text, language


def parse_compare_request(body) -> tuple[str, str, bool, str | None]:
    """Validate a /api/compare body → (english_query, hindi_query, baseline, hinglish_query or None)."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    eng, hin = _text_field(body, "english_query"), _text_field(body, "hindi_query")
    if not eng or not hin:
        raise ValueError("Both english_query and hindi_query are required")
    hinglish = _text_field(body, "hinglish_query") if body.get("hinglish_query") is not None else ""
    return eng, hin, bool(body.get("baseline")), hinglish or None


class RAGHandler(SimpleHTTPRequestHandler):
    """Serves the web UI and the JSON API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if urlparse(self.path).path == "/api/status":
            self._send_json({"has_key": is_api_key_valid(), "model": LLM_MODEL, "voice": voice.is_available()})
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/query":
            self._handle(self._query)
        elif path == "/api/compare":
            self._handle(self._compare)
        elif path == "/api/transcribe":
            self._handle(self._transcribe, raw=True)
        elif path == "/api/speak":
            self._handle(self._speak)
        else:
            self.send_error(404, "Not found")

    # --- Handlers ---

    def _query(self, body) -> dict:
        user_query, language = parse_query_request(body)
        history, reply_language = parse_voice_options(body)
        trace = run_query(user_query, language_override=language, history=history, reply_language=reply_language)
        # What the voice features read aloud: the same final_answer, as plain sentences
        trace["speech_sentences"] = voice.speech_sentences(trace["final_answer"])
        return trace

    def _compare(self, body) -> dict:
        eng, hin, baseline, hinglish = parse_compare_request(body)
        return compare_multilingual_questions(eng, hin, include_baseline=baseline, hinglish_query=hinglish)

    def _transcribe(self, audio: bytes) -> dict:
        language = parse_transcribe_language(self.path)
        return voice.transcribe(audio, self.headers.get("Content-Type", "audio/webm"), language)

    def _speak(self, body) -> bytes:
        return voice.speak(*parse_speak_request(body))

    def _handle(self, fn, raw: bool = False):
        try:
            result = fn(self._read_body() if raw else self._read_json())
            if isinstance(result, bytes):
                self._send_bytes(result, "audio/mpeg")
            else:
                self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    # --- Utility methods ---

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length > voice.MAX_AUDIO_BYTES:
            raise ValueError(f"Request body is larger than {voice.MAX_AUDIO_BYTES // (1024 * 1024)} MB")
        return self.rfile.read(length)

    def _read_json(self):
        return json.loads(self._read_body().decode("utf-8") or "{}")

    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[RAG Server] {args[0]}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not is_api_key_valid():
        print("WARNING: GROQ_API_KEY is not set in .env — questions will return an error.")

    # One thread per request: a slow comparison (rate-limit waits) no longer blocks other requests
    server = ThreadingHTTPServer((HOST, PORT), RAGHandler)
    server.daemon_threads = True
    print(f"Swastham RAG running on http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
