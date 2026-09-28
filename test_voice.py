"""Offline unit tests for the voice assistant and voice agent — no Groq key, microphone or network needed.

Run: python -m unittest test_voice.py
"""

import unittest
from unittest.mock import MagicMock, patch

import rag_pipeline
import server
import voice
from llm_client import LLMError
from test_hinglish import HG_ANSWER
from test_parity import GOOD_BACK, HI_ANSWER, PipelineTestCase, _entry

MARKDOWN_ANSWER = """**BMI cut-offs for Indian adults**

- A BMI of 23 to 24.9 is overweight [Chunk 3]
- 25 or above is obese

Please consult your doctor."""


class SpeechTextTests(unittest.TestCase):
    def test_markdown_becomes_plain_sentences(self):
        text = voice.clean_for_speech(MARKDOWN_ANSWER)
        self.assertEqual(text, "BMI cut-offs for Indian adults. A BMI of 23 to 24.9 is overweight. "
                               "25 or above is obese. Please consult your doctor.")

    def test_hindi_lines_end_with_danda(self):
        self.assertEqual(voice.clean_for_speech("- डॉक्टर से सलाह लें"), "डॉक्टर से सलाह लें।")

    def test_split_keeps_decimals_and_merges_short_pieces(self):
        pieces = voice.split_sentences("Yes. A BMI of 24.9 is overweight. Ask your doctor before changing insulin.")
        self.assertEqual(pieces, ["Yes. A BMI of 24.9 is overweight.", "Ask your doctor before changing insulin."])

    def test_split_on_hindi_danda(self):
        self.assertEqual(len(voice.split_sentences(HI_ANSWER)), 2)

    def test_long_sentence_is_capped(self):
        pieces = voice.split_sentences("a" * (voice.MAX_SPEAK_CHARS + 10))
        self.assertTrue(all(len(p) <= voice.MAX_SPEAK_CHARS for p in pieces))

    def test_speech_is_the_final_answer_unchanged(self):
        # Voice must not add or drop content: every word of the answer is spoken
        spoken = " ".join(voice.speech_sentences(MARKDOWN_ANSWER))
        for word in ("23", "24.9", "25", "obese", "consult", "doctor"):
            self.assertIn(word, spoken)

    def test_voice_selection(self):
        self.assertEqual(voice.voice_for("hi"), voice.VOICES["hi"])
        self.assertEqual(voice.voice_for("en"), voice.VOICES["en"])
        self.assertEqual(voice.voice_for("hinglish"), voice.VOICES["en"])  # Roman script


def _whisper_result(text, language, no_speech=0.0):
    return {"text": text, "language": language,
            "segments": [{"text": text, "no_speech_prob": no_speech}]}


class TranscribeTests(unittest.TestCase):
    def test_urdu_detection_is_retried_as_hindi(self):
        with patch.object(voice, "_whisper", side_effect=[_whisper_result("مجھے بھوک", "urdu"),
                                                          _whisper_result("मुझे भूख लगती है", "hindi")]) as w:
            result = voice.transcribe(b"audio", "audio/webm")
        self.assertEqual(result, {"text": "मुझे भूख लगती है", "language": "hi"})
        self.assertEqual(w.call_args_list[1].args[2], "hi")

    def test_hinglish_is_forced_to_hindi_recognition(self):
        with patch.object(voice, "_whisper", return_value=_whisper_result("क्या मैं फल खा सकता हूँ", "hindi")) as w:
            voice.transcribe(b"audio", "audio/webm", "hinglish")
        self.assertEqual(w.call_args.args[2], "hi")

    def test_silence_is_rejected(self):
        with patch.object(voice, "_whisper", return_value=_whisper_result("Thank you.", "english", no_speech=0.9)):
            with self.assertRaisesRegex(ValueError, "No speech"):
                voice.transcribe(b"audio")

    def test_audio_size_is_capped(self):
        with self.assertRaises(ValueError):
            voice.transcribe(b"x" * (voice.MAX_AUDIO_BYTES + 1))

    def test_whisper_request_is_multipart_with_retry_helper(self):
        response = MagicMock(status_code=200)
        response.json.return_value = _whisper_result("hello", "english")
        with patch.object(voice, "is_api_key_valid", return_value=True), \
                patch.object(voice, "_post_with_retries", return_value=response) as post:
            voice._whisper(b"audio", "audio/webm;codecs=opus", "en")
        kwargs = post.call_args.kwargs
        self.assertEqual(post.call_args.args[1], voice.GROQ_STT_URL)
        self.assertEqual(kwargs["files"]["file"][0], "speech.webm")
        self.assertEqual(kwargs["data"]["language"], "en")


class SpeakTests(unittest.TestCase):
    def test_speak_uses_language_voice(self):
        async def fake(text, voice_name):
            return voice_name.encode()
        with patch.object(voice, "edge_tts", MagicMock()), patch.object(voice, "_synthesise", side_effect=fake):
            self.assertEqual(voice.speak("नमस्ते", "hi"), voice.VOICES["hi"].encode())

    def test_speak_rejects_empty_and_long_text(self):
        for text in ("", "a" * (voice.MAX_SPEAK_CHARS + 1)):
            with self.assertRaises(ValueError):
                voice.speak(text, "en")


class VoiceServerTests(unittest.TestCase):
    def test_voice_options_are_validated(self):
        history = [{"question": "What is BMI?", "answer": "BMI is ..."}]
        self.assertEqual(server.parse_voice_options({"history": history, "reply_language": "hinglish"}),
                         (history, "hinglish"))
        self.assertEqual(server.parse_voice_options({}), (None, None))
        for bad in ({"history": "x"}, {"history": [{"question": 1, "answer": ""}]},
                    {"history": [{"question": "q", "answer": "a"}] * 7}, {"reply_language": "fr"}):
            with self.assertRaises(ValueError):
                server.parse_voice_options(bad)

    def test_transcribe_language_parameter(self):
        self.assertIsNone(server.parse_transcribe_language("/api/transcribe"))
        self.assertIsNone(server.parse_transcribe_language("/api/transcribe?language=auto"))
        self.assertEqual(server.parse_transcribe_language("/api/transcribe?language=hi"), "hi")
        with self.assertRaises(ValueError):
            server.parse_transcribe_language("/api/transcribe?language=fr")

    def test_speak_body_is_validated(self):
        self.assertEqual(server.parse_speak_request({"text": " Hi ", "language": "hi"}), ("Hi", "hi"))
        for bad in ([], {"text": ""}, {"text": "x", "language": "fr"}, {"text": "a" * 2000}):
            with self.assertRaises(ValueError):
                server.parse_speak_request(bad)


class ConversationPipelineTests(PipelineTestCase):
    HISTORY = [{"question": "Can I follow the Dixit plan while taking insulin?", "answer": "Consult your doctor."}]

    def test_no_history_means_no_contextualisation(self):
        with patch.object(rag_pipeline, "contextualize_question") as ctx, \
                patch.object(rag_pipeline, "_answer_for", return_value=(_entry(), False)):
            trace = rag_pipeline.query("Why am I hungry?")
        ctx.assert_not_called()
        self.assertNotIn("standalone_query", trace)

    def test_follow_up_is_rewritten_before_canonicalisation(self):
        standalone = "Can I follow the Dixit plan while pregnant?"
        with patch.object(rag_pipeline, "contextualize_question", return_value=standalone) as ctx, \
                patch.object(rag_pipeline, "_answer_for", return_value=(_entry(), False)) as answer:
            trace = rag_pipeline.query("And if I'm pregnant?", history=self.HISTORY)
        ctx.assert_called_once_with("And if I'm pregnant?", self.HISTORY)
        self.assertEqual(trace["standalone_query"], standalone)
        self.assertEqual(answer.call_args.args[0], rag_pipeline._canonical_question(standalone))

    def test_hindi_speech_can_be_answered_in_hinglish(self):
        translation = {"answer": HG_ANSWER, "back_translation": GOOD_BACK,
                       "parity": {"parity_passed": True, "checks": []}, "reused": False}
        with patch.object(rag_pipeline, "translate_to_english", return_value="Why am I hungry?"), \
                patch.object(rag_pipeline, "_answer_for", return_value=(_entry(), False)), \
                patch.object(rag_pipeline, "_hinglish_for", return_value=translation) as hinglish:
            trace = rag_pipeline.query("मुझे भूख क्यों लगती है?", reply_language="hinglish")
        hinglish.assert_called_once()
        self.assertEqual(trace["detected_language"], "hi")
        self.assertEqual(trace["output_language"], "hinglish")
        self.assertEqual(trace["final_answer"], HG_ANSWER)

    def test_reply_language_still_fails_closed(self):
        failed = {"parity_passed": False, "checks": [{"name": "numbers_match", "passed": False, "details": ""}]}
        translation = {"answer": "…", "back_translation": "…", "parity": failed, "reused": False}
        with patch.object(rag_pipeline, "_answer_for", return_value=(_entry(), False)), \
                patch.object(rag_pipeline, "_hindi_for", return_value=translation):
            with self.assertRaises(LLMError):
                rag_pipeline.query("Why am I hungry?", reply_language="hi")


if __name__ == "__main__":
    unittest.main()
