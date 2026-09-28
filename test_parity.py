"""Offline unit tests for the parity logic — no Groq key or network needed.

Run: python -m unittest test_parity.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import llm_client
import rag_pipeline
import server
from ingest import _split_long_text, chunk_document
from llm_client import LLMError, normalise_question
from validator import (
    _recommendation_terms,
    detect_cautions,
    extract_numbers,
    parity_report,
    safety_caution_check,
    validate_answer,
)

EN_ANSWER = (
    "For Indian adults, a BMI of 23 to 24.9 is overweight and 25 or above is obese. "
    "Indians carry more visceral fat and develop insulin resistance at lower BMI values. "
    "Please consult your doctor for personalised advice."
)
HI_ANSWER = (
    "भारतीय वयस्कों के लिए 23 से 24.9 का बीएमआई अधिक वजन और 25 या उससे अधिक मोटापा है। "
    "कृपया अपने डॉक्टर से परामर्श लें।"
)
GOOD_BACK = (
    "For Indian adults, a BMI of 23 to 24.9 is overweight and 25 or above is obese. "
    "Indians have more visceral fat and develop insulin resistance at lower BMI values. "
    "Please consult your doctor for personalised advice."
)

INSULIN_Q = "I take insulin for type 2 diabetes. Can I follow the Dixit two-meal plan?"
INSULIN_CHUNKS = [{"id": 1, "text": "People on insulin or sulfonylurea medication risk hypoglycaemia "
                                    "and should consult their physician before fasting gaps."}]


def _check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


class NumberAndCautionTests(unittest.TestCase):
    def test_extract_numbers_normalises_devanagari_digits(self):
        self.assertEqual(extract_numbers("BMI २३ to 24.9."), {"23", "24.9"})

    def test_detect_cautions(self):
        self.assertEqual(detect_cautions("I am pregnant and on insulin"),
                         {"pregnancy", "diabetes_medication"})
        self.assertEqual(detect_cautions("What is a BMI?"), set())

    def test_safety_check_fails_without_caution(self):
        result = safety_caution_check("Eat two meals a day and walk daily.", INSULIN_Q, INSULIN_CHUNKS)
        self.assertFalse(result["passed"])

    def test_safety_check_passes_with_caution_and_referral(self):
        answer = "Because you take insulin, skipping meals can cause low blood sugar. Talk to your doctor first."
        self.assertTrue(safety_caution_check(answer, INSULIN_Q, INSULIN_CHUNKS)["passed"])

    def test_validation_feedback_asks_for_missing_caution(self):
        result = validate_answer("Eat two meals a day and walk daily for insulin.", {}, INSULIN_CHUNKS,
                                 question=INSULIN_Q)
        self.assertFalse(result["passed"])
        self.assertIn("doctor", result["feedback"])


class ParityReportTests(unittest.TestCase):
    def test_faithful_translation_passes(self):
        report = parity_report(EN_ANSWER, HI_ANSWER, GOOD_BACK, [1, 2, 3], [1, 2, 3])
        self.assertTrue(report["parity_passed"], report)

    def test_dropped_number_fails(self):
        report = parity_report(EN_ANSWER, "भारतीय वयस्कों के लिए 25 या अधिक मोटापा है।", GOOD_BACK)
        self.assertFalse(_check(report, "numbers_match")["passed"])

    def test_dropped_referral_fails(self):
        back = "For Indian adults, a BMI of 23 to 24.9 is overweight and 25 or above is obese."
        report = parity_report(EN_ANSWER, HI_ANSWER, back)
        self.assertFalse(_check(report, "cautions_preserved")["passed"])
        self.assertFalse(_check(report, "content_coverage")["passed"])

    def test_different_retrieval_fails(self):
        report = parity_report(EN_ANSWER, HI_ANSWER, GOOD_BACK, [1, 2, 3], [7, 8, 9])
        self.assertFalse(_check(report, "retrieval_overlap")["passed"])


class WordMatchingTests(unittest.TestCase):
    def test_kidney_is_not_a_child_caution(self):
        self.assertEqual(detect_cautions("Obesity raises the risk of kidney disease."), set())
        self.assertEqual(detect_cautions("Kids and adolescents need a paediatrician."), {"children"})

    def test_insulin_resistance_is_not_medication(self):
        self.assertEqual(detect_cautions("Insulin resistance develops at a lower BMI."), set())
        self.assertEqual(detect_cautions("If you take insulin, talk to your doctor."), {"diabetes_medication"})

    def test_recommendation_markers_match_whole_words(self):
        self.assertEqual(_recommendation_terms("Your sugar intake matters."), set())
        self.assertEqual(_recommendation_terms("Take your medicine and keep walking."), {"take", "walk"})


def _entry(passed=True, answer="Feeling hungry all the time can happen with poor sleep, stress, or blood sugar swings."):
    return {
        "retrieved_chunks": [{"id": 1, "text": "hunger can be due to poor sleep or stress", "heading": "hunger"}],
        "answer_plan": {"required_points": [], "important_conditions": [], "restrictions": [], "evidence": []},
        "canonical_answer": answer,
        "validation": {"passed": passed, "checks": [], "feedback": "", "score": 1.0},
        "attempts": 1,
    }


class PipelineTestCase(unittest.TestCase):
    """Runs the pipeline against a throwaway answer store; the canonicaliser is the offline normaliser."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store_path = Path(tmp.name) / "answer_store.json"
        for p in (patch.object(rag_pipeline, "ANSWER_STORE_PATH", self.store_path),
                  patch.object(rag_pipeline, "_fingerprint", return_value="test"),
                  patch.object(rag_pipeline, "canonicalize_query", side_effect=normalise_question)):
            p.start()
            self.addCleanup(p.stop)
        rag_pipeline._cache.pop("store", None)
        self.addCleanup(rag_pipeline._cache.pop, "store", None)


class CanonicalQuestionTests(PipelineTestCase):
    def test_same_meaning_hunger_questions_share_canonical_form(self):
        self.assertEqual(normalise_question("why do i keep feeling hungry all the time"),
                         normalise_question("Why do I feel hungry all the time?"))

    def test_contractions_and_punctuation_are_normalised(self):
        self.assertEqual(normalise_question("What's BMI?"), normalise_question("what is BMI"))

    def test_llm_canonicaliser_runs_once_per_wording(self):
        with patch.object(rag_pipeline, "canonicalize_query", return_value="What is BMI?") as canon:
            for q in ("What's BMI?", "what is bmi", "What is BMI"):
                self.assertEqual(rag_pipeline._canonical_question(q), "What is BMI?")
        canon.assert_called_once()

    def test_canonicalize_query_offline_returns_normalised_text(self):
        with patch.object(llm_client, "is_api_key_valid", return_value=False):
            self.assertEqual(llm_client.canonicalize_query("What's BMI?"), "what is bmi")


class AnswerStoreTests(PipelineTestCase):
    def test_failed_validation_is_not_stored(self):
        with patch.object(rag_pipeline, "_build_canonical_answer", return_value=_entry(passed=False)) as build:
            rag_pipeline._answer_for("why am i hungry")
            _, reused = rag_pipeline._answer_for("why am i hungry")
        self.assertFalse(reused)
        self.assertEqual(build.call_count, 2)

    def test_passed_validation_is_stored_and_reused(self):
        with patch.object(rag_pipeline, "_build_canonical_answer", return_value=_entry()) as build:
            rag_pipeline._answer_for("why am i hungry")
            _, reused = rag_pipeline._answer_for("Why am I hungry?")
        self.assertTrue(reused)
        build.assert_called_once()

    def test_store_with_other_fingerprint_is_ignored(self):
        self.store_path.write_text(json.dumps({"fingerprint": "old", "queries": {},
                                               "answers": {"why am i hungry": _entry()}}), encoding="utf-8")
        self.assertIsNone(rag_pipeline._store_get("answers", "why am i hungry"))

    def test_old_flat_store_format_is_ignored(self):
        self.store_path.write_text(json.dumps({"why am i hungry": _entry(passed=False)}), encoding="utf-8")
        self.assertIsNone(rag_pipeline._store_get("answers", "why am i hungry"))


class HindiTranslationTests(PipelineTestCase):
    def test_parity_failure_retranslates_once_with_feedback(self):
        lost_referral = "For Indian adults, a BMI of 23 to 24.9 is overweight and 25 or above is obese."
        with patch.object(rag_pipeline, "translate_to_hindi", return_value=HI_ANSWER) as to_hi, \
                patch.object(rag_pipeline, "translate_to_english", side_effect=[lost_referral, GOOD_BACK]):
            result = rag_pipeline._translate_checked(EN_ANSWER)
        self.assertTrue(result["parity"]["parity_passed"])
        self.assertEqual(result["attempts"], 2)
        self.assertIn("cautions_preserved", to_hi.call_args_list[1].kwargs["feedback"])

    def test_hindi_losing_a_caution_is_not_shown(self):
        failed = {"parity_passed": False, "checks": [{"name": "cautions_preserved", "passed": False, "details": ""}]}
        hindi = {"answer": "…", "back_translation": "…", "parity": failed, "attempts": 2, "reused": False}
        with patch.object(rag_pipeline, "translate_to_english", return_value="Why am I hungry?"), \
                patch.object(rag_pipeline, "_answer_for", return_value=(_entry(), False)), \
                patch.object(rag_pipeline, "_hindi_for", return_value=hindi):
            with self.assertRaises(LLMError):
                rag_pipeline.query("मुझे भूख क्यों लगती है?")

    def test_checked_translation_is_stored_and_reused(self):
        passed = {"answer": HI_ANSWER, "back_translation": GOOD_BACK,
                  "parity": {"parity_passed": True, "checks": []}, "attempts": 1}
        with patch.object(rag_pipeline, "_build_canonical_answer", return_value=_entry(answer=EN_ANSWER)), \
                patch.object(rag_pipeline, "_translate_checked", return_value=passed) as translate:
            entry, _ = rag_pipeline._answer_for("bmi cutoffs")
            first = rag_pipeline._hindi_for("bmi cutoffs", entry)
            entry, _ = rag_pipeline._answer_for("bmi cutoffs")
            second = rag_pipeline._hindi_for("bmi cutoffs", entry)
        translate.assert_called_once()
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])


class CompareTests(PipelineTestCase):
    @patch.object(rag_pipeline, "_answer_for")
    @patch.object(rag_pipeline, "question_similarity", return_value=0.3)
    @patch.object(rag_pipeline, "translate_to_english", return_value="How much should I walk daily?")
    def test_different_intent_is_rejected(self, _translate, _similarity, answer_for):
        result = rag_pipeline.compare_multilingual_questions(
            "What are the revised BMI cutoffs for Indian adults?", "मैं रोज़ कितना चलूँ?")
        self.assertEqual(result["status"], rag_pipeline.INTENT_MISMATCH)
        answer_for.assert_not_called()

    @patch.object(rag_pipeline, "question_similarity", return_value=0.95)
    @patch.object(rag_pipeline, "translate_to_hindi", return_value="मैं भूखा हूँ")
    @patch.object(rag_pipeline, "_build_canonical_answer", return_value=_entry())
    def test_compare_uses_one_canonical_answer_for_hindi_translation(self, _build, to_hi, _similarity):
        back = ["why do i keep feeling hungry all the time", "I am hungry", "I am hungry"]
        with patch.object(rag_pipeline, "translate_to_english", side_effect=back):
            result = rag_pipeline.compare_multilingual_questions(
                "why do i feel hungry all the time", "मुझे हर समय भूख क्यों लगती है?")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["same_canonical_question"])
        self.assertEqual(result["english"]["canonical_answer"], result["hindi"]["translated_from"])
        for call in to_hi.call_args_list:
            self.assertEqual(call.args[0], result["english"]["canonical_answer"])
        self.assertNotIn("query_similarity", [c["name"] for c in result["parity"]["checks"]])


def _groq_response(status=200, finish_reason="stop", content="ok"):
    resp = MagicMock(status_code=status, text="", headers={})
    resp.json.return_value = {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}
    resp.raise_for_status.return_value = None
    return resp


class GroqCallTests(unittest.TestCase):
    def _call(self, *responses):
        client = MagicMock()
        client.__enter__.return_value.post.side_effect = list(responses)
        with patch.object(llm_client, "is_api_key_valid", return_value=True), \
                patch.object(llm_client.httpx, "Client", return_value=client), \
                patch.object(llm_client.time, "sleep"):
            return llm_client._call_groq([{"role": "user", "content": "hi"}]), client

    def test_truncated_output_raises(self):
        with self.assertRaises(LLMError):
            self._call(_groq_response(finish_reason="length"))

    def test_empty_output_raises(self):
        with self.assertRaises(LLMError):
            self._call(_groq_response(content=None))

    def test_server_error_is_retried(self):
        text, client = self._call(_groq_response(status=503), _groq_response(content="answer"))
        self.assertEqual(text, "answer")
        self.assertEqual(client.__enter__.return_value.post.call_count, 2)


class ServerRequestTests(unittest.TestCase):
    def test_query_body_is_validated(self):
        self.assertEqual(server.parse_query_request({"query": " hi ", "language": "hi"}), ("hi", "hi"))
        for bad in ([], {"query": 5}, {"query": ""}, {"query": "x", "language": "fr"}):
            with self.assertRaises(ValueError):
                server.parse_query_request(bad)

    def test_compare_body_is_validated(self):
        self.assertEqual(server.parse_compare_request({"english_query": "a", "hindi_query": "b"}), ("a", "b", False, None))
        with self.assertRaises(ValueError):
            server.parse_compare_request({"english_query": "a", "hindi_query": None})

class ChunkingTests(unittest.TestCase):
    def test_long_text_split_does_not_explode(self):
        text = ("This is a sentence about obesity. " * 150).strip()
        pieces = _split_long_text(text, max_size=2500, overlap=200)
        self.assertLessEqual(len(pieces), 3)

    def test_heading_hierarchy_includes_parts(self):
        doc = "# Title\n\n## PART I\n\n### 1. Topic\nBody text here.\n"
        chunks = chunk_document(doc)
        self.assertEqual(chunks[0]["heading"], "PART I > 1. Topic")


if __name__ == "__main__":
    unittest.main()
