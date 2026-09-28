"""Offline unit tests for Hinglish support and the trilingual evaluation — no Groq key or network needed.

Run: python -m unittest test_hinglish.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llm_client
import rag_pipeline
import server
from evaluation import metrics
from evaluation.report import summarise, write_reports
from hinglish import detect_hinglish, find_protected_terms, mask_terms, normalise_hinglish, restore_terms
from hinglish import normalizer
from llm_client import LLMError
from test_parity import EN_ANSWER, GOOD_BACK, PipelineTestCase, _entry

GIVEN_INPUTS = [
    "kya mai fruit kha sakta hu",
    "kya main fruits kha sakta hoon",
    "mujhe diabetes hai can i eat fruit",
    "exercise karne ke pehle snack lena chahiye kya",
    "Mujhe diabetes hai, kya main fruits kha sakta hu?",
]
ENGLISH_INPUTS = [
    "Can I eat fruits if I have diabetes?",
    "What is the main cause of obesity?",
    "Is sugar bad for me?",
    "I am so hungry, to be honest",
    "Should I have a snack before exercising?",
    "I take insulin for type 2 diabetes. Can I follow the Dixit two-meal plan?",
]
HG_ANSWER = ("Indian adults ke liye 23 se 24.9 ka BMI overweight hai aur 25 ya usse zyada obese hai. "
             "Indians mein visceral fat zyada hota hai aur kam BMI par hi insulin resistance ho jata hai. "
             "Personal salah ke liye apne doctor se zaroor consult karein.")
DATASET = Path(__file__).parent / "evaluation" / "trilingual_eval_set.json"


class DetectorTests(unittest.TestCase):
    def test_given_inputs_are_hinglish(self):
        for text in GIVEN_INPUTS:
            self.assertTrue(detect_hinglish(text)["is_hinglish"], text)

    def test_english_is_not_hinglish(self):
        for text in ENGLISH_INPUTS:
            self.assertFalse(detect_hinglish(text)["is_hinglish"], text)

    def test_devanagari_is_not_hinglish(self):
        self.assertFalse(detect_hinglish("अगर मुझे डायबिटीज है तो क्या मैं फल खा सकता हूँ?")["is_hinglish"])

    def test_every_dataset_question_is_classified_correctly(self):
        for item in json.loads(DATASET.read_text(encoding="utf-8")):
            for text in [item["hinglish"], *item["variants"]]:
                self.assertTrue(detect_hinglish(text)["is_hinglish"], text)
            self.assertFalse(detect_hinglish(item["en"])["is_hinglish"], item["en"])

    def test_pipeline_routes_hinglish_but_not_english(self):
        self.assertEqual(rag_pipeline.resolve_language(GIVEN_INPUTS[0])[0], "hinglish")
        self.assertEqual(rag_pipeline.resolve_language(ENGLISH_INPUTS[0])[0], "en")
        self.assertEqual(rag_pipeline.resolve_language("मुझे भूख क्यों लगती है?")[0], "hi")
        self.assertEqual(rag_pipeline.resolve_language(GIVEN_INPUTS[0], "en")[0], "en")


class NormaliserTests(unittest.TestCase):
    def test_spelling_variants_share_one_key(self):
        self.assertEqual(normalise_hinglish("kya mai fruit kha sakta hu"),
                         normalise_hinglish("Kya main fruit kha sakta hoon?"))
        self.assertEqual(normalise_hinglish("muje har samay bhuk kyon lagti h"),
                         normalise_hinglish("mujhe har samay bhook kyun lagti h"))

    def test_medical_terms_keep_standard_spelling_and_numbers(self):
        self.assertEqual(normalise_hinglish("Mera hba1c 6.8 hai, REMISSION possible hai?"),
                         "mera HbA1c 6.8 hai remission possible hai")

    def test_mask_and_restore_round_trip(self):
        text = normalise_hinglish("type 2 diabetes aur insulin resistance mein kya fark hai, diabetes")
        masked, mapping = mask_terms(text)
        self.assertNotIn("diabetes", masked)
        self.assertEqual(set(mapping.values()), {"type 2 diabetes", "insulin resistance", "diabetes"})
        self.assertEqual(restore_terms(masked, mapping), text)

    def test_placeholders_have_no_digits(self):
        masked, _ = mask_terms("mera HbA1c aur type 2 diabetes")
        self.assertEqual(llm_client_numbers(masked), set())

    def test_gate_rejects_lost_placeholder_changed_number_and_roman_hindi(self):
        masked, mapping = mask_terms("mujhe diabetes hai kya 2 fruits kha sakta hoon")
        good = "मुझे [[TA]] है, क्या 2 fruits खा सकता हूँ?"
        self.assertEqual(normalizer.conversion_problems(masked, good, mapping), [])
        self.assertTrue(normalizer.conversion_problems(masked, "मुझे डायबिटीज है, क्या 2 fruits खा सकता हूँ?", mapping))
        self.assertTrue(normalizer.conversion_problems(masked, "मुझे [[TA]] है, क्या 3 fruits खा सकता हूँ?", mapping))
        self.assertTrue(normalizer.conversion_problems(masked, "mujhe [[TA]] hai kya 2 fruits kha sakta hoon", mapping))

    def test_conversion_retries_once_then_fails_closed(self):
        with patch.object(llm_client, "is_api_key_valid", return_value=True), \
                patch.object(llm_client, "convert_roman_to_devanagari",
                             side_effect=["mujhe [[TA]] hai", "मुझे [[TA]] है"]) as convert:
            result = normalizer.to_standard_hindi("mujhe diabetes hai")
        self.assertEqual(result["standard_hindi"], "मुझे diabetes है")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("Roman script", convert.call_args_list[1].kwargs["feedback"])

        with patch.object(llm_client, "is_api_key_valid", return_value=True), \
                patch.object(llm_client, "convert_roman_to_devanagari", return_value="mujhe hai"):
            with self.assertRaises(LLMError):
                normalizer.to_standard_hindi("mujhe diabetes hai")

    def test_offline_returns_normalised_text(self):
        with patch.object(llm_client, "is_api_key_valid", return_value=False):
            result = normalizer.to_standard_hindi("kya mai fruit kha sakta hu")
        self.assertEqual(result["standard_hindi"], "kya main fruit kha sakta hoon")
        self.assertEqual(result["attempts"], 0)


def llm_client_numbers(text):
    from validator import extract_numbers
    return extract_numbers(text)


class HinglishOutputGateTests(unittest.TestCase):
    def test_faithful_hinglish_passes(self):
        self.assertEqual(llm_client._hinglish_problems(EN_ANSWER, HG_ANSWER), [])

    def test_devanagari_output_is_rejected(self):
        problems = llm_client._hinglish_problems(EN_ANSWER, "भारतीय वयस्कों के लिए 23 से 24.9 का BMI और 25 insulin resistance")
        self.assertTrue(any("Devanagari" in p for p in problems))

    def test_plain_english_output_is_rejected(self):
        self.assertTrue(any("plain English" in p for p in llm_client._hinglish_problems(EN_ANSWER, EN_ANSWER)))

    def test_lost_medical_term_is_rejected(self):
        lost = HG_ANSWER.replace("insulin resistance", "insulin pratirodh")
        self.assertTrue(any("insulin resistance" in p for p in llm_client._hinglish_problems(EN_ANSWER, lost)))

    def test_protected_terms_are_whole_words(self):
        self.assertEqual(find_protected_terms("prediabetes and BMI"), {"prediabetes", "BMI"})


class HinglishPipelineTests(PipelineTestCase):
    def test_hinglish_takes_the_hindi_path_and_reuses_the_english_answer(self):
        conversion = {"normalised": "mujhe bhook kyun lagti hai", "masked": "", "terms": [],
                      "standard_hindi": "मुझे भूख क्यों लगती है", "attempts": 1}
        translation = {"answer": HG_ANSWER, "back_translation": GOOD_BACK,
                       "parity": {"parity_passed": True, "checks": []}, "attempts": 1}
        with patch.object(rag_pipeline, "_build_canonical_answer", return_value=_entry()) as build, \
                patch.object(rag_pipeline, "to_standard_hindi", return_value=conversion) as to_hi, \
                patch.object(rag_pipeline, "translate_to_english",
                             return_value="Why do I feel hungry all the time?") as to_en, \
                patch.object(rag_pipeline, "_translate_checked", return_value=translation) as translate:
            en = rag_pipeline.query("Why do I feel hungry all the time?")
            hg = rag_pipeline.query("Mujhe bhook kyun lagti hai?")
            hg_again = rag_pipeline.query("mujhe bhookh kyu lagti hai")

        build.assert_called_once()                     # one canonical answer for both languages
        to_hi.assert_called_once()                     # conversion cached per normalised wording
        to_en.assert_called_with("मुझे भूख क्यों लगती है")  # the SAME translator as Hindi users
        translate.assert_called_once_with(en["canonical_answer"], "hinglish")
        self.assertEqual(hg["detected_language"], "hinglish")
        self.assertEqual(hg["english_query"], en["english_query"])
        self.assertEqual(hg["answer_plan"], en["answer_plan"])
        self.assertEqual(hg["final_answer"], HG_ANSWER)
        self.assertTrue(hg_again["hinglish_conversion_reused"])
        self.assertTrue(hg_again["translation_reused"])

    def test_hinglish_losing_a_caution_is_not_shown(self):
        failed = {"parity_passed": False, "checks": [{"name": "cautions_preserved", "passed": False, "details": ""}]}
        translation = {"answer": "…", "back_translation": "…", "parity": failed, "attempts": 2, "reused": False}
        with patch.object(rag_pipeline, "_prepare_query", return_value={"translated_query": "Why am I hungry?"}), \
                patch.object(rag_pipeline, "_answer_for", return_value=(_entry(), False)), \
                patch.object(rag_pipeline, "_hinglish_for", return_value=translation):
            with self.assertRaisesRegex(LLMError, "Hinglish"):
                rag_pipeline.query("mujhe bhook kyun lagti hai")

    def test_hinglish_parity_uses_hinglish_translators(self):
        with patch.object(rag_pipeline, "translate_to_hinglish", return_value=HG_ANSWER) as forward, \
                patch.object(rag_pipeline, "back_translate_hinglish", return_value=GOOD_BACK) as back:
            result = rag_pipeline._translate_checked(EN_ANSWER, "hinglish")
        self.assertTrue(result["parity"]["parity_passed"], result["parity"])
        forward.assert_called_once()
        back.assert_called_once_with(HG_ANSWER)

    def test_stored_hinglish_is_not_part_of_the_trace(self):
        entry = {**_entry(), "hindi": {"answer": "x"}, "hinglish": {"answer": "y"}}
        self.assertNotIn("hinglish", rag_pipeline._trace_from(entry))
        self.assertNotIn("hindi", rag_pipeline._trace_from(entry))


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        self.assertEqual(metrics.chunk_overlap([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(metrics.chunk_overlap([1, 2, 3, 4], [1, 2, 3, 9]), 0.6)
        self.assertEqual(metrics.missing_evidence([1, 2, 3, 4], [1, 2, 9]), [3, 4])
        self.assertEqual(metrics.rank_overlap([1, 2], [1, 2]), 1.0)
        self.assertLess(metrics.rank_overlap([1, 2], [2, 1]), 1.0)

    def test_retrieval_comparison_score_drift(self):
        ref = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.6}]
        other = [{"id": 1, "score": 0.7}, {"id": 5, "score": 0.5}]
        comp = metrics.retrieval_comparison(ref, other)
        self.assertEqual(comp["missing_evidence"], [2])
        self.assertAlmostEqual(comp["score_drift"], 0.1)

    def test_plan_consistency(self):
        plan = {"required_points": ["Fruits are not allowed while diabetic"], "important_conditions": [],
                "restrictions": []}
        self.assertEqual(metrics.plan_consistency(plan, dict(plan)), 1.0)
        other = {"required_points": ["Walk 45 minutes daily"], "important_conditions": [], "restrictions": []}
        self.assertLess(metrics.plan_consistency(plan, other), 0.5)

    def test_answer_similarity_lexical(self):
        self.assertGreaterEqual(metrics.answer_similarity(EN_ANSWER, GOOD_BACK)["value"], 0.85)

    def test_hallucination_rate(self):
        chunks = [{"text": "For Indian adults a BMI of 23 to 24.9 is overweight and 25 or above is obese."}]
        grounded = metrics.hallucination_rate("For Indian adults, a BMI of 23 to 24.9 is overweight.", chunks)
        self.assertEqual(grounded["rate"], 0.0)
        invented = metrics.hallucination_rate("Eating mangoes cures diabetes within 40 days completely.", chunks)
        self.assertEqual(invented["rate"], 1.0)
        self.assertEqual(invented["unsupported_numbers"], ["40"])


class ReportTests(unittest.TestCase):
    def _row(self):
        comp = {"chunk_ids": [1, 2], "overlap": 1.0, "rank_overlap": 1.0, "missing_evidence": [],
                "missing_count": 0, "score_drift": 0.0, "same_canonical": True, "query_similarity": 1.0}
        return {"id": "T1", "safety": False, "status": "ok",
                "detection": {"hinglish_detected": True, "english_misdetected": False},
                "canonical": {"en": "q", "hi": "q", "hinglish": "q"},
                "retrieval": {"hi": comp, "hinglish": comp,
                              "hinglish_raw": {**comp, "overlap": 0.2, "missing_count": 2}},
                "variants": [{"text": "x", "same_canonical": True, "overlap": 1.0}]}

    def test_summary_and_files(self):
        rows = [self._row(), {"id": "T2", "safety": True, "status": "error: boom"}]
        summary = summarise(rows)
        self.assertEqual(summary["evaluated"], 1)
        self.assertEqual(summary["retrieval"]["hinglish"]["passed"], 1)
        self.assertEqual(summary["retrieval"]["hinglish_raw"]["total_missing_evidence"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_reports(rows, "retrieval-only", Path(tmp))
            self.assertIn("normaliser on vs off", paths["md"].read_text(encoding="utf-8"))
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8"))["summary"]["errors"][0]["id"], "T2")


class ServerHinglishTests(unittest.TestCase):
    def test_hinglish_language_and_compare_field(self):
        self.assertEqual(server.parse_query_request({"query": "kya", "language": "hinglish"}), ("kya", "hinglish"))
        self.assertEqual(server.parse_compare_request({"english_query": "a", "hindi_query": "b",
                                                       "hinglish_query": " c "}), ("a", "b", False, "c"))
        with self.assertRaises(ValueError):
            server.parse_compare_request({"english_query": "a", "hindi_query": "b", "hinglish_query": 5})


if __name__ == "__main__":
    unittest.main()
