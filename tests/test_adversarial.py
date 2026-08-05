from __future__ import annotations

import unittest
from pathlib import Path

from ste_reward import RewardConfig, score_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = PROJECT_ROOT / "policy" / "glossary.yaml"


class RewardAdversarialTests(unittest.TestCase):
    def test_single_uppercase_nonce_word_cannot_bypass_dictionary(self) -> None:
        result = score_text("ZORP.", text_type="description")
        self.assertFalse(result.compliant)
        self.assertIn("HARNESS-ALL-CAPS-BYPASS", result.rule_counts)

    def test_punctuation_only_output_fails(self) -> None:
        result = score_text("...?!")
        self.assertEqual(-1.0, result.reward)
        self.assertIn("HARNESS-NO-LEXICAL-CONTENT", result.rule_counts)

    def test_numeric_only_output_fails(self) -> None:
        result = score_text("12345")
        self.assertEqual(-1.0, result.reward)
        self.assertIn("HARNESS-NO-LEXICAL-CONTENT", result.rule_counts)

    def test_non_latin_output_fails_english_content_gate(self) -> None:
        result = score_text("你好。")
        self.assertEqual(-1.0, result.reward)
        self.assertIn("HARNESS-NO-LEXICAL-CONTENT", result.rule_counts)

    def test_lowercase_nonce_words_fail_dictionary(self) -> None:
        result = score_text("Flurble the wuggle zorp.", text_type="description")
        self.assertFalse(result.compliant)
        self.assertIn("STE-VOCAB-UNAPPROVED", result.rule_counts)

    def test_invalid_long_output_never_gets_positive_reward(self) -> None:
        text = "The door is open. " * 7
        result = score_text(text, text_type="description")
        self.assertFalse(result.strict_compliant)
        self.assertLessEqual(result.reward, 0.0)
        self.assertIn("STE-PARAGRAPH-LENGTH", result.rule_counts)

    def test_warning_is_negative_under_default_strict_policy(self) -> None:
        result = score_text(
            "Open the door and close the door.",
            text_type="procedure",
        )
        self.assertTrue(result.compliant)
        self.assertFalse(result.strict_compliant)
        self.assertLess(result.reward, 0.0)

    def test_warning_can_receive_partial_reward_under_soft_policy(self) -> None:
        result = score_text(
            "Open the door and close the door.",
            text_type="procedure",
            config=RewardConfig(strict=False),
        )
        self.assertTrue(result.compliant)
        self.assertGreater(result.reward, 0.0)
        self.assertLess(result.reward, 1.0)

    def test_project_technical_terms_are_accepted(self) -> None:
        result = score_text(
            "The language model has training data.",
            text_type="description",
            glossary_path=GLOSSARY,
        )
        self.assertTrue(result.strict_compliant, result.findings)
        self.assertEqual(1.0, result.reward)


if __name__ == "__main__":
    unittest.main()
