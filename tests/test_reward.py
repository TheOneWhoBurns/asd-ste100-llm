from __future__ import annotations

import unittest

from ste_reward import RewardConfig, STERewardFunction, score_batch, score_text


class RewardHarnessTests(unittest.TestCase):
    def test_short_imperative_passes_procedure_gate(self) -> None:
        result = score_text("Open the door.", text_type="procedure")
        self.assertTrue(result.strict_compliant, result.findings)
        self.assertEqual(1.0, result.reward)

    def test_passive_procedure_fails(self) -> None:
        result = score_text("The cover must be removed.", text_type="procedure")
        self.assertFalse(result.compliant)
        self.assertIn("STE-PASSIVE", result.rule_counts)
        self.assertLess(result.reward, 1.0)

    def test_start_imperative_is_not_rejected_as_a_noun(self) -> None:
        result = score_text(
            "Start the file transfer again.",
            text_type="procedure",
            allowed_terms=["file transfer"],
        )
        self.assertTrue(result.strict_compliant, result.findings)

    def test_label_boundary_does_not_create_pos_warning(self) -> None:
        result = score_text(
            "Select SUBMIT.",
            text_type="procedure",
            allowed_terms=["SUBMIT"],
        )
        self.assertTrue(result.strict_compliant, result.findings)

    def test_semicolon_is_an_error(self) -> None:
        result = score_text(
            "Remove the cover; examine the seal.",
            text_type="procedure",
        )
        self.assertFalse(result.compliant)
        self.assertEqual(1, result.rule_counts["STE-SEMICOLON"])

    def test_contraction_is_an_error(self) -> None:
        result = score_text("Do not move it if it isn't stable.")
        self.assertFalse(result.compliant)
        self.assertEqual(1, result.rule_counts["STE-CONTRACTION"])

    def test_empty_output_gets_floor_reward(self) -> None:
        result = score_text("")
        self.assertFalse(result.compliant)
        self.assertEqual(-1.0, result.reward)
        self.assertEqual(1, result.rule_counts["HARNESS-EMPTY"])

    def test_uppercase_dictionary_bypass_is_blocked(self) -> None:
        result = score_text("FLURBLE WUGGLE ZORP.", text_type="description")
        self.assertFalse(result.compliant)
        self.assertIn("HARNESS-ALL-CAPS-BYPASS", result.rule_counts)

    def test_description_paragraph_has_six_sentence_limit(self) -> None:
        text = " ".join(f"The unit is item {number}." for number in range(7))
        result = score_text(text, text_type="description")
        self.assertFalse(result.compliant)
        self.assertEqual(1, result.rule_counts["STE-PARAGRAPH-LENGTH"])

    def test_warning_can_be_soft_or_strict(self) -> None:
        text = "Open the door and close the door."
        strict = score_text(
            text,
            text_type="procedure",
            config=RewardConfig(strict=True),
        )
        soft = score_text(
            text,
            text_type="procedure",
            config=RewardConfig(strict=False),
        )
        if strict.summary["warning"]:
            self.assertFalse(strict.strict_compliant)
            self.assertTrue(soft.compliant)

    def test_batch_preserves_input_fields(self) -> None:
        rows = [
            {"id": "a", "output": "Open the door.", "text_type": "procedure"},
            {"id": "b", "output": "", "text_type": "description"},
        ]
        scored = score_batch(rows)
        self.assertEqual("a", scored[0]["id"])
        self.assertIn("ste_reward", scored[0])
        self.assertEqual(-1.0, scored[1]["ste_reward"]["reward"])

    def test_reward_is_always_bounded(self) -> None:
        samples = [
            "",
            "Open the door.",
            "The cover should have been being removed; it isn't stable.",
            "FLURBLE WUGGLE ZORP.",
        ]
        for sample in samples:
            reward = score_text(sample).reward
            self.assertGreaterEqual(reward, -1.0)
            self.assertLessEqual(reward, 1.0)

    def test_trainer_adapter_accepts_chat_completions(self) -> None:
        reward_function = STERewardFunction(text_type="procedure")
        rewards = reward_function(
            [
                [{"role": "assistant", "content": "Open the door."}],
                [{"role": "assistant", "content": "FLURBLE WUGGLE ZORP."}],
            ]
        )
        self.assertEqual([1.0, -1.0], rewards)

    def test_trainer_adapter_uses_per_row_text_type(self) -> None:
        rewards = STERewardFunction()(
            ["Open the door.", "The door is open."],
            text_type=["procedure", "description"],
        )
        self.assertEqual([1.0, 1.0], rewards)

    def test_per_example_technical_term_is_allowed(self) -> None:
        without_term = score_text("The servers are hot.", text_type="description")
        with_term = score_text(
            "The servers are hot.",
            text_type="description",
            allowed_terms=["server"],
        )
        self.assertFalse(without_term.compliant)
        self.assertTrue(with_term.strict_compliant, with_term.findings)

    def test_single_technical_term_string_is_allowed(self) -> None:
        result = score_text(
            "The server is hot.",
            text_type="description",
            allowed_terms="server",
        )
        self.assertTrue(result.strict_compliant, result.findings)

    def test_trainer_adapter_accepts_per_row_technical_terms(self) -> None:
        rewards = STERewardFunction(text_type="description")(
            ["The server is hot.", "The website is available."],
            technical_terms=[["server"], ["website"]],
        )
        self.assertEqual([1.0, 1.0], rewards)

    def test_batch_uses_row_technical_terms(self) -> None:
        result = score_batch(
            [
                {
                    "output": "The server is hot.",
                    "text_type": "description",
                    "technical_terms": ["server"],
                }
            ]
        )[0]["ste_reward"]
        self.assertTrue(result["strict_compliant"], result["findings"])


if __name__ == "__main__":
    unittest.main()
