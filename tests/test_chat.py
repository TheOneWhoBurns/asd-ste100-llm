from __future__ import annotations

import unittest

from scripts.chat import clean_response, prompt_terms


class ChatHelperTests(unittest.TestCase):
    def test_reasoning_block_is_not_shown(self) -> None:
        self.assertEqual("Open the door.", clean_response("<think>draft</think>Open the door."))
        self.assertEqual(
            "Open the door.",
            clean_response("Thinking Process:\nDraft.\nFinal Answer: Open the door."),
        )

    def test_user_terms_are_tokenized_for_task_vocabulary(self) -> None:
        self.assertEqual(
            ["Restart", "the", "router"],
            prompt_terms("Restart the router"),
        )


if __name__ == "__main__":
    unittest.main()
