"""The backend accepts only the literal canonical prompt."""

from __future__ import annotations

import unittest

from humanoid_lab.psi0_bridge.prompt import (
    CANONICAL_PROMPT,
    PROMPT_CONTRACT,
    PromptMismatch,
    matches_canonical,
    require_canonical,
)


class CanonicalPromptTest(unittest.TestCase):
    def test_the_exact_string_is_accepted(self) -> None:
        self.assertTrue(matches_canonical(CANONICAL_PROMPT))
        self.assertEqual(require_canonical(CANONICAL_PROMPT), CANONICAL_PROMPT)

    def test_case_and_whitespace_variants_are_rejected(self) -> None:
        variants = {
            "upper": CANONICAL_PROMPT.upper(),
            "lower": CANONICAL_PROMPT.lower(),
            "leading space": " " + CANONICAL_PROMPT,
            "trailing newline": CANONICAL_PROMPT + "\n",
            "collapsed": " ".join(CANONICAL_PROMPT.split()) + " ",
        }
        # "collapsed" is identical to the canonical text except for a trailing
        # space, which the literal contract must reject as well.
        for name, variant in variants.items():
            with self.subTest(case=name):
                self.assertFalse(matches_canonical(variant))
                with self.assertRaises(PromptMismatch):
                    require_canonical(variant)

    def test_paraphrases_are_rejected(self) -> None:
        for bad in ("", "Stack the blocks.", CANONICAL_PROMPT + " Please.", CANONICAL_PROMPT[:-1]):
            with self.subTest(bad=bad):
                self.assertFalse(matches_canonical(bad))
                with self.assertRaises(PromptMismatch):
                    require_canonical(bad)

    def test_the_error_names_the_literal_contract(self) -> None:
        with self.assertRaises(PromptMismatch) as ctx:
            require_canonical("stack the three cubic blocks on the black tape in the order red, yellow, blue.")
        message = str(ctx.exception)
        self.assertIn(PROMPT_CONTRACT, message)
        self.assertIn(CANONICAL_PROMPT, message)

    def test_prompt_mismatch_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(PromptMismatch, ValueError))

    def test_require_canonical_never_forwards_caller_text(self) -> None:
        self.assertEqual(require_canonical(CANONICAL_PROMPT), CANONICAL_PROMPT)

    def test_non_strings_are_rejected(self) -> None:
        for value in (None, 42, ["x"]):
            with self.subTest(value=value):
                self.assertFalse(matches_canonical(value))
                with self.assertRaises(PromptMismatch):
                    require_canonical(value)


if __name__ == "__main__":
    unittest.main()
