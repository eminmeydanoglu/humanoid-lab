"""The one canonical prompt this bridge accepts, as a literal string.

The web form prefills an editable copy of ``CANONICAL_PROMPT`` for review, but
the backend answers only an exact, byte-for-byte match: a paraphrase, a case
change or extra whitespace would silently change the policy's condition, so
anything else is rejected before it reaches the model.
"""

from __future__ import annotations

CANONICAL_PROMPT = (
    "Stack the three cubic blocks on the black tape in the order red, yellow, blue."
)

TASK_PROMPTS = {
    "BlockStacking": CANONICAL_PROMPT,
    "PickApple": "Put the apple into the plate.",
    "PickGum": "Put the gum into the plate.",
}

#: The literal contract the backend enforces: `text == CANONICAL_PROMPT`.
PROMPT_CONTRACT = "text must equal CANONICAL_PROMPT exactly (no case, spacing or punctuation changes)"


class PromptMismatch(ValueError):
    """The submitted instruction is not the canonical prompt."""


def matches_canonical(text: str) -> bool:
    """True only for the exact canonical string."""
    return isinstance(text, str) and text == CANONICAL_PROMPT


def require_canonical(text: str) -> str:
    """Return ``CANONICAL_PROMPT`` or raise; never forwards the caller's text."""
    if not matches_canonical(text):
        raise PromptMismatch(f"{PROMPT_CONTRACT}: {CANONICAL_PROMPT}")
    return CANONICAL_PROMPT
