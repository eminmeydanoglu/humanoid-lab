"""Fruits adapter: resolve body and hand blocks from modality metadata."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED = ("left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand")


def action_slices(dataset: Path) -> dict[str, slice]:
    modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))["action"]
    if tuple(modality) != EXPECTED:
        raise ValueError(f"unexpected Fruits action block order: {tuple(modality)}")
    result = {name: slice(int(spec["start"]), int(spec["end"])) for name, spec in modality.items()}
    if result["right_hand"].stop != 43:
        raise ValueError("Fruits action must be 43D")
    return result
