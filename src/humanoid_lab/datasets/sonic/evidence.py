"""Evidence directory and human-review state handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REVIEW_STATES = {"pending_human_review", "accepted", "accepted_with_notes", "rejected"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_review(evidence_dir: Path) -> Path:
    path = evidence_dir / "human_review.json"
    write_json(path, {"status": "pending_human_review", "notes": ""})
    return path


def validate_review(value: dict[str, Any]) -> None:
    if value.get("status") not in REVIEW_STATES:
        raise ValueError(f"invalid human review status: {value.get('status')!r}")
