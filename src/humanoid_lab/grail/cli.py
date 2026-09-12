"""Build a deterministic prompt manifest from GRAIL ``pickup_table`` robot stems.

The robot directory is the only source of truth: one ``*.pkl`` file is one
source trajectory, and the file stem is its identity.  The manifest is written
as JSONL in sorted ``source_motion_id`` order and atomically replaced, so a
failed run never leaves a half-written manifest behind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

from humanoid_lab.grail.prompts import (
    MANIFEST_SCHEMA_VERSION,
    PROMPT_POLICY_VERSION,
    PromptLayerError,
    PromptRecord,
    build_prompt_record,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_DIR = (
    REPO_ROOT / "data" / "datasets" / "grail" / "data" / "pickup_table" / "robot"
)


def read_robot_stems(robot_dir: Path) -> list[str]:
    """Return the sorted stems of every ``*.pkl`` trajectory in *robot_dir*."""
    if not robot_dir.is_dir():
        raise PromptLayerError(f"robot data directory does not exist: {robot_dir}")
    stems = sorted(path.stem for path in robot_dir.glob("*.pkl") if path.is_file())
    if not stems:
        raise PromptLayerError(f"no *.pkl trajectories under {robot_dir}")
    return stems


def build_manifest(robot_dir: Path) -> list[PromptRecord]:
    """Build one prompt record per source trajectory, ordered by motion id."""
    records = (build_prompt_record(stem) for stem in read_robot_stems(robot_dir))
    return sorted(records, key=lambda record: record.source_motion_id)


def write_manifest(records: Sequence[PromptRecord], output: Path) -> None:
    """Write *records* as JSONL to *output*, replacing it atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    handle_descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(handle_descriptor, 0o644)  # mkstemp is 0600; the manifest is shareable data
        with os.fdopen(handle_descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_manifest_row(), ensure_ascii=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "a GRAIL pickup_table prompt manifest (JSONL) from robot trajectory stems"
        )
    )
    parser.add_argument(
        "--robot-dir",
        type=Path,
        default=DEFAULT_ROBOT_DIR,
        help=f"directory of robot *.pkl trajectories (default: {DEFAULT_ROBOT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="manifest path to write (JSONL, replaced atomically)",
    )
    args = parser.parse_args(argv)

    if args.output.is_dir():
        print(
            f"generate-grail-prompt-manifest: error: output is a directory: {args.output}",
            file=sys.stderr,
        )
        return 2
    try:
        records = build_manifest(args.robot_dir)
        write_manifest(records, args.output)
    except PromptLayerError as exc:
        print(f"generate-grail-prompt-manifest: error: {exc}", file=sys.stderr)
        return 2

    counts = Counter(record.prompt_template_id for record in records)
    print(f"schema_version: {MANIFEST_SCHEMA_VERSION}")
    print(f"prompt_policy_version: {PROMPT_POLICY_VERSION}")
    print(f"records: {len(records)}")
    print(f"unique source_motion_id: {len(set(r.source_motion_id for r in records))}")
    print("template distribution:")
    for template_id in sorted(counts):
        print(f"  {template_id}: {counts[template_id]}")
    print(f"output: {args.output}")
    return 0
