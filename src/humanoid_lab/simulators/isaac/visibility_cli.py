"""Collect per-sequence visibility reports into one data-selection table.

Reads the ``visibility.json`` reports a replay run leaves behind, flattens each
into one candidate row (grasp time, window statistics and the selection
booleans) and writes JSONL or CSV. Reports that are not COMPLETED are skipped
unless ``--include-incomplete`` is given, so the table never mixes a failed run
into a selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from humanoid_lab.simulators.isaac.visibility import (
    VISIBILITY_REPORT_NAME,
    VisibilityError,
    candidate_csv,
    candidate_row,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORTS_ROOT = REPO_ROOT / "data" / "outputs" / "grail-replay"


def load_reports(root: Path) -> list[dict[str, Any]]:
    """Every visibility report below *root*, in sequence-key order."""
    if not root.is_dir():
        raise VisibilityError(f"reports root does not exist: {root}")
    reports = []
    for path in sorted(root.glob(f"*/{VISIBILITY_REPORT_NAME}")):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisibilityError(f"cannot read visibility report {path}: {exc}") from exc
    if not reports:
        raise VisibilityError(f"no {VISIBILITY_REPORT_NAME} under {root}")
    return reports


def select_rows(
    reports: Sequence[Mapping[str, Any]],
    *,
    include_incomplete: bool = False,
    below_threshold_only: bool = False,
    out_of_frame_only: bool = False,
) -> list[dict[str, Any]]:
    """Flatten and filter reports into selection rows."""
    rows = []
    for report in reports:
        if not include_incomplete and report.get("result") != "COMPLETED":
            continue
        rows.append(candidate_row(report))
    rows.sort(key=lambda row: row["sequence_key"])
    if below_threshold_only:
        rows = [row for row in rows if row["object_below_threshold_for_whole_window"]]
    if out_of_frame_only:
        rows = [row for row in rows if row["object_out_of_frame_for_whole_window"]]
    return rows


def write_rows(rows: Sequence[Mapping[str, Any]], output: Path | None, output_format: str) -> None:
    """Write the selection rows; files are replaced atomically."""
    text = (
        candidate_csv(rows)
        if output_format == "csv"
        else "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    )
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    handle_descriptor, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(handle_descriptor, 0o644)  # mkstemp is 0600; the table is shareable data
        with os.fdopen(handle_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=f"directory of per-sequence replay outputs (default: {DEFAULT_REPORTS_ROOT})",
    )
    parser.add_argument("--output", type=Path, help="output path (default: stdout)")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl", help="row format")
    parser.add_argument(
        "--below-threshold-only",
        action="store_true",
        help="keep only sequences whose object is below the threshold for the whole grasp window",
    )
    parser.add_argument(
        "--out-of-frame-only",
        action="store_true",
        help="keep only sequences whose object is out of frame for the whole grasp window",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also emit reports whose result is not COMPLETED",
    )
    args = parser.parse_args(argv)

    try:
        reports = load_reports(args.reports_root)
        rows = select_rows(
            reports,
            include_incomplete=args.include_incomplete,
            below_threshold_only=args.below_threshold_only,
            out_of_frame_only=args.out_of_frame_only,
        )
        write_rows(rows, args.output, args.format)
    except VisibilityError as exc:
        print(f"collect-grail-visibility: error: {exc}", file=sys.stderr)
        return 2

    selected = sum(1 for row in rows if row["object_below_threshold_for_whole_window"])
    out_of_frame = sum(1 for row in rows if row["object_out_of_frame_for_whole_window"])
    print(f"reports: {len(reports)}", file=sys.stderr)
    print(f"rows: {len(rows)}", file=sys.stderr)
    print(f"below threshold for the whole window: {selected}", file=sys.stderr)
    print(f"out of frame for the whole window: {out_of_frame}", file=sys.stderr)
    print(f"output: {args.output if args.output else 'stdout'}", file=sys.stderr)
    return 0
