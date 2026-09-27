"""Acceptance checks must require sustained arm and both-hand motion."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/flux-verify-tracking.py"
spec = importlib.util.spec_from_file_location("flux_verify_tracking", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def rows(*, left_moves=True, count=12):
    result = []
    for index in range(count):
        body = [0.0] * 29
        body[15] = index * 0.03
        left = [index * 0.03 if left_moves else 0.0] + [0.0] * 6
        right = [index * 0.03] + [0.0] * 6
        result.append({
            "wall_time_ns": index * 600_000_000,
            "sim_s": index * 0.6,
            "body_measured": body,
            "body_target": body,
            "left_hand_measured": left,
            "left_hand_target": left,
            "right_hand_measured": right,
            "right_hand_target": right,
            "support_active": False,
        })
    return result


@pytest.mark.parametrize("left_moves,count,expected", [
    (True, 12, "PASS"),
    (False, 12, "FAIL"),
    (True, 3, "FAIL"),
])
def test_acceptance_requires_both_hands_and_duration(monkeypatch, capsys, tmp_path, left_moves, count, expected):
    monkeypatch.setattr(verifier, "load_rows", lambda _: rows(left_moves=left_moves, count=count))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"result": "COMPLETED", "controller": {"kind": "flux_dds"}}))
    rc = verifier.main([
        "--tracking", str(tmp_path / "tracking.parquet"),
        "--metrics", str(summary),
        "--min-command-hz", "0",
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == expected
    assert rc == (0 if expected == "PASS" else 1)
    if not left_moves:
        assert any("left hand" in problem for problem in report["problems"])
    if count == 3:
        assert any("control lasted" in problem for problem in report["problems"])
