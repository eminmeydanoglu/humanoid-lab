"""Online/offline token-match: retained stream schema, offline recompute, rejection.

The offline mode must recompute every metric from the retained NDJSON alone (no
ZMQ at all) and agree with the online pass, and malformed streams must fail
closed instead of silently producing a partial match.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "psi0-token-match.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("psi0_token_match", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = load_probe()


def token(*values: float) -> list[float]:
    vector = [0.0] * PROBE.TOKEN_DIM
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def published(seq: int, frame_index: int, values: list[float]) -> dict:
    return {"kind": "published", "seq": seq, "t_wall": 1.0 + seq, "t_mono": float(seq),
            "frame_index": frame_index, "token": token(*values), "left_hand": [], "right_hand": []}


def consumed(seq: int, values: list[float], **stored) -> dict:
    event = {"kind": "consumed", "seq": seq, "t_wall": 1.0 + seq, "t_mono": float(seq),
             "token_state": token(*values),
             "source": {"state_endpoint": "tcp://127.0.0.1:5557", "state_topic": "g1_debug"}}
    event.update(stored)
    return event


def attach_online_fields(events: list[dict]) -> list[dict]:
    """Attach the match fields the online pass writes, using the same matcher."""
    window: deque = deque(maxlen=PROBE.WINDOW)
    latest: int | None = None
    for event in events:
        if event["kind"] == "published":
            latest = event["frame_index"]
            window.append((latest, np.asarray(event["token"], dtype=np.float32)))
        else:
            event.update(PROBE._match_one(window, latest, np.asarray(event["token_state"], dtype=np.float32)))
    return events


def write_stream(events: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8")
    for event in events:
        handle.write(json.dumps(event) + "\n")
    handle.close()
    return Path(handle.name)


class OfflineRecomputeTest(unittest.TestCase):
    def tearDown(self) -> None:
        for path in getattr(self, "paths", []):
            path.unlink(missing_ok=True)

    def _stream(self, events: list[dict]) -> Path:
        path = write_stream(events)
        self.paths = getattr(self, "paths", []) + [path]
        return path

    def test_offline_recomputes_the_exact_result(self) -> None:
        events = attach_online_fields([
            consumed(0, [0.1]),                              # nothing published yet
            published(1, 10, [0.1]),
            consumed(2, [0.1]),
            published(3, 11, [0.2]),
            published(4, 12, [0.3]),
            consumed(5, [0.2]),
            consumed(6, [0.9]),                              # never on the grid
        ])
        result = PROBE.run_offline(self._stream(events))
        self.assertEqual(result["mode"], "offline")
        self.assertEqual(result["published_count"], 3)
        self.assertEqual(result["published_frame_index"], {"first": 10, "last": 12})
        self.assertEqual(result["consumed_samples"], 4)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["unmatched"], 2)
        self.assertEqual(result["unmatched_reasons"],
                         {"no published token observed yet": 1, "diff_above_tolerance": 1})
        # The closest published token to [0.9] is [0.3] -> diff 0.6 (not 0.8).
        self.assertAlmostEqual(result["max_abs_diff"], 0.6, places=5)
        self.assertEqual(result["matched_max_abs_diff"], 0.0)
        self.assertEqual(result["lag_frames"], {"min": 0, "max": 1})
        # The retained stream carries the online match fields, so recomputing from
        # the raw arrays must reproduce them exactly.
        self.assertEqual(result["event_field_mismatches"], [])
        self.assertEqual(result["verified_events"], 4)

    def test_offline_stream_carries_both_event_kinds_with_match_fields(self) -> None:
        events = attach_online_fields([published(0, 7, [0.25]), consumed(1, [0.25])])
        result = PROBE.run_offline(self._stream(events))
        self.assertEqual(result["matched"], 1)
        stored = [e for e in events if e["kind"] == "consumed"][0]
        self.assertEqual(stored["seq"], 1)
        self.assertIn("t_mono", stored)
        self.assertEqual(len(stored["token_state"]), PROBE.TOKEN_DIM)
        self.assertEqual(stored["matched_frame_index"], 7)
        self.assertEqual(stored["latest_published_frame_index"], 7)
        self.assertEqual(stored["lag_frames"], 0)
        self.assertEqual(stored["source"]["state_topic"], "g1_debug")

    def test_offline_flags_stored_fields_that_do_not_match_the_raw_arrays(self) -> None:
        events = [
            published(0, 0, [0.1]),
            consumed(1, [0.1], matched=False, matched_frame_index=None,
                     latest_published_frame_index=None, lag_frames=None,
                     max_abs_diff=0.0, reason="diff_above_tolerance"),  # wrong: it does match
        ]
        result = PROBE.run_offline(self._stream(events))
        self.assertEqual(result["matched"], 1)
        self.assertTrue(result["event_field_mismatches"])
        self.assertIn("matched", {entry["field"] for entry in result["event_field_mismatches"]})

    def test_malformed_streams_are_rejected(self) -> None:
        cases = {
            "not json": "not json\n",
            "missing kind": json.dumps({"seq": 0, "t_mono": 0.0}) + "\n",
            "unknown kind": json.dumps({"kind": "other", "seq": 0, "t_mono": 0.0}) + "\n",
            "missing order": json.dumps({"kind": "published", "frame_index": 0, "token": token(0.1)}) + "\n",
            "short published token": json.dumps(
                {"kind": "published", "seq": 0, "t_mono": 0.0, "frame_index": 0, "token": [0.0] * 63}) + "\n",
            "short consumed token": json.dumps(
                {"kind": "consumed", "seq": 0, "t_mono": 0.0, "token_state": [0.0] * 80}) + "\n",
            "empty": "",
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                path = Path(tempfile.mkstemp(suffix=".ndjson")[1])
                path.write_text(body, encoding="utf-8")
                self.paths = getattr(self, "paths", []) + [path]
                with self.assertRaises(PROBE.StreamError):
                    PROBE.run_offline(path)


class OnlineOfflineAgreementTest(unittest.TestCase):
    def test_online_and_offline_core_metrics_are_identical(self) -> None:
        events = [
            published(0, 0, [0.0625, -0.0625]),
            consumed(1, [0.0625, -0.0625]),
            published(2, 1, [0.125]),
            consumed(3, [0.125]),
            consumed(4, [0.5]),
        ]
        online, _ = PROBE._match_events(events)
        path = write_stream(events)
        try:
            offline = PROBE.run_offline(path)
        finally:
            path.unlink(missing_ok=True)
        for key in ("published_count", "consumed_samples", "matched", "unmatched",
                    "unmatched_reasons", "max_abs_diff", "matched_max_abs_diff",
                    "lag_frames", "tolerance"):
            self.assertEqual(online[key], offline[key], key)

    def test_offline_mode_never_imports_zmq(self) -> None:
        events = [published(0, 0, [0.1]), consumed(1, [0.1])]
        path = write_stream(events)
        out = Path(tempfile.mkstemp(suffix=".json")[1])
        try:
            # sys.modules["zmq"] = None makes any `import zmq` fail loudly.
            code = (
                "import sys, importlib.util, pathlib\n"
                "sys.modules['zmq'] = None\n"
                "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(mod)\n"
                "result = mod.run_offline(pathlib.Path(sys.argv[2]))\n"
                "print(result['matched'], result['consumed_samples'])\n"
            )
            proc = subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(path)],
                                  capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "1 1")
        finally:
            path.unlink(missing_ok=True)
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
