"""The opt-in demonstration-token warm start: stream loading and its publisher.

Two things have to hold before any session runs one:

* a prepared stream file is either exactly usable or refused (shape, finiteness,
  positive control rate, agreeing lengths) -- a settle must never be the place a
  malformed artifact is discovered;
* the publisher sends its ticks at the control rate, repeats the last token once
  the stream is exhausted, stops when halted, and can only ever reach SONIC
  through the router's own source gate.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from humanoid_lab.psi0_bridge.warmstart import (
    FileSimulationClock,
    SimulationClockSample,
    WarmStartError,
    WarmStartStream,
    load_token_stream,
    stream_from_tokens,
)


def write_stream(path: Path, *, ticks: int = 4, control_hz: float = 200.0,
                 tokens: object | None = None, left: object | None = None,
                 right: object | None = None, omit: str | None = None) -> Path:
    payload = {
        "control_hz": control_hz,
        "source": {"episode_index": 283, "segment": "start phase"},
        "tokens": [[0.0625 * (index % 3)] * 64 for index in range(ticks)],
        "left_hand_joints": [[0.1 * index] * 7 for index in range(ticks)],
        "right_hand_joints": [[-0.1 * index] * 7 for index in range(ticks)],
    }
    if tokens is not None:
        payload["tokens"] = tokens
    if left is not None:
        payload["left_hand_joints"] = left
    if right is not None:
        payload["right_hand_joints"] = right
    if omit is not None:
        payload.pop(omit)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakeRouter:
    """Records what a real ActionRouter would have been offered."""

    def __init__(self) -> None:
        self.accepted: list[bytes] = []
        self.gate = True

    def submit_warmstart(self, payload: bytes) -> bool:
        if not self.gate:
            return False
        self.accepted.append(payload)
        return True


class ManualClock:
    def __init__(self, sim_s: float | None = None) -> None:
        self.sim_s = sim_s
        self.revision = 0

    def read(self):
        if self.sim_s is None:
            return None
        return SimulationClockSample(self.sim_s, self.revision)

    def set(self, sim_s: float) -> None:
        self.sim_s = sim_s
        self.revision += 1


class FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, kind: str, **fields):
        self.events.append((kind, fields))
        return fields

    def states(self) -> list[str]:
        return [fields.get("state") for _, fields in self.events if "state" in fields]


def packer(token, left, right, index):
    return json.dumps({
        "token": [float(value) for value in token],
        "left": [float(value) for value in left],
        "right": [float(value) for value in right],
        "index": int(index),
    }).encode("utf-8")


class LoadTokenStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "warmstart.json"

    def test_a_prepared_stream_round_trips(self) -> None:
        stream = load_token_stream(write_stream(self.path, ticks=5, control_hz=50.0))
        self.assertEqual(stream.ticks, 5)
        self.assertEqual(stream.tokens.shape, (5, 64))
        self.assertEqual(stream.left_hand_joints.shape, (5, 7))
        self.assertEqual(stream.right_hand_joints.shape, (5, 7))
        self.assertAlmostEqual(stream.duration_s, 0.1)
        self.assertEqual(stream.source["episode_index"], 283)
        self.assertIsNotNone(stream.sha256)
        self.assertEqual(stream.summary()["ticks"], 5)

    def test_a_missing_file_is_a_warm_start_error(self) -> None:
        with self.assertRaises(WarmStartError):
            load_token_stream(Path(self.tmp.name) / "nope.json")

    def test_rejections(self) -> None:
        cases = {
            "not json": lambda path: path.write_text("{", encoding="utf-8"),
            "missing tokens": lambda path: write_stream(path, omit="tokens"),
            "a wrong token width": lambda path: write_stream(path, tokens=[[0.0] * 63]),
            "a wrong hand width": lambda path: write_stream(path, left=[[0.0] * 6] * 4),
            "lengths disagree": lambda path: write_stream(path, right=[[0.0] * 7] * 3),
            "no ticks": lambda path: write_stream(path, tokens=[], left=[], right=[]),
            "non finite": lambda path: write_stream(path, tokens=[[float("nan")] * 64] * 4),
            "zero rate": lambda path: write_stream(path, control_hz=0.0),
        }
        for label, write in cases.items():
            with self.subTest(label):
                write(self.path)
                with self.assertRaises(WarmStartError):
                    load_token_stream(self.path)


class FileSimulationClockTest(unittest.TestCase):
    def test_missing_partial_and_complete_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock.txt"
            clock = FileSimulationClock(path)
            self.assertIsNone(clock.read())
            path.write_text("", encoding="ascii")
            self.assertIsNone(clock.read())
            path.write_text("0.", encoding="ascii")
            self.assertIsNone(clock.read())
            path.write_text("broken", encoding="ascii")
            self.assertIsNone(clock.read())
            path.write_text("1.250000\n", encoding="ascii")
            self.assertAlmostEqual(clock.read().sim_s, 1.25)


class WarmStartStreamTest(unittest.TestCase):
    def build(self, *, ticks: int = 4, control_hz: float = 200.0, telemetry=None,
              clock=None, clock_timeout_s: float = 0.2):
        router = FakeRouter()
        stream = stream_from_tokens(
            [[0.01 * index] * 64 for index in range(ticks)],
            [[0.1 * index] * 7 for index in range(ticks)],
            [[-0.1 * index] * 7 for index in range(ticks)],
            control_hz=control_hz, source={"episode_index": 283},
        )
        return WarmStartStream(
            stream, router, telemetry=telemetry, packer=packer,
            simulation_clock=clock, clock_timeout_s=clock_timeout_s,
            log=lambda message: None,
        ), router

    def test_the_stream_is_published_at_the_control_rate_then_held(self) -> None:
        publisher, router = self.build(ticks=4, control_hz=200.0)
        publisher.arm(0.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(router.accepted) < 12:
            time.sleep(0.01)
        publisher.halt()
        self.assertGreaterEqual(len(router.accepted), 12)
        records = [json.loads(payload) for payload in router.accepted]
        # The four prepared ticks go out in order and the last one is repeated.
        self.assertEqual([record["index"] for record in records[:4]], [0, 1, 2, 3])
        self.assertEqual([record["token"][0] for record in records[:4]], [0.0, 0.01, 0.02, 0.03])
        self.assertTrue(all(record["token"][0] == 0.03 for record in records[4:]))
        self.assertFalse(publisher.armed)

    def test_a_delay_keeps_the_first_tick_back(self) -> None:
        publisher, router = self.build(ticks=2, control_hz=200.0)
        publisher.arm(0.3)
        time.sleep(0.15)
        self.assertEqual(router.accepted, [])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not router.accepted:
            time.sleep(0.01)
        publisher.halt()
        self.assertTrue(router.accepted)

    def test_halt_stops_the_publisher(self) -> None:
        publisher, router = self.build(ticks=2, control_hz=200.0)
        publisher.arm(0.0)
        time.sleep(0.1)
        publisher.halt()
        sent = len(router.accepted)
        time.sleep(0.1)
        self.assertEqual(len(router.accepted), sent)
        publisher.halt()  # idempotent

    def test_a_closed_router_gate_sends_nothing(self) -> None:
        publisher, router = self.build(ticks=2, control_hz=200.0)
        router.gate = False
        publisher.arm(0.0)
        time.sleep(0.15)
        publisher.halt()
        self.assertEqual(router.accepted, [])
        self.assertEqual(publisher.status()["sent"], 0)

    def test_telemetry_records_arm_start_and_halt(self) -> None:
        telemetry = FakeTelemetry()
        publisher, _ = self.build(ticks=2, control_hz=200.0, telemetry=telemetry)
        publisher.arm(0.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and "started" not in telemetry.states():
            time.sleep(0.01)
        publisher.halt()
        states = telemetry.states()
        self.assertEqual(states[0], "armed")
        self.assertIn("started", states)
        self.assertEqual(states[-1], "halted")
        self.assertTrue(all(kind == "warmstart" for kind, _ in telemetry.events))

    def test_rearming_replaces_the_previous_stream(self) -> None:
        publisher, router = self.build(ticks=2, control_hz=200.0)
        publisher.arm(5.0)
        publisher.arm(0.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not router.accepted:
            time.sleep(0.01)
        publisher.halt()
        self.assertTrue(router.accepted)

    def test_simulation_clock_waits_for_fresh_sample_and_sends_sequential_frames(self) -> None:
        clock = ManualClock(9.0)
        publisher, router = self.build(ticks=3, control_hz=50.0, clock=clock)
        publisher.arm(0.0)
        time.sleep(0.03)
        self.assertEqual(router.accepted, [])
        clock.set(9.02)  # A newer sample from the previous episode is still stale.
        time.sleep(0.03)
        self.assertEqual(router.accepted, [])
        clock.set(0.02)
        time.sleep(0.03)
        clock.set(0.10)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(router.accepted) < 5:
            time.sleep(0.005)
        publisher.halt()
        records = [json.loads(payload) for payload in router.accepted]
        self.assertEqual([row["index"] for row in records[:5]], [0, 1, 2, 3, 4])
        self.assertEqual([row["token"][0] for row in records[:5]], [0.0, 0.01, 0.02, 0.02, 0.02])
        self.assertEqual(records[1]["left"], [0.1] * 7)
        self.assertEqual(records[1]["right"], [-0.1] * 7)

    def test_arm_after_confirmed_reset_uses_trusted_pre_reset_baseline(self) -> None:
        clock = ManualClock(9.0)
        publisher, router = self.build(ticks=3, control_hz=50.0, clock=clock)
        baseline = publisher.clock_sample()
        clock.set(0.02)  # Reset completed and the harness validated the new episode.
        publisher.arm(0.0, clock_baseline=baseline)
        clock.set(0.10)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(router.accepted) < 5:
            time.sleep(0.005)
        publisher.halt()
        self.assertEqual([json.loads(row)["index"] for row in router.accepted[:5]], [0, 1, 2, 3, 4])

    def test_trusted_baseline_rejects_forward_sample_from_previous_episode(self) -> None:
        clock = ManualClock(9.0)
        publisher, router = self.build(clock=clock, clock_timeout_s=0.2)
        baseline = publisher.clock_sample()
        clock.set(9.02)
        publisher.arm(0.0, clock_baseline=baseline)
        time.sleep(0.04)
        self.assertEqual(router.accepted, [])
        clock.set(0.02)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not router.accepted:
            time.sleep(0.005)
        publisher.halt()
        self.assertTrue(router.accepted)

    def test_simulation_clock_pause_and_resume_preserves_sequence(self) -> None:
        clock = ManualClock(5.0)
        publisher, router = self.build(ticks=6, control_hz=50.0, clock=clock, clock_timeout_s=0.2)
        baseline = publisher.clock_sample()
        clock.set(0.0)
        publisher.arm(0.0, clock_baseline=baseline)
        clock.set(0.04)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(router.accepted) < 3:
            time.sleep(0.005)
        paused = len(router.accepted)
        time.sleep(0.05)
        self.assertEqual(len(router.accepted), paused)
        clock.set(0.10)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(router.accepted) < 6:
            time.sleep(0.005)
        publisher.halt()
        self.assertEqual([json.loads(row)["index"] for row in router.accepted[:6]], list(range(6)))

    def test_full_1509_frame_stream_is_released_without_skips(self) -> None:
        clock = ManualClock(40.0)
        publisher, router = self.build(ticks=1509, control_hz=50.0, clock=clock)
        baseline = publisher.clock_sample()
        clock.set(0.0)
        publisher.arm(0.0, clock_baseline=baseline)
        clock.set(30.16)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(router.accepted) < 1509:
            time.sleep(0.005)
        publisher.halt()
        indices = [json.loads(row)["index"] for row in router.accepted]
        self.assertEqual(indices[:1509], list(range(1509)))

    def test_simulation_clock_pause_freezes_then_times_out(self) -> None:
        clock = ManualClock(4.0)
        publisher, router = self.build(clock=clock, clock_timeout_s=0.08)
        publisher.arm(0.0)
        clock.set(0.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not router.accepted:
            time.sleep(0.005)
        sent = len(router.accepted)
        time.sleep(0.04)
        self.assertEqual(len(router.accepted), sent)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and publisher.status()["error"] is None:
            time.sleep(0.005)
        self.assertIn("stopped advancing", publisher.status()["error"] or "")
        publisher.halt()

    def test_simulation_clock_backwards_fails_safely(self) -> None:
        clock = ManualClock(3.0)
        publisher, _ = self.build(clock=clock)
        publisher.arm(0.0)
        clock.set(0.1)
        time.sleep(0.02)
        clock.set(0.08)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and publisher.status()["error"] is None:
            time.sleep(0.005)
        self.assertIn("moved backwards", publisher.status()["error"] or "")
        publisher.halt()

    def test_simulation_clock_missing_fails_safely(self) -> None:
        clock = ManualClock(None)
        publisher, router = self.build(clock=clock, clock_timeout_s=0.05)
        publisher.arm(0.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and publisher.status()["error"] is None:
            time.sleep(0.005)
        self.assertEqual(router.accepted, [])
        self.assertIn("no fresh post-reset", publisher.status()["error"] or "")
        publisher.halt()

    def test_a_missing_packer_is_reported_not_raised(self) -> None:
        router = FakeRouter()
        stream = stream_from_tokens(
            [[0.0] * 64], [[0.0] * 7], [[0.0] * 7], control_hz=200.0,
        )
        publisher = WarmStartStream(stream, router, log=lambda message: None)
        with mock.patch(
            "humanoid_lab.psi0_bridge.warmstart.default_packer",
            side_effect=ImportError("no gear_sonic"),
        ):
            publisher.arm(0.0)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and publisher.status()["error"] is None:
                time.sleep(0.01)
        publisher.halt()
        self.assertEqual(router.accepted, [])
        self.assertIn("cannot import the protocol v4 packer", publisher.status()["error"] or "")


if __name__ == "__main__":
    unittest.main()
