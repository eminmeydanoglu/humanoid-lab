#!/usr/bin/env python3
"""Transport test for the shipped DDS bridge.

Drives the real ``serve()`` from ``scripts/sonic-isaac-dds-bridge.py`` against
the real ``StateLink`` from ``tools/sonic_isaac_ipc.py`` over loopback. The DDS
adapter is the only thing stubbed, since it needs CycloneDDS and a GPU host.

This pins the direction of the link: the runner listens and the bridge connects.
Getting that backwards leaves SONIC with no robot state at all.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sonic_isaac_contract import BODY_JOINT_COUNT  # noqa: E402
from sonic_isaac_ipc import LowCmdFrame, StateFrame, StateLink  # noqa: E402


def load_bridge():
    spec = importlib.util.spec_from_file_location(
        "sonic_isaac_dds_bridge", ROOT / "scripts" / "sonic-isaac-dds-bridge.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bridge = load_bridge()


def vector(value: float) -> tuple[float, ...]:
    return (value,) * BODY_JOINT_COUNT


def state_frame() -> StateFrame:
    return StateFrame(
        tick_us=1,
        root_pos=(0.0, 0.0, 0.75),
        root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        root_lin_vel=(0.0, 0.0, 0.0),
        root_ang_vel=(0.0, 0.0, 0.0),
        root_acc=(0.0, 0.0, 0.0),
        torso_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        torso_gyro=(0.0, 0.0, 0.0),
        body_q=vector(0.1),
        body_dq=vector(0.0),
        body_ddq=vector(0.0),
        body_tau_est=vector(0.0),
    )


class FakeDds:
    """Records published state and returns one command frame, then None."""

    def __init__(self, command: LowCmdFrame | None) -> None:
        self.published: list[StateFrame] = []
        self._command = command
        self.taken = 0
        self.received = 0

    def publish(self, frame: StateFrame) -> None:
        self.published.append(frame)

    def take_cmd(self):
        self.taken += 1
        self.received += 0
        command, self._command = self._command, None
        return command


class BridgeTransportTest(unittest.TestCase):
    def run_bridge(self, link: StateLink, dds: FakeDds):
        result: dict = {}

        def target():
            result["rc"] = bridge.serve(host=link.host, port=link.port, dds=dds)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, result

    def test_bridge_connects_to_the_runner_and_consumes_state(self) -> None:
        link = StateLink(port=0)
        link.open()
        port = link.port
        dds = FakeDds(None)
        thread, result = self.run_bridge(link, dds)

        # The bridge is the client: it must reach the runner's listener.
        deadline = time.monotonic() + 10.0
        connected = False
        while time.monotonic() < deadline:
            link.accept()
            if link.connected:
                connected = True
                break
            time.sleep(0.02)
        self.assertTrue(connected, "bridge never connected to the runner's link")

        for _ in range(5):
            self.assertTrue(link.publish(state_frame()))
            time.sleep(0.02)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and len(dds.published) < 5:
            time.sleep(0.02)
        self.assertEqual(len(dds.published), 5)
        # Frames are float32 on the wire, so compare within that precision.
        published = dds.published[0]
        self.assertEqual(len(published.body_q), BODY_JOINT_COUNT)
        for value in published.body_q:
            self.assertAlmostEqual(value, 0.1, places=6)
        self.assertAlmostEqual(published.root_pos[2], 0.75, places=6)

        link.close()
        thread.join(timeout=10)

    def test_lowcmd_from_dds_reaches_the_runner(self) -> None:
        link = StateLink(port=0)
        link.open()
        command = LowCmdFrame(
            sequence=42, q=vector(0.5), dq=vector(0.0), tau=vector(1.0),
            kp=vector(20.0), kd=vector(1.0),
        )
        dds = FakeDds(command)
        thread, result = self.run_bridge(link, dds)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            link.accept()
            if link.connected:
                break
            time.sleep(0.02)

        received = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and received is None:
            link.publish(state_frame())
            received = link.take_command()
            time.sleep(0.02)

        self.assertIsNotNone(received, "the runner never received the lowcmd frame")
        self.assertEqual(received.sequence, 42)
        for value in received.q:
            self.assertAlmostEqual(value, 0.5, places=6)
        for value in received.kp:
            self.assertAlmostEqual(value, 20.0, places=6)

        link.close()
        thread.join(timeout=10)

    def test_bridge_survives_an_idle_gap_longer_than_the_socket_timeout(self) -> None:
        """A pause in publishing must not be mistaken for a closed link."""
        link = StateLink(port=0)
        link.open()
        dds = FakeDds(None)
        thread, _ = self.run_bridge(link, dds)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            link.accept()
            if link.connected:
                break
            time.sleep(0.02)
        self.assertTrue(link.connected)

        # The bridge's socket timeout is 5 s; idle past it and then resume.
        self.assertTrue(link.publish(state_frame()))
        time.sleep(7.0)
        for _ in range(3):
            link.publish(state_frame())
            time.sleep(0.02)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and len(dds.published) < 4:
            time.sleep(0.05)
        self.assertGreaterEqual(
            len(dds.published), 4,
            "the bridge dropped the link during an idle gap",
        )
        link.close()
        thread.join(timeout=10)

    def test_bridge_reports_failure_when_nothing_is_listening(self) -> None:
        import socket

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        class UnusedDds:
            def publish(self, frame):  # pragma: no cover - must not be called
                raise AssertionError("no frame should be published")

            def take_cmd(self):  # pragma: no cover - must not be called
                return None

        rc = bridge.serve(host="127.0.0.1", port=port, dds=UnusedDds(),
                          connect_timeout_s=1.0)
        self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
