#!/usr/bin/env python3
"""Loopback regression for accepting a native body result one action behind Isaac."""
from __future__ import annotations

import socket
import sys
import time
import unittest
from pathlib import Path

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cloudwalk_closed_loop import BodyCommand, SonicState


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class NativeReturnIntegrationTests(unittest.TestCase):
    @unittest.skipIf(zmq is None, "pyzmq is a Raider runtime dependency")
    def test_body_result_is_received_when_its_frame_precedes_next_outbound_action(self):
        port = free_port()
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        subscriber = context.socket(zmq.SUB)
        publisher.bind(f"tcp://127.0.0.1:{port}")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.15)
        publisher.send(BodyCommand(7, 123, (0.25,) * 29).pack())
        poller = zmq.Poller(); poller.register(subscriber, zmq.POLLIN)
        self.assertIn(subscriber, dict(poller.poll(1000)))
        result = BodyCommand.unpack(subscriber.recv())
        next_outbound_sequence, last_command_sequence = 8, -1
        if result.sequence > last_command_sequence:
            last_command_sequence = result.sequence
        self.assertEqual(last_command_sequence, next_outbound_sequence - 1)
        publisher.close(); subscriber.close(); context.term()

    def test_native_source_has_independent_state_drain_and_v4_frame_sequence(self):
        native = (Path(__file__).resolve().parent / "native" / "sonic_closed_loop_harness.cpp").read_text()
        isaac = (Path(__file__).resolve().parents[1] / "scripts" / "run-cloudwalk-isaac.py").read_text()
        self.assertIn("std::thread state_receiver", native)
        self.assertIn("state_count < kHistory", native)
        self.assertIn("read_i64(fields[1].data)", native)
        self.assertIn("native_body_publish", native)
        self.assertIn("last_command_sequence", isaac)
        self.assertIn("isaac_body_receive", isaac)


if __name__ == "__main__":
    unittest.main()
