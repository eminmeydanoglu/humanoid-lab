from __future__ import annotations

import socket
import time
import unittest

try:
    import zmq
except ImportError:  # pragma: no cover
    zmq = None

from humanoid_lab.psi0_bridge.action_router import ActionRouter


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipIf(zmq is None, "pyzmq is not installed")
class ActionRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public_port = free_port()
        self.groot_port = free_port()
        self.context = zmq.Context()
        self.groot = self.context.socket(zmq.PUB)
        self.groot.setsockopt(zmq.LINGER, 0)
        self.groot.bind(f"tcp://127.0.0.1:{self.groot_port}")
        self.output = self.context.socket(zmq.SUB)
        self.output.setsockopt(zmq.LINGER, 0)
        self.output.setsockopt(zmq.SUBSCRIBE, b"")
        self.output.connect(f"tcp://127.0.0.1:{self.public_port}")
        self.router = ActionRouter(
            public_endpoint=f"tcp://127.0.0.1:{self.public_port}",
            groot_endpoint=f"tcp://127.0.0.1:{self.groot_port}",
        )
        time.sleep(0.2)

    def tearDown(self) -> None:
        self.router.close()
        self.groot.close()
        self.output.close()
        self.context.term()

    def receive(self) -> bytes:
        self.assertTrue(self.output.poll(1000))
        return self.output.recv()

    def test_forwards_only_selected_source(self) -> None:
        self.router.select("psi")
        self.router.resume()
        self.router.psi_sink.resume()
        self.assertTrue(self.router.psi_sink.publish(b"psi"))
        self.assertEqual(self.receive(), b"psi")
        self.groot.send(b"old-groot")
        self.assertFalse(self.output.poll(100))

        self.router.select("groot")
        self.router.resume()
        time.sleep(0.1)
        self.groot.send(b"groot")
        self.assertEqual(self.receive(), b"groot")
        self.assertFalse(self.router.psi_sink.publish(b"old-psi"))

    def test_halt_closes_both_policy_gates_but_control_still_passes(self) -> None:
        self.router.select("psi")
        self.router.halt()
        self.router.psi_sink.resume()
        self.assertFalse(self.router.psi_sink.publish(b"psi"))
        self.groot.send(b"groot")
        self.assertFalse(self.output.poll(100))
        self.router.send_control(b"control")
        self.assertEqual(self.receive(), b"control")

    def test_a_warm_start_stream_reaches_the_port_only_while_selected(self) -> None:
        """The opt-in demo-token source is forwarded like any other, and the
        hand-off to the policy is one atomic selection: nothing of the warm
        start may follow it into the deployment."""
        self.router.select("warmstart")
        self.router.resume()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self.router.submit_warmstart(b"warm")
            if self.output.poll(100):
                break
        self.assertEqual(self.receive(), b"warm")

        self.router.select("groot")
        self.router.resume()
        time.sleep(0.1)
        self.assertFalse(self.router.submit_warmstart(b"late-warm"))
        self.groot.send(b"policy")
        self.assertEqual(self.receive(), b"policy")
        self.assertFalse(self.output.poll(100))

    def test_halt_closes_the_warm_start_gate(self) -> None:
        self.router.select("warmstart")
        self.router.resume()
        self.router.halt()
        self.assertFalse(self.router.submit_warmstart(b"warm"))
        with self.assertRaises(ValueError):
            self.router.select("elsewhere")

    def test_a_failing_recorder_cannot_stop_the_action_path(self) -> None:
        """A recorder exception once killed this router thread on the first
        GR00T message, and the whole episode ran with no commands at all."""
        class Exploding:
            def sample_frame(self, source: str) -> None:
                raise IndexError("too many indices for array")

            def applied_action(self, source: str, payload: bytes) -> None:
                raise IndexError("too many indices for array")

        self.router.telemetry = Exploding()
        self.router.select("groot")
        self.router.resume()
        time.sleep(0.1)
        self.groot.send(b"groot")
        self.assertEqual(self.receive(), b"groot")
        self.groot.send(b"groot-again")
        self.assertEqual(self.receive(), b"groot-again")
        self.assertIn("IndexError", self.router.status()["telemetry_error"] or "")
        self.assertEqual(self.router.status()["sent"], 2)


if __name__ == "__main__":
    unittest.main()
