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


if __name__ == "__main__":
    unittest.main()
