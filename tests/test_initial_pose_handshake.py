"""The opt-in initial-pose handshake: identity, cadence, gating and the default.

The command must be the VLA client's *own* initial pose -- its
``LATENT_INITIAL_MOTION_TOKEN`` with both Dex3 hands open -- published through the
router at the control rate until the policy takes the port.  These cases pin the
identity, the hold and the atomic hand-off; where the pinned SONIC tree is
importable (the container image) the payload is also checked byte for byte
against the client's own packer.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from humanoid_lab.psi0_bridge.initial_pose import (
    CLIENT_CONSTANT,
    CLIENT_MODULE,
    InitialPoseError,
    initial_pose_handshake,
    load_initial_pose_command,
)
from humanoid_lab.psi0_bridge.warmstart import WarmStartStream, stream_from_tokens

try:
    import zmq
except ImportError:  # pragma: no cover
    zmq = None


#: Stand-in for the client's constant: the real head, deterministic tail.
def fake_client_constant() -> np.ndarray:
    return np.array([-0.0625, 0.0, -0.0625, -0.125] + [0.0625] * 60, dtype=np.float32)


#: Stand-in for the client's protocol v4 packer: fixed-width, so a decoded
#: payload shows both the token it carried and the tick it was sent on.
def fake_packer(token, left_hand, right_hand, index) -> bytes:
    return (
        np.asarray(token, dtype=np.float32).reshape(-1).tobytes()
        + np.asarray(left_hand, dtype=np.float32).reshape(-1).tobytes()
        + np.asarray(right_hand, dtype=np.float32).reshape(-1).tobytes()
        + np.asarray([index], dtype=np.int64).tobytes()
    )


def unpack(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    token = np.frombuffer(payload, dtype=np.float32, count=64, offset=0)
    left = np.frombuffer(payload, dtype=np.float32, count=7, offset=256)
    right = np.frombuffer(payload, dtype=np.float32, count=7, offset=284)
    index = int(np.frombuffer(payload, dtype=np.int64, count=1, offset=312)[0])
    return token, left, right, index


class RecordingRouter:
    """The router surface the stream uses, without a socket."""

    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.payloads: list[bytes] = []
        self.selected: list[str] = []
        self.resumed = 0
        self.halted = 0

    def submit_initial_pose(self, payload: bytes) -> bool:
        if not self.accept:
            return False
        self.payloads.append(payload)
        return True

    def submit_warmstart(self, payload: bytes) -> bool:
        if not self.accept:
            return False
        self.payloads.append(payload)
        return True


class LoadTest(unittest.TestCase):
    def test_the_command_is_the_clients_token_with_open_hands(self) -> None:
        command = load_initial_pose_command(provider=fake_client_constant)
        self.assertEqual(command.token.shape, (64,))
        self.assertEqual(command.token.dtype, np.float32)
        self.assertTrue(np.array_equal(command.token, fake_client_constant().astype(np.float32)))
        self.assertTrue(np.allclose(command.left_hand_joints, 0.0))
        self.assertTrue(np.allclose(command.right_hand_joints, 0.0))
        self.assertEqual(command.control_hz, 50.0)

    def test_the_summary_names_its_source(self) -> None:
        command = load_initial_pose_command(provider=fake_client_constant)
        summary = command.summary()
        self.assertEqual(summary["module"], CLIENT_MODULE)
        self.assertEqual(summary["constant"], CLIENT_CONSTANT)
        self.assertEqual(summary["token_dim"], 64)
        self.assertEqual(summary["token_sha256_f32"], command.token_sha256)
        self.assertIn("open", summary["hand_joints"])

    def test_a_wrong_or_missing_constant_fails_before_a_session(self) -> None:
        with self.assertRaises(InitialPoseError):
            load_initial_pose_command(provider=lambda: np.zeros(63))
        with self.assertRaises(InitialPoseError):
            load_initial_pose_command(provider=lambda: np.full(64, np.nan))
        with self.assertRaises(InitialPoseError):
            load_initial_pose_command(provider=lambda: [1.0] * 64, control_hz=0.0)

        def broken():
            raise ImportError("no gear_sonic here")

        with self.assertRaises(InitialPoseError) as ctx:
            load_initial_pose_command(provider=broken)
        self.assertIn(CLIENT_CONSTANT, str(ctx.exception))

    def test_the_stream_is_one_tick_that_carries_the_command(self) -> None:
        first = load_initial_pose_command(provider=fake_client_constant)
        stream = first.stream()
        self.assertEqual(stream.ticks, 1)
        self.assertEqual(stream.control_hz, 50.0)
        self.assertEqual(stream.tokens.shape, (1, 64))
        self.assertEqual(stream.left_hand_joints.shape, (1, 7))
        self.assertEqual(stream.right_hand_joints.shape, (1, 7))
        self.assertEqual(stream.source["token_sha256_f32"], first.token_sha256)


class StreamTest(unittest.TestCase):
    def handshake(self, router: RecordingRouter) -> WarmStartStream:
        command = load_initial_pose_command(provider=fake_client_constant)
        return initial_pose_handshake(router, command=command, packer=fake_packer)

    def wait_for(self, predicate, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_it_holds_the_same_token_at_the_control_rate_until_halted(self) -> None:
        router = RecordingRouter()
        handshake = self.handshake(router)
        handshake.arm(0.0)
        self.assertTrue(self.wait_for(lambda: len(router.payloads) >= 20), "no ticks published")
        handshake.halt()
        self.assertEqual(handshake.source, "initial_pose")

        decoded = [unpack(payload) for payload in router.payloads]
        tokens = np.array([row[0] for row in decoded])
        self.assertTrue(np.array_equal(tokens, np.broadcast_to(tokens[0], tokens.shape)))
        self.assertTrue(np.array_equal(tokens[0], fake_client_constant()))
        self.assertTrue(all(np.allclose(row[1], 0.0) and np.allclose(row[2], 0.0) for row in decoded))
        indices = [row[3] for row in decoded]
        self.assertEqual(indices, list(range(len(indices))), "the tick index must advance")

        ticks = len(router.payloads)
        period = 1.0 / 50.0
        time.sleep(5 * period + 0.05)
        self.assertEqual(len(router.payloads), ticks, "halt must stop the stream")
        self.assertFalse(handshake.armed)

    def test_a_closed_router_gate_does_not_raise(self) -> None:
        router = RecordingRouter(accept=False)
        handshake = self.handshake(router)
        handshake.arm(0.0)
        time.sleep(0.2)
        handshake.halt()
        self.assertEqual(router.payloads, [])
        self.assertIsNone(handshake.status()["error"])

    def test_the_default_stream_is_still_the_warmstart_source(self) -> None:
        """The handshake's changes must not move the demonstration warm start."""
        router = RecordingRouter()
        stream = stream_from_tokens([[0.5] * 64], [[0.0] * 7], [[0.0] * 7])
        warm = WarmStartStream(stream, router, packer=fake_packer,
                               log=lambda message: None)
        self.assertEqual(warm.source, "warmstart")
        warm.arm(0.0)
        self.assertTrue(self.wait_for(lambda: len(router.payloads) >= 2))
        warm.halt()


def _client_tree_available() -> bool:
    try:
        import importlib

        importlib.import_module(CLIENT_MODULE)
    except Exception:  # noqa: BLE001 - absence is the normal case on the host
        return False
    return True


@unittest.skipUnless(_client_tree_available(), "the pinned SONIC tree is not importable here")
class ClientIdentityTest(unittest.TestCase):
    """The command *is* the client's, verified against the client's own code."""

    def test_it_is_the_clients_constant_packed_by_the_clients_packer(self) -> None:
        from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

        from humanoid_lab.psi0_bridge.warmstart import default_packer

        command = load_initial_pose_command()
        self.assertTrue(np.array_equal(
            command.token, np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)))
        # The message publish_initial_pose() sends on the client's own socket.
        expected = pack_pose_message(
            {
                "token_state": np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32).reshape(1, 64),
                "frame_index": np.array([0], dtype=np.int64),
                "left_hand_joints": np.zeros(7, dtype=np.float32).reshape(1, 7),
                "right_hand_joints": np.zeros(7, dtype=np.float32).reshape(1, 7),
            },
            topic="pose", version=4,
        )
        packed = default_packer()(command.token, command.left_hand_joints,
                                  command.right_hand_joints, 0)
        self.assertEqual(packed, expected)


@unittest.skipIf(zmq is None, "pyzmq is not installed")
class RouterSourceTest(unittest.TestCase):
    def test_initial_pose_is_a_gated_source_like_the_others(self) -> None:
        import socket

        import zmq as zmq_module

        from humanoid_lab.psi0_bridge.action_router import ActionRouter

        def free_port() -> int:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                return sock.getsockname()[1]

        public_port, groot_port = free_port(), free_port()
        context = zmq_module.Context()
        groot = context.socket(zmq_module.PUB)
        groot.setsockopt(zmq_module.LINGER, 0)
        groot.bind(f"tcp://127.0.0.1:{groot_port}")
        output = context.socket(zmq_module.SUB)
        output.setsockopt(zmq_module.LINGER, 0)
        output.setsockopt(zmq_module.SUBSCRIBE, b"")
        output.connect(f"tcp://127.0.0.1:{public_port}")
        router = ActionRouter(public_endpoint=f"tcp://127.0.0.1:{public_port}",
                              groot_endpoint=f"tcp://127.0.0.1:{groot_port}")
        try:
            self.assertIn("initial_pose", router.SOURCES)
            router.select("initial_pose")
            router.resume()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                router.submit_initial_pose(b"initial-pose")
                if output.poll(100):
                    break
            self.assertEqual(output.recv(), b"initial-pose")

            # The hand-off: once the policy source is selected, neither the
            # handshake nor a late tick of it may reach the port again.
            router.select("groot")
            router.resume()
            time.sleep(0.1)
            self.assertFalse(router.submit_initial_pose(b"late-initial-pose"))
            groot.send(b"policy")
            self.assertTrue(output.poll(1000))
            self.assertEqual(output.recv(), b"policy")
            self.assertFalse(output.poll(100))

            router.halt()
            self.assertFalse(router.submit_initial_pose(b"halted"))
        finally:
            router.close()
            groot.close()
            output.close()
            context.term()


if __name__ == "__main__":
    unittest.main()
