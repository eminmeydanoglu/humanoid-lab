"""The demo-token decoder replay: observation assembly, orders, and metric alignment.

The replay only means something if the 994D observation is laid out exactly the way
the pinned deployment lays it out, if the joint-order hypotheses really are distinct
permutations, and if the alignment metrics measure what they claim.  These tests pin
those three things on synthetic input, with no session, no GPU and no dataset.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "demo-token-decoder-replay.py"
SONIC_DEPLOY = Path("/opt/src/sonic/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref")
OBS_CONFIG = Path("/data/models/sonic/sonic_v1_1/observation_config.yaml")
MODEL = Path("/data/models/sonic/sonic_v1_1/model_decoder.onnx")


def load_replay():
    spec = importlib.util.spec_from_file_location("demo_token_decoder_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


REPLAY = load_replay()


def synthetic_semantics() -> "REPLAY.DeploymentSemantics":
    """29 joints, because the deployment's action and target rules are 29-wide by construction."""
    isaaclab_to_mujoco = np.roll(np.arange(29), 1)
    mujoco_to_isaaclab = np.argsort(isaaclab_to_mujoco)
    layout = [("token_state", 0, 64), ("his_base_angular_velocity_10frame_step1", 64, 30),
              ("his_body_joint_positions_10frame_step1", 94, 290),
              ("his_body_joint_velocities_10frame_step1", 384, 290),
              ("his_last_actions_10frame_step1", 674, 290), ("his_gravity_dir_10frame_step1", 964, 30)]
    return REPLAY.DeploymentSemantics(
        deploy_root=Path("/nonexistent"), control_dt=0.02, isaaclab_to_mujoco=isaaclab_to_mujoco,
        mujoco_to_isaaclab=mujoco_to_isaaclab, action_scale=np.full(29, 0.5),
        default_angles=np.arange(29, dtype=float), registry={}, layout=layout, input_dim=994, token_dim=64,
        source_sha256={})


class ObservationLayoutTest(unittest.TestCase):
    def test_blocks_land_at_the_offsets_the_config_declares(self) -> None:
        semantics = synthetic_semantics()
        token = np.arange(64, dtype=float)
        q = np.arange(290, dtype=float).reshape(10, 29)
        observation = REPLAY.build_observation(token, {"q": q, "dq": q, "last_action": q, "gravity": q[:, :3],
                                                       "angvel": q[:, :3]}, semantics.layout)
        self.assertEqual(observation.shape, (994,))
        np.testing.assert_array_equal(observation[:64], token)
        np.testing.assert_array_equal(observation[94:384], q.reshape(-1))
        np.testing.assert_array_equal(observation[674:964], q.reshape(-1))

    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        semantics = synthetic_semantics()
        with self.assertRaises(REPLAY.ReplayError):
            REPLAY.build_observation(np.zeros(64), {"q": np.zeros((10, 28)), "dq": np.zeros((10, 29)),
                                                   "last_action": np.zeros((10, 29)), "gravity": np.zeros((10, 3)),
                                                   "angvel": np.zeros((10, 3))}, semantics.layout)


class HistoryTest(unittest.TestCase):
    def test_windows_end_at_the_current_tick_and_pad_the_front(self) -> None:
        values = np.arange(12, dtype=float).reshape(6, 2)
        windows = REPLAY.history_windows(values, frames=3)
        np.testing.assert_array_equal(windows[0], np.array([[0, 0], [0, 0], [0, 1]]))
        np.testing.assert_array_equal(windows[1], np.array([[0, 0], [0, 1], [2, 3]]))
        np.testing.assert_array_equal(windows[5], np.array([[6, 7], [8, 9], [10, 11]]))

    def test_the_action_window_ends_one_tick_before_the_state_window(self) -> None:
        semantics = synthetic_semantics()
        tokens = np.zeros((4, 64))
        q = np.arange(116, dtype=float).reshape(4, 29)
        decoder = StubDecoder()
        REPLAY.replay_stream(decoder, semantics, tokens=tokens, q_measured=q, dq_measured=q,
                             gravity=np.zeros((4, 3)), angvel=np.zeros((4, 3)),
                             hypothesis="identity", last_action_source="measured")
        last_block = slice(674, 964)
        np.testing.assert_array_equal(decoder.observations[0, last_block], np.zeros(290))
        window = decoder.observations[3, last_block].reshape(10, 29)
        np.testing.assert_array_equal(window[-1], (q[2] - semantics.default_angles) / semantics.action_scale)

    def test_shift_series_holds_the_edge_values(self) -> None:
        values = np.arange(10, dtype=float).reshape(5, 2)
        np.testing.assert_array_equal(REPLAY.shift_series(values, 0), values)
        np.testing.assert_array_equal(REPLAY.shift_series(values, 2)[:2], np.repeat(values[:1], 2, axis=0))
        np.testing.assert_array_equal(REPLAY.shift_series(values, -2)[-2:], np.repeat(values[-1:], 2, axis=0))


class PermutationTest(unittest.TestCase):
    def test_the_hypotheses_are_distinct_permutations(self) -> None:
        semantics = synthetic_semantics()
        permutations = {name: tuple(semantics.permutation(name)) for name in REPLAY.PERM_HYPOTHESES}
        self.assertEqual(len(set(permutations.values())), len(REPLAY.PERM_HYPOTHESES))

    def test_history_order_inverts_the_permutation(self) -> None:
        semantics = synthetic_semantics()
        for name in REPLAY.PERM_HYPOTHESES:
            permutation = semantics.permutation(name)
            order = semantics.history_order(name)
            np.testing.assert_array_equal(order[permutation], np.arange(len(permutation)))

    def test_target_and_action_round_trip(self) -> None:
        semantics = synthetic_semantics()
        action = np.linspace(-0.5, 0.5, 29)[None, :]
        for name in REPLAY.PERM_HYPOTHESES:
            target = REPLAY.action_to_target(action, semantics, name)
            np.testing.assert_allclose(REPLAY.target_to_action(target, semantics, name), action, atol=1e-12)


class QuantisationTest(unittest.TestCase):
    def test_fsq_grid_is_enforced(self) -> None:
        values = np.array([-0.9, -0.03, 0.03, 0.59, 0.7])
        quantised = REPLAY.fsq_quantize(values)
        np.testing.assert_allclose(quantised, np.array([-0.625, 0.0, 0.0, 0.5625, 0.625]))
        np.testing.assert_allclose(quantised / REPLAY.FSQ_STEP, np.round(quantised / REPLAY.FSQ_STEP))


class QuaternionTest(unittest.TestCase):
    def test_identity_attitude_reads_gravity_down(self) -> None:
        np.testing.assert_allclose(REPLAY.gravity_from_quaternion(np.array([[1.0, 0.0, 0.0, 0.0]])),
                                   np.array([[0.0, 0.0, -1.0]]), atol=1e-12)

    def test_a_pitched_pelvis_tilts_the_gravity_vector(self) -> None:
        half = np.pi / 4
        quaternion = np.array([[np.cos(half), np.sin(half), 0.0, 0.0]])  # +90 degrees about x
        np.testing.assert_allclose(REPLAY.gravity_from_quaternion(quaternion), np.array([[0.0, -1.0, 0.0]]),
                                   atol=1e-12)

    def test_body_angular_velocity_of_a_constant_yaw_rate(self) -> None:
        rate = 0.7
        times = np.arange(20) * 0.02
        half = rate * times / 2.0
        quaternion = np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=1)
        omega = REPLAY.quaternion_angular_velocity(quaternion, np.full(len(times), 0.02))
        np.testing.assert_allclose(omega[:, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(omega[:, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(omega[:, 2], rate, atol=1e-4)  # a finite-difference estimate


class AlignmentTest(unittest.TestCase):
    def test_optimal_lag_finds_a_pure_delay(self) -> None:
        measured = np.stack([np.arange(60.0), np.zeros(60), np.zeros(60)], axis=1)
        delayed = np.concatenate([np.repeat(measured[:1], 7, axis=0), measured[:-7]], axis=0)
        lag, rmse = REPLAY.optimal_lag(measured, delayed)
        self.assertEqual(lag, 7)
        self.assertLess(rmse, 1e-9)

    def test_alignment_uses_the_shared_span(self) -> None:
        measured = np.arange(10.0).reshape(-1, 1)
        decoded = np.arange(10.0).reshape(-1, 1) + 100
        first, second = REPLAY.align_pair(measured, decoded, 3)
        self.assertEqual(len(first), len(second))
        self.assertEqual(second[0, 0], 103.0)
        self.assertEqual(first[0, 0], 0.0)


class StubDecoder:
    """A decoder that returns a fixed action; enough to inspect the observations it is fed."""

    input_dim = 994

    def __init__(self) -> None:
        self.observations: np.ndarray | None = None

    def run(self, observations: np.ndarray) -> np.ndarray:
        frames = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        self.observations = frames if self.observations is None else np.concatenate(
            [self.observations, frames], axis=0)
        return np.zeros((frames.shape[0], 29), dtype=np.float64)


class DeploymentSourceTest(unittest.TestCase):
    """The pinned deployment is present in this image; the parse must match it."""

    def setUp(self) -> None:
        if not (SONIC_DEPLOY.is_dir() and OBS_CONFIG.is_file()):
            self.skipTest("pinned SONIC deployment source is not mounted")

    def test_layout_and_control_period_come_out_of_the_source(self) -> None:
        semantics = REPLAY.parse_deployment_semantics(SONIC_DEPLOY, OBS_CONFIG, 994)
        self.assertEqual(semantics.input_dim, 994)
        self.assertEqual(semantics.token_dim, 64)
        self.assertAlmostEqual(semantics.control_dt, 0.02, places=9)
        self.assertEqual([name for name, _, _ in semantics.layout],
                         ["token_state", "his_base_angular_velocity_10frame_step1",
                          "his_body_joint_positions_10frame_step1", "his_body_joint_velocities_10frame_step1",
                          "his_last_actions_10frame_step1", "his_gravity_dir_10frame_step1"])
        self.assertAlmostEqual(float(semantics.default_angles[15]), 0.2, places=9)
        np.testing.assert_array_equal(np.argsort(semantics.isaaclab_to_mujoco), semantics.mujoco_to_isaaclab)

    def test_a_wrong_model_width_is_refused(self) -> None:
        with self.assertRaises(REPLAY.ReplayError):
            REPLAY.parse_deployment_semantics(SONIC_DEPLOY, OBS_CONFIG, 993)


class DecoderContractTest(unittest.TestCase):
    def setUp(self) -> None:
        if not MODEL.is_file():
            self.skipTest("pinned decoder ONNX is not present")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime is not installed here")

    def test_the_decoder_is_the_pinned_one_and_is_deterministic(self) -> None:
        provenance = Path("/data/models/sonic/MODEL_PROVENANCE.json")
        pinned = None
        if provenance.is_file():
            import json

            document = json.loads(provenance.read_text(encoding="utf-8"))
            pinned = next(entry["sha256"] for entry in document["files"]
                          if entry["path"].endswith("model_decoder.onnx"))
        decoder = REPLAY.Decoder(MODEL, pinned, input_dim=994)
        observation = np.zeros((1, 994), dtype=np.float32)
        first = decoder.run(observation)
        second = decoder.run(observation)
        self.assertEqual(first.shape, (1, 29))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.allclose(first, 0.0))


if __name__ == "__main__":
    unittest.main()
