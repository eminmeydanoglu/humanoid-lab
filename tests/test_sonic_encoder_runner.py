"""Real ONNX encoder tests: contract, packing, repeatability, golden tokens.

These run in the environment that pins onnxruntime (``./dev.sh sonic-tests`` runs
this module with the sonic-sim interpreter).  Exact parity with the C++ deployment
is not testable here: ``motion_anchor_orientation_heading*`` consumes the *live*
measured base quaternion, which none of the datasets records.  What is tested is
everything that is deterministic — the pinned model contract, the official input
packing, run-to-run repeatability, and golden tokens for a fixture whose
orientation block depends on the heading policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from humanoid_lab.datasets.sonic.encoder_observation import (
    ENCODER_INPUT_DIM,
    FUTURE_OFFSETS,
    OrientationPolicy,
    build_g1_encoder_observation,
)
from humanoid_lab.datasets.sonic.encoder_runner import (
    ENCODER_INPUT_NAME,
    ENCODER_OUTPUT_NAME,
    ONNXRUNTIME_HINT,
    EncoderContractError,
    G1EncoderRunner,
    require_onnxruntime,
)
from humanoid_lab.datasets.sonic.heading import quat_from_angle_axis, quat_mul
from humanoid_lab.datasets.sonic.provenance import sha256_file
from humanoid_lab.datasets.sonic.schema import CanonicalEpisode

MODEL_DIR = Path("/data/models/sonic-isaac/sonic_v1_1")
MODEL_PATH = MODEL_DIR / "model_encoder.onnx"
GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures/sonic_encoder_golden.json"
GOLDEN_FRAMES = 5

try:  # pragma: no cover - depends on the interpreter
    import onnxruntime  # noqa: F401

    HAVE_ONNXRUNTIME = True
except ImportError:  # pragma: no cover
    HAVE_ONNXRUNTIME = False


def yaw(angle: float) -> np.ndarray:
    return quat_from_angle_axis(angle, (0.0, 0.0, 1.0))


def fixture_episode(frames: int = GOLDEN_FRAMES) -> CanonicalEpisode:
    """Deterministic fixture: yawing root plus moving arms, so policy matters."""
    timestamps = np.arange(frames, dtype=np.float64) / 50.0
    joint_pos = np.zeros((frames, 29), dtype=np.float32)
    joint_pos[:, 15:29] = np.linspace(-0.4, 0.6, frames)[:, None]
    joint_pos[:, 0] = np.linspace(0.0, 0.1, frames)
    body_quat = np.stack([yaw(0.15 * index) for index in range(frames)])
    return CanonicalEpisode(
        timestamps=timestamps,
        joint_pos=joint_pos,
        joint_vel=np.gradient(joint_pos, axis=0).astype(np.float32) * 50.0,
        body_quat_wxyz=body_quat.astype(np.float32),
        body_pos=np.tile([0.0, 0.0, 0.79], (frames, 1)).astype(np.float32),
        left_hand_joints=np.zeros((frames, 7), dtype=np.float32),
        right_hand_joints=np.zeros((frames, 7), dtype=np.float32),
    )


def fixture_observation() -> np.ndarray:
    observation, _ = build_g1_encoder_observation(fixture_episode())
    return observation


@unittest.skipUnless(HAVE_ONNXRUNTIME, "onnxruntime is not installed in this interpreter")
@unittest.skipUnless(MODEL_PATH.is_file(), "pinned encoder model is not mounted")
class EncoderRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = G1EncoderRunner(MODEL_PATH)
        cls.observation = fixture_observation()
        cls.tokens = cls.runner.encode_frames(cls.observation)

    def test_model_contract_is_the_pinned_one(self) -> None:
        info = self.runner.info
        self.assertEqual(info.input_name, ENCODER_INPUT_NAME)
        self.assertEqual(info.output_name, ENCODER_OUTPUT_NAME)
        self.assertEqual(info.input_shape, (1, ENCODER_INPUT_DIM))
        self.assertEqual(info.output_shape, (1, 64))
        self.assertEqual(info.input_type, "tensor(float)")
        self.assertEqual(info.providers, ("CPUExecutionProvider",))
        self.assertEqual(info.sha256, sha256_file(MODEL_PATH))

    def test_checksum_mismatch_fails_closed(self) -> None:
        with self.assertRaises(EncoderContractError):
            G1EncoderRunner(MODEL_PATH, expected_sha256="0" * 64)

    def test_batch_is_one_and_frames_loop_matches_single_frame_calls(self) -> None:
        single = np.stack([self.runner.encode(row) for row in self.observation])
        np.testing.assert_array_equal(single, self.tokens)
        self.assertEqual(self.tokens.shape, (GOLDEN_FRAMES, 64))

    def test_repeatability_is_bitwise(self) -> None:
        again = self.runner.encode_frames(self.observation)
        np.testing.assert_array_equal(again, self.tokens)
        self.assertTrue(np.isfinite(self.tokens).all())

    def test_input_validation(self) -> None:
        with self.assertRaises(EncoderContractError):
            self.runner.encode(np.zeros(ENCODER_INPUT_DIM - 1, dtype=np.float32))
        with self.assertRaises(EncoderContractError):
            self.runner.encode(np.zeros((2, ENCODER_INPUT_DIM), dtype=np.float32))
        broken = self.observation[0].copy()
        broken[7] = np.nan
        with self.assertRaises(EncoderContractError):
            self.runner.encode(broken)

    def test_official_packing_layout(self) -> None:
        """C++ packing byte/value parity: fixed offsets, float32, row-major."""
        row = self.observation[0]
        packed = np.ascontiguousarray(row, dtype=np.float32)
        self.assertEqual(packed.nbytes, ENCODER_INPUT_DIM * 4)
        self.assertEqual(packed.dtype.byteorder in ("<", "="), True)
        episode = fixture_episode()
        for offset, name in ((0, "encoder_mode_4"), (4, "joint_positions")):
            self.assertTrue(np.isfinite(row[offset]), name)
        np.testing.assert_allclose(
            packed[4 : 4 + 29], episode.joint_pos[0], atol=1e-6
        )
        np.testing.assert_allclose(
            packed[4 + 29 : 4 + 58], episode.joint_pos[min(int(FUTURE_OFFSETS[1]), GOLDEN_FRAMES - 1)], atol=1e-6
        )
        np.testing.assert_array_equal(packed[644:], np.zeros(ENCODER_INPUT_DIM - 644, dtype=np.float32))
        self.assertEqual(
            hashlib.sha256(packed.tobytes()).hexdigest(),
            hashlib.sha256(np.asarray(row, dtype=np.float32).tobytes()).hexdigest(),
        )

    def test_heading_policy_changes_tokens(self) -> None:
        legacy, _ = build_g1_encoder_observation(
            fixture_episode(), policy=OrientationPolicy.LEGACY_RAW_WORLD
        )
        legacy_tokens = self.runner.encode_frames(legacy)
        self.assertGreater(float(np.abs(legacy_tokens - self.tokens).max()), 1e-3)

    def test_golden_tokens(self) -> None:
        if os.environ.get("SONIC_ENCODER_GOLDEN_UPDATE") == "1":  # pragma: no cover - manual refresh
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(
                json.dumps(
                    {
                        "note": "golden tokens of tests/test_sonic_encoder_runner.py fixture_observation",
                        "model_sha256": self.runner.info.sha256,
                        "onnxruntime_version": self.runner.info.onnxruntime_version,
                        "orientation_policy": OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME.value,
                        "tokens": self.tokens.tolist(),
                    },
                    indent=1,
                )
                + "\n",
                encoding="utf-8",
            )
            self.skipTest("golden fixture refreshed")
        self.assertTrue(GOLDEN_PATH.is_file(), f"missing golden fixture {GOLDEN_PATH}")
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(golden["orientation_policy"], OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME.value)
        expected = np.asarray(golden["tokens"], dtype=np.float32)
        self.assertEqual(expected.shape, self.tokens.shape)
        np.testing.assert_allclose(self.tokens, expected, atol=1e-5)


class OnnxruntimeDiagnosticsTest(unittest.TestCase):
    def test_missing_onnxruntime_explains_the_right_environment(self) -> None:
        with unittest.mock.patch("builtins.__import__", side_effect=ImportError("no module")):
            with self.assertRaises(RuntimeError) as context:
                require_onnxruntime()
        self.assertIn("/opt/venvs/sonic-sim", str(context.exception))
        self.assertIn("sonic-encode", str(context.exception))
        self.assertEqual(str(context.exception), ONNXRUNTIME_HINT)


if __name__ == "__main__":
    unittest.main()
