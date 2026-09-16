"""Tests for the GRAIL kinematic replay path.

The pure-python checks run anywhere.  The dataset checks need numpy, joblib and
the real pickup_table release, so they skip when those are absent; run them in
the container:

    docker exec <project>-dev bash -lc \
        'source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && \
         cd /workspace/humanoid-lab && PYTHONPATH=src python -m unittest tests.simulator.test_grail_replay'
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.replay import (  # noqa: E402
    EXTERNAL_RESOLUTION,
    QUAT_CONVENTION,
    RENDER_FRAME_MULTIPLIER,
    TRAJECTORY_JOINT_NAMES,
    ReplayError,
    detect_grasp_source_frame,
    interpolated_frame,
    joint_layout,
    joint_positions,
    joint_step_summary,
    load_sequence,
    validate_sequence_key,
)

try:  # The dataset checks need the GRAIL conversion dependencies.
    import joblib
    import numpy as np

    HAVE_DEPS = True
except ImportError:  # pragma: no cover - host interpreters without numpy
    HAVE_DEPS = False

DATA_ROOT = ROOT / "data/datasets/grail/data/pickup_table"
APPLE_KEY = "pickup_table__apple_0__000"
# MuJoCo -> IsaacLab body order that prepare_vis_shard applies, written out here
# so the test does not mirror the implementation.
MUJOCO_TO_ISAACLAB_BODY = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)

requires_deps = unittest.skipUnless(HAVE_DEPS, "numpy/joblib unavailable")
requires_dataset = unittest.skipUnless(DATA_ROOT.is_dir(), f"dataset missing: {DATA_ROOT}")


class SequenceKeyTests(unittest.TestCase):
    def test_plain_stems_are_accepted(self) -> None:
        validate_sequence_key(APPLE_KEY)

    def test_paths_and_empty_keys_are_rejected(self) -> None:
        for key in ("", ".", "..", "../robot/x", "a/b", "/abs"):
            with self.subTest(key=key), self.assertRaises(ReplayError):
                validate_sequence_key(key)


class GraspDetectionTests(unittest.TestCase):
    """The grasp event is the release's own discrete right-hand command."""

    def test_first_open_to_closed_transition_is_the_grasp(self) -> None:
        series = [-1.0] * 5 + [1.0] * 3
        self.assertEqual(detect_grasp_source_frame(series), 5)

    def test_multiple_close_transitions_are_ambiguous(self) -> None:
        with self.assertRaisesRegex(ReplayError, "ambiguous"):
            detect_grasp_source_frame([-1.0, -1.0, 1.0, -1.0, 1.0])

    def test_series_that_never_closes_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReplayError, "never closes"):
            detect_grasp_source_frame([-1.0] * 4)

    def test_series_that_starts_closed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReplayError, "already closed"):
            detect_grasp_source_frame([1.0, 1.0, 1.0])

    def test_non_discrete_values_are_rejected(self) -> None:
        for series in ([-1.0, 0.0, 1.0], [-1.0, 0.5, 1.0], [-1.0, 2.0]):
            with self.subTest(series=series), self.assertRaisesRegex(ReplayError, "discrete"):
                detect_grasp_source_frame(series)

    def test_non_finite_values_are_rejected(self) -> None:
        for series in ([-1.0, float("nan"), 1.0], [-1.0, float("inf")]):
            with self.subTest(series=series), self.assertRaisesRegex(ReplayError, "finite"):
                detect_grasp_source_frame(series)

    def test_too_short_and_nested_series_are_rejected(self) -> None:
        for series in ([], [-1.0]):
            with self.subTest(series=series), self.assertRaisesRegex(ReplayError, "at least 2"):
                detect_grasp_source_frame(series)
        with self.assertRaisesRegex(ReplayError, "one-dimensional"):
            detect_grasp_source_frame([[-1.0, 1.0], [-1.0, 1.0]])

    def test_key_prefixes_the_error(self) -> None:
        with self.assertRaisesRegex(ReplayError, "^pickup_table__unit_0__000: "):
            detect_grasp_source_frame([-1.0], key="pickup_table__unit_0__000")


class JointMappingTests(unittest.TestCase):
    def test_release_order_is_43_unique_joints(self) -> None:
        self.assertEqual(len(TRAJECTORY_JOINT_NAMES), 43)
        self.assertEqual(len(set(TRAJECTORY_JOINT_NAMES)), 43)

    def test_hand_columns_follow_the_recording_articulation_order(self) -> None:
        # The exporter appends hand_dof_pos verbatim from robot.data.joint_pos, so
        # the columns are in the recording articulation's order - not SONIC's
        # declared G1_HAND_JOINTS list (third_party/GRAIL/imports/SONIC/gear_sonic/
        # envs/wrapper/manager_env_wrapper.py: "Hand DOFs are the last N joints
        # (in Isaac order, not G1_HAND_JOINTS order)"). The profile asset reports
        # the same order, which is why the mapping below is normally positional.
        self.assertEqual(
            TRAJECTORY_JOINT_NAMES[29:],
            (
                "left_hand_index_0_joint",
                "left_hand_middle_0_joint",
                "left_hand_thumb_0_joint",
                "right_hand_index_0_joint",
                "right_hand_middle_0_joint",
                "right_hand_thumb_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_1_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_thumb_2_joint",
            ),
        )

    def test_matching_articulation_maps_positionally(self) -> None:
        self.assertEqual(joint_layout(TRAJECTORY_JOINT_NAMES), list(range(43)))

    def test_reordered_articulation_is_mapped_by_name(self) -> None:
        names = list(TRAJECTORY_JOINT_NAMES)
        names[0], names[1], names[2] = names[1], names[2], names[0]  # a 3-cycle
        layout = joint_layout(names)
        for column, index in enumerate(layout):
            self.assertEqual(names[index], TRAJECTORY_JOINT_NAMES[column])
        self.assertEqual(sorted(layout), list(range(43)))

    @requires_deps
    def test_positions_place_each_column_on_its_own_joint(self) -> None:
        # A 3-cycle is not its own inverse, so a gather would pass the array
        # through a wrong permutation while a scatter places every column.
        layout = [2, 0, 1]
        dof = np.array([10.0, 11.0, 12.0], dtype=np.float32)
        positions = joint_positions(dof, layout)
        self.assertEqual(list(positions), [11.0, 12.0, 10.0])
        for column, index in enumerate(layout):
            self.assertEqual(positions[index], dof[column])

    def test_wrong_joint_count_missing_names_and_duplicates_fail(self) -> None:
        with self.assertRaises(ReplayError):
            joint_layout(list(TRAJECTORY_JOINT_NAMES[:42]))
        renamed = ["other_joint" if name == "left_knee_joint" else name for name in TRAJECTORY_JOINT_NAMES]
        with self.assertRaises(ReplayError):
            joint_layout(renamed)
        duplicated = list(TRAJECTORY_JOINT_NAMES)
        duplicated[-1] = duplicated[-2]
        with self.assertRaises(ReplayError):
            joint_layout(duplicated)


@requires_deps
class InterpolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = {
            "total_frames": 2,
            "dof_pos": np.stack([np.zeros(43), np.full(43, 0.4)]).astype(np.float32),
            "root_pos_w": np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], dtype=np.float32),
            "root_quat_w": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "object_pos_w": np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
            # The opposite sign is the same rotation and must not take a long path.
            "object_quat_w": np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        }

    def test_source_keyframes_are_preserved_and_midpoint_is_inserted(self) -> None:
        self.assertEqual(RENDER_FRAME_MULTIPLIER, 2)
        np.testing.assert_allclose(interpolated_frame(self.trajectory, 0)["dof_pos"], 0.0)
        np.testing.assert_allclose(interpolated_frame(self.trajectory, 1)["dof_pos"], 0.2)
        np.testing.assert_allclose(interpolated_frame(self.trajectory, 2)["dof_pos"], 0.4)
        np.testing.assert_allclose(interpolated_frame(self.trajectory, 3)["dof_pos"], 0.4)
        np.testing.assert_allclose(
            interpolated_frame(self.trajectory, 1)["root_pos_w"], [1.0, 2.0, 3.0]
        )

    def test_quaternions_use_normalized_shortest_arc_interpolation(self) -> None:
        midpoint = interpolated_frame(self.trajectory, 1)
        np.testing.assert_allclose(midpoint["root_quat_w"], [2**-0.5, 0.0, 0.0, 2**-0.5], atol=1e-6)
        np.testing.assert_allclose(midpoint["object_quat_w"], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(midpoint["root_quat_w"]), 1.0, atol=1e-6)

    def test_joint_step_summary_names_the_largest_source_jump(self) -> None:
        self.trajectory["dof_pos"][1, 35] = 0.8
        summary = joint_step_summary(self.trajectory)
        self.assertEqual(summary["hands"]["joint"], TRAJECTORY_JOINT_NAMES[35])
        self.assertAlmostEqual(summary["hands"]["radians"], 0.8)


class IsolationTests(unittest.TestCase):
    """Replay must stay clear of SONIC, controllers and DDS."""

    FORBIDDEN = ("controllers", "sonic", "rclpy", "cyclonedds", "dds")

    def test_cpu_render_loop_steps_physics_to_propagate_link_transforms(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/replay.py").read_text()
        self.assertIn("self._sim.step(render=False)", source)
        self.assertNotIn("self._sim.forward()", source)

    def test_module_source_imports_no_controller_dds_or_sonic(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/replay.py").read_text()
        offenders = [
            line.strip()
            for line in source.splitlines()
            if re.match(r"\s*(from|import)\s", line)
            and any(token in line for token in self.FORBIDDEN)
        ]
        self.assertEqual(offenders, [])

    def test_importing_replay_pulls_in_no_simulator_or_dds(self) -> None:
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "import humanoid_lab.simulators.isaac.replay;"
            "print([m for m in sys.modules if m.startswith('humanoid_lab.controllers')"
            " or m == 'humanoid_lab.simulators.isaac.service' or m in ('rclpy', 'cyclonedds')])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]", result.stderr)


def _write_motion_lib(
    root: Path, key: str, frames: int = 6, hands: bool = True, scale: tuple[float, ...] = (1.5, 1.5, 1.5)
) -> None:
    """Write a minimal motion-lib directory whose pkls use a different inner key."""
    for name in ("robot", "objects", "meta", "object_usd"):
        (root / name).mkdir(parents=True, exist_ok=True)
    root_rot_xyzw = np.tile(np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=np.float32), (frames, 1))
    entry = {
        "root_trans_offset": np.tile(np.array([0.1, -0.2, 0.8], dtype=np.float32), (frames, 1)),
        "root_rot": root_rot_xyzw,
        "dof": np.tile(np.arange(29, dtype=np.float32), (frames, 1)),
        "fps": 25.0,
        # Open for the first half, closed from the middle frame on.
        "hand_action_right": np.concatenate(
            [np.full(frames // 2, -1.0, dtype=np.float32), np.full(frames - frames // 2, 1.0, dtype=np.float32)]
        ),
    }
    if hands:
        entry["hand_dof_pos"] = np.tile(np.arange(100, 114, dtype=np.float32), (frames, 1))
    joblib.dump({"source_motion_name_0001": entry}, root / "robot" / f"{key}.pkl")
    joblib.dump(
        {
            "source_motion_name_0001": {
                "root_pos": np.tile(np.array([0.0, 0.0, 0.9], dtype=np.float32), (frames, 1, 1)),
                "root_quat": root_rot_xyzw.reshape(frames, 1, 4),
                "scale": np.array(scale, dtype=np.float32),
            }
        },
        root / "objects" / f"{key}.pkl",
    )
    joblib.dump(
        {
            "table_pos": np.array([0.4, 0.5, 0.3], dtype=np.float32),
            "table_quat": np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=np.float32),
            "table_size": np.array([3.0, 1.0, 0.08], dtype=np.float32),
        },
        root / "meta" / f"{key}.pkl",
    )
    (root / "object_usd" / f"{key}.usd").write_text("#usda 1.0\n")


@requires_deps
class SyntheticDatasetTests(unittest.TestCase):
    """Conversion, DOF ordering and metadata transfer on a controlled library."""

    def _load(self, hands: bool = True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pickup_table"
            _write_motion_lib(root, "pickup_table__unit_0__000", hands=hands)
            return load_sequence("pickup_table__unit_0__000", root)

    def test_single_inner_key_fallback_and_43_dof_order(self) -> None:
        sequence = self._load()
        self.assertEqual((sequence.frames, sequence.fps), (6, 25.0))
        dof = np.asarray(sequence.trajectory["dof_pos"])
        self.assertEqual(dof.shape, (6, 43))
        # Body columns follow the MuJoCo -> IsaacLab reorder, hands follow verbatim.
        np.testing.assert_array_equal(dof[0, :29], np.asarray(MUJOCO_TO_ISAACLAB_BODY, dtype=np.float32))
        np.testing.assert_array_equal(dof[0, 29:], np.arange(100, 114, dtype=np.float32))

    def test_grasp_comes_from_the_source_right_hand_command(self) -> None:
        sequence = self._load()
        # The synthetic library closes at source frame 3 of 6.
        self.assertEqual(sequence.grasp_source_frame, 3)
        self.assertAlmostEqual(sequence.grasp_time_seconds, 3 / 25.0)
        self.assertEqual(sequence.grasp_render_frame, 6)

    def test_missing_or_invalid_right_hand_signal_fails_loudly(self) -> None:
        cases = {
            "absent": lambda entry: entry.pop("hand_action_right"),
            "non-discrete": lambda entry: entry.update(hand_action_right=np.zeros(6, dtype=np.float32)),
            "never closes": lambda entry: entry.update(
                hand_action_right=np.full(6, -1.0, dtype=np.float32)
            ),
            "wrong length": lambda entry: entry.update(
                hand_action_right=np.array([-1.0, 1.0], dtype=np.float32)
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "pickup_table"
                key = "pickup_table__unit_0__000"
                _write_motion_lib(root, key)
                payload = joblib.load(root / "robot" / f"{key}.pkl")
                mutate(next(iter(payload.values())))
                joblib.dump(payload, root / "robot" / f"{key}.pkl")
                with self.assertRaises(ReplayError):
                    load_sequence(key, root)

    def test_quaternions_arrive_as_wxyz(self) -> None:
        sequence = self._load()
        # Source is a 90 deg yaw in xyzw; the renderer needs (w, x, y, z).
        np.testing.assert_allclose(np.asarray(sequence.trajectory["root_quat_w"])[0], [0.70710678, 0.0, 0.0, 0.70710678], atol=1e-5)
        np.testing.assert_allclose(np.asarray(sequence.trajectory["object_quat_w"])[0], [0.70710678, 0.0, 0.0, 0.70710678], atol=1e-5)
        self.assertEqual(QUAT_CONVENTION, "xyzw")

    def test_object_scale_and_table_metadata_are_transferred(self) -> None:
        sequence = self._load()
        np.testing.assert_allclose(sequence.trajectory["object_scale"], [1.5, 1.5, 1.5])
        np.testing.assert_allclose(sequence.table_size, [3.0, 1.0, 0.08])
        np.testing.assert_allclose(sequence.table_pos, [0.4, 0.5, 0.3])
        # 90 deg yaw about Z, converted from the release's xyzw.
        np.testing.assert_allclose(sequence.table_quat_wxyz, [0.70710678, 0.0, 0.0, 0.70710678], atol=1e-6)

    def test_body_only_library_fails_loudly(self) -> None:
        with self.assertRaises(ReplayError):
            self._load(hands=False)

    def test_ambiguous_43_column_source_body_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pickup_table"
            key = "pickup_table__unit_0__000"
            _write_motion_lib(root, key)
            payload = joblib.load(root / "robot" / f"{key}.pkl")
            entry = next(iter(payload.values()))
            entry["dof"] = np.zeros((6, 43), dtype=np.float32)
            joblib.dump(payload, root / "robot" / f"{key}.pkl")
            with self.assertRaises(ReplayError) as caught:
                load_sequence(key, root)
            self.assertIn("source body dof", str(caught.exception))

    def test_invalid_source_scale_fails_loudly(self) -> None:
        # The GRAIL converter would only warn and substitute identity scale.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pickup_table"
            _write_motion_lib(root, "pickup_table__unit_0__000", scale=(-1.0, 1.0, 1.0))
            with self.assertRaises(ReplayError) as caught:
                load_sequence("pickup_table__unit_0__000", root)
            self.assertIn("object scale", str(caught.exception))

    def test_missing_files_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pickup_table"
            _write_motion_lib(root, "pickup_table__unit_0__000")
            with self.assertRaises(ReplayError) as caught:
                load_sequence("pickup_table__absent_0__000", root)
            self.assertIn("missing dataset file", str(caught.exception))


@requires_deps
@requires_dataset
class ReleasedDatasetTests(unittest.TestCase):
    def test_apple_motion_has_43_dof_and_canonical_quaternions(self) -> None:
        sequence = load_sequence(APPLE_KEY, DATA_ROOT)
        self.assertEqual((sequence.frames, sequence.fps), (250, 25.0))
        self.assertEqual(np.asarray(sequence.trajectory["dof_pos"]).shape, (250, 43))
        self.assertTrue(np.all(np.isfinite(sequence.trajectory["dof_pos"])))
        self.assertEqual(sequence.object_usd, DATA_ROOT / "object_usd" / f"{APPLE_KEY}.usd")
        np.testing.assert_allclose(sequence.trajectory["object_scale"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(sequence.table_size, [2.0, 0.6, 0.04])
        # The motion starts with the object resting on the table and the robot
        # upright, so the scalar slot dominates in wxyz. Reading the release's
        # xyzw as wxyz would move it to the last slot instead.
        object_quat = np.asarray(sequence.trajectory["object_quat_w"])[0]
        root_quat = np.asarray(sequence.trajectory["root_quat_w"])[0]
        self.assertGreater(abs(object_quat[0]), 0.9)
        self.assertGreater(abs(root_quat[0]), 0.7)
        np.testing.assert_allclose(np.linalg.norm(object_quat), 1.0, atol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(root_quat), 1.0, atol=1e-4)
        self.assertEqual(EXTERNAL_RESOLUTION, (1920, 1080))
        self.assertEqual(sequence.render_frames, 500)
        self.assertEqual(sequence.render_fps, 50.0)

    def test_apple_grasp_is_the_first_right_hand_close_of_the_source_pkl(self) -> None:
        sequence = load_sequence(APPLE_KEY, DATA_ROOT)
        self.assertEqual(sequence.grasp_source_frame, 96)
        self.assertAlmostEqual(sequence.grasp_time_seconds, 96 / 25.0)
        self.assertEqual(sequence.grasp_render_frame, 192)
        # Independently re-derive the transition from the raw source pkl.
        payload = joblib.load(sequence.robot_pkl)
        entry = next(iter(payload.values()))
        command = np.asarray(entry["hand_action_right"], dtype=np.float64)
        transitions = np.flatnonzero((command[:-1] < 0) & (command[1:] > 0))
        self.assertEqual(len(transitions), 1)
        self.assertEqual(int(transitions[0]) + 1, sequence.grasp_source_frame)

    def test_apple_hand_columns_fit_the_independently_declared_joint_limits(self) -> None:
        sequence = load_sequence(APPLE_KEY, DATA_ROOT)
        hand = np.asarray(sequence.trajectory["dof_pos"])[:, 29:]
        # Limits come from the pinned Dex3 URDF, not from replay.py. Their signs
        # distinguish left/right and thumb/index roles, catching common order swaps.
        limits = (
            (-1.57079632, 0.0),
            (-1.57079632, 0.0),
            (-1.04719755, 1.04719755),
            (0.0, 1.57079632),
            (0.0, 1.57079632),
            (-1.04719755, 1.04719755),
            (-1.74532925, 0.0),
            (-1.74532925, 0.0),
            (-0.72431163, 1.04719755),
            (0.0, 1.74532925),
            (0.0, 1.74532925),
            (-1.04719755, 0.72431163),
            (0.0, 1.74532925),
            (-1.74532925, 0.0),
        )
        for column, (lower, upper) in enumerate(limits):
            with self.subTest(joint=TRAJECTORY_JOINT_NAMES[29 + column]):
                self.assertGreaterEqual(float(hand[:, column].min()), lower - 1e-4)
                self.assertLessEqual(float(hand[:, column].max()), upper + 1e-4)


if __name__ == "__main__":
    unittest.main()
