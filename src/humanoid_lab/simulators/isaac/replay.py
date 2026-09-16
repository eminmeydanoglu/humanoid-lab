"""Kinematic GRAIL replay.

Opens one recorded ``pickup_table`` motion in Isaac: the Dex3 G1 from the
profile asset, the motion's object USD with its textures, the metadata table,
the profile head camera and one fixed external camera. The recorded robot and
object state is written directly to the stage and rendered; gravity is off and
no actuator, controller, DDS or SONIC code is involved.

The dynamic simulator in ``service.py`` is not involved either: replay only
shares the profile, the contracts and the process lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import RunProfile
from .visibility import (
    VISIBILITY_REPORT_NAME,
    FrameVisibility,
    VisibilityConfig,
    VisibilityError,
    build_report,
    window_bounds,
)

DEFAULT_DATA_ROOT = Path("/data/datasets/grail/data/pickup_table")
DEFAULT_OUTPUT_ROOT = Path("/outputs/grail-replay")
# The public release stores every quaternion as xyzw; "auto" is never used.
QUAT_CONVENTION = "xyzw"
EXTERNAL_RESOLUTION = (1920, 1080)
# Only used by motions whose meta pkl carries no table_size.
TABLE_FALLBACK_SIZE = (2.0, 0.6, 0.04)
# Preserve every recorded keyframe and insert one midpoint between adjacent
# samples. GRAIL pickup_table motions are 25 Hz, which is visibly coarse for
# the small Dex3 finger links.
RENDER_FRAME_MULTIPLIER = 2

# Column order of a pickup_table trajectory. The 29 body DOFs are reordered from
# MuJoCo to IsaacLab order by prepare_vis_shard; the 14 hand DOFs that
# export_successful_rollouts.py appended verbatim are in the *recording*
# articulation's order, which is not SONIC's declared G1_HAND_JOINTS list
# (grail/imports/SONIC/gear_sonic/envs/wrapper/manager_env_wrapper.py: "Hand DOFs
# are the last N joints (in Isaac order, not G1_HAND_JOINTS order)"). The profile
# asset shares that order, so the columns below are matched against the live
# articulation by name and normally land positionally.
TRAJECTORY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
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
)


class ReplayError(ValueError):
    """A replay input, dataset file or scene contract was violated."""


@dataclass(frozen=True)
class ReplaySequence:
    """One validated motion: its files, its trajectory, its table and its grasp."""

    key: str
    data_root: Path
    robot_pkl: Path
    object_pkl: Path
    meta_pkl: Path
    object_usd: Path
    trajectory: Mapping[str, Any]
    table_pos: tuple[float, float, float]
    table_quat_wxyz: tuple[float, float, float, float]
    table_size: tuple[float, float, float]
    grasp_source_frame: int
    grasp_time_seconds: float

    @property
    def frames(self) -> int:
        return int(self.trajectory["total_frames"])

    @property
    def fps(self) -> float:
        return float(self.trajectory["fps"])

    @property
    def render_frames(self) -> int:
        return self.frames * RENDER_FRAME_MULTIPLIER

    @property
    def render_fps(self) -> float:
        return self.fps * RENDER_FRAME_MULTIPLIER

    @property
    def grasp_render_frame(self) -> int:
        """Render frame that reproduces the grasp keyframe exactly."""
        return self.grasp_source_frame * RENDER_FRAME_MULTIPLIER


def validate_sequence_key(key: str) -> None:
    """Reject keys that are not one plain file name stem."""
    if not key or key in {".", ".."} or Path(key).name != key:
        raise ReplayError(f"invalid sequence key {key!r}: expected one file name stem")


def detect_grasp_source_frame(hand_action_right: Sequence[float], *, key: str = "") -> int:
    """Return the source frame at which the right hand first closes.

    The release stores one discrete command per source frame: -1 is open and +1
    is closed, so the grasp frame is the first closed frame that follows an open
    one. Every ``pickup_table`` motion resolves to exactly one such transition.
    Non-discrete, non-finite, already-closed and transition-free series are
    rejected instead of being interpreted.
    """
    label = f"{key}: " if key else ""
    try:
        values = [float(value) for value in hand_action_right]
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"{label}hand_action_right is not a one-dimensional numeric series") from exc
    if len(values) < 2:
        raise ReplayError(f"{label}hand_action_right has {len(values)} frames, a transition needs at least 2")
    for frame, value in enumerate(values):
        if not math.isfinite(value):
            raise ReplayError(f"{label}hand_action_right is not finite at source frame {frame}")
        if value not in (-1.0, 1.0):
            raise ReplayError(
                f"{label}hand_action_right must be discrete -1 (open) or +1 (closed), "
                f"got {value!r} at source frame {frame}"
            )
    transitions = [
        frame
        for frame in range(1, len(values))
        if values[frame - 1] < 0.0 < values[frame]
    ]
    if len(transitions) == 1:
        return transitions[0]
    if len(transitions) > 1:
        raise ReplayError(
            f"{label}hand_action_right closes {len(transitions)} times at source frames "
            f"{transitions}; the grasp event is ambiguous"
        )
    if values[0] > 0.0:
        raise ReplayError(
            f"{label}the right hand is already closed at the first source frame; "
            "the grasp transition is not inside the recording"
        )
    raise ReplayError(f"{label}hand_action_right never closes: no open-to-closed transition")


def joint_layout(articulation_names: Sequence[str]) -> list[int]:
    """Return the articulation index of each trajectory column, matched by name.

    Guards the assumption that the trajectory's column order is the one the
    articulation actually uses, instead of trusting a positional write.
    """
    expected = len(TRAJECTORY_JOINT_NAMES)
    if len(articulation_names) != expected:
        raise ReplayError(
            f"articulation has {len(articulation_names)} joints, "
            f"trajectory has {expected} — wrong robot asset?"
        )
    if len(set(articulation_names)) != len(articulation_names):
        raise ReplayError("articulation contains duplicate joint names")
    index = {name: position for position, name in enumerate(articulation_names)}
    missing = [name for name in TRAJECTORY_JOINT_NAMES if name not in index]
    if missing:
        raise ReplayError(f"articulation is missing trajectory joints: {', '.join(missing)}")
    return [index[name] for name in TRAJECTORY_JOINT_NAMES]


def joint_positions(dof: Any, layout: Sequence[int]) -> Any:
    """Arrange one trajectory row into articulation order.

    ``layout[column]`` is that column's articulation index, so this places each
    column at its own joint (a scatter). Reading it as a gather would drive the
    wrong joints whenever the two orders differ.
    """
    import numpy as np

    positions = np.zeros(len(layout), dtype=np.float32)
    positions[list(layout)] = dof
    return positions


def _slerp_wxyz(first: Any, second: Any, alpha: float) -> Any:
    """Interpolate unit quaternions on the shortest arc."""
    import numpy as np

    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        value = q0 + alpha * (q1 - q0)
        value /= np.linalg.norm(value)
        return value.astype(np.float32)
    angle = np.arccos(dot)
    value = (np.sin((1.0 - alpha) * angle) * q0 + np.sin(alpha * angle) * q1) / np.sin(angle)
    return value.astype(np.float32)


def interpolated_frame(trajectory: Mapping[str, Any], render_frame: int) -> dict[str, Any]:
    """Return one render sample while preserving every recorded keyframe."""
    import numpy as np

    source_frame, subframe = divmod(render_frame, RENDER_FRAME_MULTIPLIER)
    last = int(trajectory["total_frames"]) - 1
    lower = min(source_frame, last)
    upper = min(lower + 1, last)
    alpha = subframe / RENDER_FRAME_MULTIPLIER

    def linear(name: str) -> Any:
        values = np.asarray(trajectory[name])
        return ((1.0 - alpha) * values[lower] + alpha * values[upper]).astype(np.float32)

    return {
        "dof_pos": linear("dof_pos"),
        "root_pos_w": linear("root_pos_w"),
        "root_quat_w": _slerp_wxyz(
            trajectory["root_quat_w"][lower], trajectory["root_quat_w"][upper], alpha
        ),
        "object_pos_w": linear("object_pos_w"),
        "object_quat_w": _slerp_wxyz(
            trajectory["object_quat_w"][lower], trajectory["object_quat_w"][upper], alpha
        ),
    }


def joint_step_summary(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the largest recorded one-frame joint changes by joint name."""
    import numpy as np

    values = np.asarray(trajectory["dof_pos"], dtype=np.float64)
    steps = np.abs(np.diff(values, axis=0))
    maxima = steps.max(axis=0) if len(steps) else np.zeros(values.shape[1])

    def group(start: int, stop: int) -> dict[str, Any]:
        local = maxima[start:stop]
        offset = int(np.argmax(local))
        index = start + offset
        return {"radians": float(local[offset]), "joint": TRAJECTORY_JOINT_NAMES[index]}

    return {
        "all": group(0, len(TRAJECTORY_JOINT_NAMES)),
        "body": group(0, 29),
        "hands": group(29, len(TRAJECTORY_JOINT_NAMES)),
    }


def _converter() -> Any:
    """GRAIL's motion-lib converter from the pinned third_party checkout."""
    checkout = Path(__file__).resolve().parents[4] / "third_party" / "GRAIL"
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    try:
        from grail.visualization import prepare_vis_shard
    except ImportError as exc:  # pragma: no cover - dataset checkout is pinned
        raise ReplayError(f"cannot import the GRAIL converter from {checkout}: {exc}") from exc
    return prepare_vis_shard


def _validate_trajectory(key: str, trajectory: Mapping[str, Any]) -> None:
    import numpy as np

    frames = int(trajectory.get("total_frames", 0))
    if frames <= 0:
        raise ReplayError(f"{key}: trajectory has no frames")
    if float(trajectory.get("fps", 0.0)) <= 0.0:
        raise ReplayError(f"{key}: trajectory has no positive fps")
    expected = {
        "root_pos_w": (frames, 3),
        "root_quat_w": (frames, 4),
        "dof_pos": (frames, len(TRAJECTORY_JOINT_NAMES)),
        "object_pos_w": (frames, 3),
        "object_quat_w": (frames, 4),
    }
    for name, shape in expected.items():
        if name not in trajectory:
            raise ReplayError(f"{key}: trajectory has no {name}")
        values = np.asarray(trajectory[name], dtype=np.float64)
        if values.shape != shape:
            raise ReplayError(f"{key}: {name} has shape {values.shape}, expected {shape}")
        if not np.all(np.isfinite(values)):
            raise ReplayError(f"{key}: {name} contains non-finite values")
        if name in {"root_quat_w", "object_quat_w"}:
            norms = np.linalg.norm(values, axis=1)
            if np.any(norms <= 1e-8):
                raise ReplayError(f"{key}: {name} contains a zero quaternion")
    scale = np.asarray(trajectory["object_scale"], dtype=np.float64).reshape(-1)
    if scale.size != 3 or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ReplayError(f"{key}: object scale must be three positive finite values")


def _xform_ops(xformable: Any) -> tuple[Any, Any, Any]:
    """Return a prim's translate/orient/scale ops, reusing the asset's own.

    A referenced object USD already authors these three ops on its default prim,
    so ``AddXformOp`` refuses to create a second copy of the same name.
    """
    from pxr import UsdGeom

    ops = {op.GetOpType(): op for op in xformable.GetOrderedXformOps()}
    for op_type, add in (
        (UsdGeom.XformOp.TypeTranslate, xformable.AddTranslateOp),
        (UsdGeom.XformOp.TypeOrient, xformable.AddOrientOp),
        (UsdGeom.XformOp.TypeScale, xformable.AddScaleOp),
    ):
        if op_type not in ops:
            ops[op_type] = add()
    return (
        ops[UsdGeom.XformOp.TypeTranslate],
        ops[UsdGeom.XformOp.TypeOrient],
        ops[UsdGeom.XformOp.TypeScale],
    )


def _set_xform_value(op: Any, values: Sequence[float]) -> None:
    """Set one xform op with the value type its precision expects."""
    from pxr import Gf, UsdGeom

    single = op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
    numbers = [float(value) for value in values]
    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        op.Set(Gf.Quatf(*numbers) if single else Gf.Quatd(*numbers))
    else:
        op.Set(Gf.Vec3f(*numbers) if single else Gf.Vec3d(*numbers))


def _source_robot_entry(key: str, robot_pkl: Path) -> Mapping[str, Any]:
    """Load the robot pkl and resolve its motion entry.

    The public release wraps one motion per file; a file that also carries the
    sequence key as its own inner key is accepted as well. Anything else is a
    structure replay does not understand.
    """
    import joblib

    payload = joblib.load(robot_pkl)
    if not isinstance(payload, Mapping) or not payload:
        raise ReplayError(f"{key}: unexpected robot pkl structure")
    if key in payload:
        entry = payload[key]
    elif len(payload) == 1:
        entry = next(iter(payload.values()))
    else:
        raise ReplayError(f"{key}: unexpected robot pkl structure")
    if not isinstance(entry, Mapping):
        raise ReplayError(f"{key}: unexpected robot pkl structure")
    return entry


def _validate_source_robot(key: str, robot_pkl: Path) -> Mapping[str, Any]:
    """Require the public-release body-plus-separate-hands source layout."""
    import numpy as np

    entry = _source_robot_entry(key, robot_pkl)
    root = np.asarray(entry.get("root_trans_offset"))
    if root.ndim != 2 or root.shape[1] != 3 or len(root) == 0:
        raise ReplayError(f"{key}: source root_trans_offset has invalid shape {root.shape}")
    body = np.asarray(entry.get("dof"))
    hands = np.asarray(entry.get("hand_dof_pos"))
    frames = len(root)
    if body.shape != (frames, 29):
        raise ReplayError(f"{key}: source body dof has shape {body.shape}, expected {(frames, 29)}")
    if hands.shape != (frames, 14):
        raise ReplayError(f"{key}: source hand_dof_pos has shape {hands.shape}, expected {(frames, 14)}")
    if "hand_action_right" not in entry:
        raise ReplayError(f"{key}: source robot pkl has no hand_action_right series")
    hand_action = np.asarray(entry["hand_action_right"])
    if hand_action.shape != (frames,):
        raise ReplayError(
            f"{key}: source hand_action_right has shape {hand_action.shape}, expected {(frames,)}"
        )
    return entry


def _validate_source_scale(key: str, object_pkl: Path) -> None:
    """Reject a malformed scale in the objects pkl.

    The GRAIL converter substitutes identity for an invalid scale with only a
    stdout warning, which would turn a broken dataset file into a silently
    wrongly sized object.
    """
    import joblib
    import numpy as np

    payload = joblib.load(object_pkl)
    if not isinstance(payload, Mapping) or not payload:
        raise ReplayError(f"{key}: unexpected objects pkl structure")
    if key in payload:
        entry = payload[key]
    elif len(payload) == 1:  # the release wraps one motion per file
        entry = next(iter(payload.values()))
    else:
        raise ReplayError(f"{key}: unexpected objects pkl structure")
    scale = entry.get("scale") if isinstance(entry, Mapping) else None
    if scale is None:
        return
    values = np.asarray(scale, dtype=np.float64).reshape(-1)
    if values.size not in (1, 3) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ReplayError(f"{key}: object scale must be positive and finite, got {scale!r}")


def _table_metadata(
    key: str, meta_pkl: Path
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]:
    import joblib
    import numpy as np

    meta = joblib.load(meta_pkl)
    if not isinstance(meta, Mapping) or "table_pos" not in meta:
        raise ReplayError(f"{key}: meta has no table_pos")
    pos = np.asarray(meta["table_pos"], dtype=np.float64).reshape(-1)
    if pos.size != 3 or not np.all(np.isfinite(pos)):
        raise ReplayError(f"{key}: table_pos is not three finite values")
    quat = np.asarray(meta.get("table_quat", (0.0, 0.0, 0.0, 1.0)), dtype=np.float64).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        raise ReplayError(f"{key}: table_quat is not four finite values")
    quat = quat[[3, 0, 1, 2]]  # release xyzw -> USD wxyz
    length = float(np.linalg.norm(quat))
    if length <= 0.0:
        raise ReplayError(f"{key}: table_quat is not a rotation")
    size = np.asarray(meta.get("table_size", TABLE_FALLBACK_SIZE), dtype=np.float64).reshape(-1)
    if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ReplayError(f"{key}: table_size is not three positive finite values")
    return (
        tuple(float(value) for value in pos),
        tuple(float(value / length) for value in quat),
        tuple(float(value) for value in size),
    )


def load_sequence(key: str, data_root: Path = DEFAULT_DATA_ROOT) -> ReplaySequence:
    """Resolve one sequence key and convert its motion to a replay trajectory."""
    import numpy as np

    validate_sequence_key(key)
    root = Path(data_root)
    robot_pkl = root / "robot" / f"{key}.pkl"
    object_pkl = root / "objects" / f"{key}.pkl"
    meta_pkl = root / "meta" / f"{key}.pkl"
    object_usd = root / "object_usd" / f"{key}.usd"
    for path in (robot_pkl, object_pkl, meta_pkl, object_usd):
        if not path.is_file():
            raise ReplayError(f"{key}: missing dataset file {path}")
    robot_entry = _validate_source_robot(key, robot_pkl)
    with tempfile.TemporaryDirectory(prefix="grail-replay-") as shard:
        converter = _converter()
        try:
            produced = converter.convert_motion_lib_to_trajectories(
                str(root), shard, motion_filter={key}, quat_convention=QUAT_CONVENTION
            )
        except SystemExit as exc:  # the helper exits the process on bad libraries
            raise ReplayError(f"{key}: GRAIL conversion failed ({exc})") from exc
        trajectory_path = Path(shard) / "trajectories" / f"{key}.trajectory.pkl"
        if key not in produced or not trajectory_path.is_file():
            raise ReplayError(f"{key}: GRAIL conversion produced no trajectory")
        with trajectory_path.open("rb") as handle:
            trajectory: dict[str, Any] = pickle.load(handle)
    # The helper canonicalizes the robot quaternion but passes the object one
    # through; the release stores both as xyzw and the renderer needs wxyz.
    _validate_trajectory(key, trajectory)
    trajectory["object_quat_w"] = np.asarray(trajectory["object_quat_w"], dtype=np.float32)[:, [3, 0, 1, 2]]
    for name in ("root_quat_w", "object_quat_w"):
        quaternions = np.asarray(trajectory[name], dtype=np.float32)
        trajectory[name] = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    _validate_trajectory(key, trajectory)
    _validate_source_scale(key, object_pkl)
    grasp_source_frame = detect_grasp_source_frame(robot_entry["hand_action_right"], key=key)
    if grasp_source_frame >= int(trajectory["total_frames"]):
        raise ReplayError(f"{key}: grasp source frame {grasp_source_frame} is outside the trajectory")
    grasp_time_seconds = grasp_source_frame / float(trajectory["fps"])
    pos, quat, size = _table_metadata(key, meta_pkl)
    return ReplaySequence(
        key=key,
        data_root=root,
        robot_pkl=robot_pkl,
        object_pkl=object_pkl,
        meta_pkl=meta_pkl,
        object_usd=object_usd,
        trajectory=trajectory,
        table_pos=pos,
        table_quat_wxyz=quat,
        table_size=size,
        grasp_source_frame=grasp_source_frame,
        grasp_time_seconds=grasp_time_seconds,
    )


class ReplayService:
    """Write one recorded motion to the stage and render it."""

    def __init__(
        self,
        profile: RunProfile,
        simulation_app: Any,
        sequence: ReplaySequence,
        *,
        output_dir: Path,
        show_ui: bool = False,
        visibility: VisibilityConfig | None = None,
    ) -> None:
        self.profile = profile
        self.app = simulation_app
        self.sequence = sequence
        self.output_dir = Path(output_dir)
        self.show_ui = show_ui
        self.visibility_config = visibility
        self.head_camera_distinct_frames = 0
        self._articulation: dict[str, Any] = {}
        self._visibility: Any = None
        self._visibility_rows: list[FrameVisibility] = []
        # Largest gap between what replay writes and what Isaac actually holds.
        self._joint_tracking_error = 0.0
        self._root_tracking_error = 0.0
        # Robot and object share the table's origin shift; only relative
        # geometry reaches the cameras and the USD stage.
        self._shift = (float(sequence.table_pos[0]), float(sequence.table_pos[1]))
        self._sim: Any = None
        self._robot: Any = None
        self._head_camera: Any = None
        self._head_camera_path = ""
        # What Isaac authored on the prim, or None while no scene exists yet.
        self._head_camera_usd: dict[str, float] | None = None
        self._external_camera: Any = None
        self._object_ops: tuple[Any, Any, Any] | None = None
        self._external_eye = (0.0, 0.0, 0.0)
        self._external_target = (0.0, 0.0, 0.0)
        self._joint_map: list[int] = []
        self._viewport_windows: list[Any] = []
        if visibility is not None:
            # A window that cannot contain one render frame must fail before any
            # rendering, not produce an empty summary afterwards.
            try:
                window_bounds(sequence.grasp_time_seconds, visibility, sequence.render_fps, sequence.render_frames)
            except VisibilityError as exc:
                raise ReplayError(f"{sequence.key}: invalid visibility window: {exc}") from exc

    def run(self) -> dict[str, Any]:
        summary = self._manifest()
        # A run that dies mid-clip must not leave the previous run's result on
        # disk next to its own truncated videos. That covers visibility.json.
        self._clear_visibility_report()
        if not self.show_ui:
            self._write_manifest(summary)
        try:
            self._build_scene()
            summary["articulation"] = self._articulation
            # The camera block carries what the prim holds, so it is refreshed
            # after the scene exists; the first write only claimed the profile.
            summary["cameras"] = self._camera_summary()
            if self.show_ui:
                self._open_ui()
            self._render_clip()
            summary["head_camera_distinct_frames"] = self.head_camera_distinct_frames
            if self.visibility_config is not None:
                summary["visibility"] = self._finish_visibility()
            summary["result"] = "COMPLETED"
        except Exception as exc:
            summary["result"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            summary["tracking_error"] = {
                "joint_radians": self._joint_tracking_error,
                "root_metres": self._root_tracking_error,
            }
            if not self.show_ui:
                self._write_manifest(summary)
        return summary

    def _clear_visibility_report(self) -> None:
        if self.visibility_config is not None:
            (self.output_dir / VISIBILITY_REPORT_NAME).unlink(missing_ok=True)

    def _head_camera_report(self) -> dict[str, Any]:
        camera = self.profile.camera
        return {
            "name": camera.name,
            "resolution": [camera.width, camera.height],
            "horizontal_fov_deg": camera.horizontal_fov_deg,
            "rendered_fov_deg": list(camera.rendered_fov_deg),
            "nominal_fov_deg": list(camera.nominal_fov_deg) if camera.nominal_fov_deg else None,
            "focal_length_mm": camera.focal_length_mm,
            "focal_length_px": list(camera.focal_length_px),
            "position_m": list(camera.position_m),
            "rotation_wxyz": list(camera.rotation_wxyz),
            "provenance": camera.provenance,
        }

    def _finish_visibility(self) -> dict[str, Any]:
        """Write the visibility report and return its manifest summary."""
        visibility = self._visibility
        config = self.visibility_config
        if visibility is None or config is None:  # pragma: no cover - guarded by run()
            raise ReplayError("visibility analysis was requested but the scene never prepared it")
        if not visibility.saw_target:
            raise ReplayError(
                "visibility analysis never saw the target object: no labelled pixel or bounding box "
                "appeared in any frame, so all-zero values would not mean 'not visible'"
            )
        report = build_report(
            sequence_key=self.sequence.key,
            grasp_source_frame=self.sequence.grasp_source_frame,
            grasp_time_seconds=self.sequence.grasp_time_seconds,
            source_fps=self.sequence.fps,
            source_frames=self.sequence.frames,
            render_fps=self.sequence.render_fps,
            render_frames=self.sequence.render_frames,
            config=config,
            head_camera=self._head_camera_report(),
            frames=self._visibility_rows,
            analysis_resolution=visibility.resolution,
            analysis_view_scale=visibility.scale,
            result="COMPLETED",
        )
        report_path = self.output_dir / VISIBILITY_REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        summary = report["summary"]
        return {
            "enabled": True,
            "report": str(report_path),
            "threshold": config.threshold,
            "before_seconds": config.before_seconds,
            "after_seconds": config.after_seconds,
            "measurement": visibility.summary(),
            "grasp_render_frame": report["grasp"]["render_frame"],
            "window_start_render_frame": report["window"]["start_render_frame"],
            "window_end_render_frame": report["window"]["end_render_frame"],
            "window_frames": summary["window_frames"],
            "window_min_fraction": summary["window_min_fraction"],
            "window_mean_fraction": summary["window_mean_fraction"],
            "grasp_fraction": summary["grasp_fraction"],
            "window_below_threshold_fraction": summary["window_below_threshold_fraction"],
            "object_below_threshold_for_whole_window": summary["object_below_threshold_for_whole_window"],
            "object_out_of_frame_for_whole_window": summary["object_out_of_frame_for_whole_window"],
            "invalid_frames": summary["invalid_frames"],
        }

    def _write_manifest(self, summary: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    def _camera_summary(self) -> dict[str, Any]:
        """Declared camera geometry, the angles it renders, and the prim's values."""
        camera = self.profile.camera
        return {
            "head": {
                "name": camera.name,
                "resolution": [camera.width, camera.height],
                "rendered_fov_deg": list(camera.rendered_fov_deg),
                "nominal_fov_deg": (
                    list(camera.nominal_fov_deg) if camera.nominal_fov_deg else None
                ),
                "focal_length_mm": camera.focal_length_mm,
                "horizontal_aperture_mm": camera.horizontal_aperture_mm,
                "vertical_aperture_mm": camera.vertical_aperture_mm,
                "focal_length_px": list(camera.focal_length_px),
                "provenance": camera.provenance,
                # What Isaac authored on the prim; null until the scene exists,
                # so "not read yet" cannot be mistaken for "read and empty".
                "usd_attributes": (
                    None if self._head_camera_usd is None else dict(self._head_camera_usd)
                ),
            },
            "external": {"name": "external", "resolution": list(EXTERNAL_RESOLUTION)},
        }

    def _manifest(self) -> dict[str, Any]:
        sequence = self.sequence
        return {
            "schema_version": 1,
            "sequence_key": sequence.key,
            "data_root": str(sequence.data_root),
            "sources": {
                "robot": str(sequence.robot_pkl),
                "objects": str(sequence.object_pkl),
                "meta": str(sequence.meta_pkl),
                "object_usd": str(sequence.object_usd),
            },
            "frames": sequence.frames,
            "fps": sequence.fps,
            "render_frames": sequence.render_frames,
            "render_fps": sequence.render_fps,
            "trajectory": {
                "dof_pos_shape": list(sequence.trajectory["dof_pos"].shape),
                "quaternion_convention": QUAT_CONVENTION,
                "joint_names": list(TRAJECTORY_JOINT_NAMES),
                "max_source_joint_step": joint_step_summary(sequence.trajectory),
            },
            "robot_asset": self.profile.robot.asset_reference,
            "object": {
                "usd": str(sequence.object_usd),
                "scale": [float(value) for value in sequence.trajectory["object_scale"]],
            },
            "table": {
                "position": list(sequence.table_pos),
                "quaternion_wxyz": list(sequence.table_quat_wxyz),
                "size": list(sequence.table_size),
            },
            "cameras": self._camera_summary(),
            "grasp": {
                "source_frame": sequence.grasp_source_frame,
                "source_time_seconds": sequence.grasp_time_seconds,
                "render_frame": sequence.grasp_render_frame,
            },
            "visibility": self._visibility_manifest(),
            "outputs": (
                [] if self.show_ui else [str(self.output_dir / f"{name}.mp4") for name in ("external", "head")]
            ),
            "result": "PENDING",
        }

    def _visibility_manifest(self) -> dict[str, Any] | None:
        """The manifest's visibility block: claimed before the run, filled after it."""
        config = self.visibility_config
        if config is None:
            return None
        return {
            "enabled": True,
            "report": str(self.output_dir / VISIBILITY_REPORT_NAME),
            "threshold": config.threshold,
            "before_seconds": config.before_seconds,
            "after_seconds": config.after_seconds,
        }

    # ------------------------------------------------------------------- scene

    def _build_scene(self) -> None:
        import carb
        import isaaclab.sim as sim_utils
        import omni.usd
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab.sim import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager

        from .visibility_sensor import LABEL_FILTER, TargetVisibility

        profile = self.profile
        device = profile.device
        # Mirror the dynamic service's CPU path: no Fabric, and physics state is
        # written back to USD, so the renderer reads exactly what replay writes.
        settings = carb.settings.get_settings()
        settings.set_int("/persistent/physics/numThreads", 4)
        settings.set_bool("/physics/fabricEnabled", False)
        settings.set_bool("/physics/updateToUsd", True)
        SimulationManager.enable_fabric(False)
        self._sim = SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / self.sequence.render_fps,
                render_interval=1,
                device=device,
                use_fabric=False,
                gravity=(0.0, 0.0, 0.0),
            )
        )
        ground = sim_utils.GroundPlaneCfg(size=(500.0, 500.0))
        ground.func("/World/ground", ground)
        dome = sim_utils.DomeLightCfg(color=(0.45, 0.55, 0.75), intensity=1500.0)
        dome.func("/World/DomeLight", dome)
        key_light = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.95, 0.88))
        key_light.func("/World/KeyLight", key_light)

        env_path = "/World/envs/env_0"
        stage = omni.usd.get_context().get_stage()
        stage.DefinePrim(env_path, "Xform")
        robot_cfg = ArticulationCfg(
            prim_path=f"{env_path}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=profile.robot.asset_reference,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            actuators={
                "body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)
            },
        )
        self._robot = Articulation(robot_cfg)
        camera = profile.camera
        visibility = self.visibility_config is not None
        head_data_types = ["rgb"] + (["semantic_segmentation"] if visibility else [])
        self._head_camera_path = f"{env_path}/Robot/{camera.parent_link}/{camera.name}"
        self._head_camera = Camera(
            CameraCfg(
                prim_path=self._head_camera_path,
                update_period=0.0,
                width=camera.width,
                height=camera.height,
                data_types=head_data_types,
                colorize_semantic_segmentation=False,
                semantic_filter=LABEL_FILTER if visibility else "*:*",
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=camera.focal_length_mm,
                    horizontal_aperture=camera.horizontal_aperture_mm,
                    vertical_aperture=camera.vertical_aperture_mm,
                    clipping_range=(0.05, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=camera.position_m, rot=camera.rotation_wxyz, convention="world"
                ),
            )
        )
        self._external_eye, self._external_target = self._external_view()
        self._external_camera = Camera(
            CameraCfg(
                prim_path="/World/ExternalCamera",
                update_period=0.0,
                width=EXTERNAL_RESOLUTION[0],
                height=EXTERNAL_RESOLUTION[1],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0, horizontal_aperture=10.0, clipping_range=(0.1, 500.0)
                ),
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            )
        )
        self._spawn_table(stage, env_path)
        object_prim = self._spawn_object(stage, env_path)
        if visibility:
            # The analysis camera shares the head camera's pose, optics and
            # pixel scale with a wider view, so it can measure projections the
            # 640x480 image clips away.
            try:
                self._visibility = TargetVisibility(profile, env_path)
            except ValueError as exc:
                raise ReplayError(f"{self.sequence.key}: visibility setup failed: {exc}") from exc

        self._sim.reset()
        sensors = [self._head_camera, self._external_camera]
        if self._visibility is not None:
            sensors.append(self._visibility.camera)
        for sensor in sensors:
            sensor.reset()
        self._robot.reset()
        self._head_camera_usd = self._read_head_camera_usd()
        if self._visibility is not None:
            try:
                self._visibility.attach(object_prim, self._head_camera)
            except ValueError as exc:
                raise ReplayError(f"{self.sequence.key}: visibility setup failed: {exc}") from exc
        sensor_device = getattr(self._external_camera, "_device", "cpu")
        self._external_camera.set_world_poses_from_view(
            torch.tensor([list(self._external_eye)], dtype=torch.float32, device=sensor_device),
            torch.tensor([list(self._external_target)], dtype=torch.float32, device=sensor_device),
        )
        names = list(self._robot.joint_names)
        self._joint_map = joint_layout(names)
        positional = self._joint_map == list(range(len(names)))
        self._articulation = {
            "joints": len(names),
            "joint_names": names,
            "positional_order": positional,
        }
        print(
            f"[replay] articulation joints={len(names)} positional_order={positional} device={device}",
            flush=True,
        )
        if not positional:
            # Trajectory columns are matched by name; show what the asset declares.
            print(f"[replay] articulation joint order: {names}", flush=True)
        for _ in range(5):  # warm up the shaders and the sensor render products
            self._sim.step(render=True)
            self._read_cameras()

    def _read_head_camera_usd(self) -> dict[str, float]:
        """Read the projection attributes Isaac authored on the camera prim.

        The manifest then carries the angles, the derived millimetres and the
        prim's own focal length and apertures, so a parameter that never reached
        the camera shows up as a difference instead of as a different view.
        """
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = UsdGeom.Camera(stage.GetPrimAtPath(self._head_camera_path))
        return {
            "focal_length_mm": float(prim.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(prim.GetHorizontalApertureAttr().Get()),
            "vertical_aperture_mm": float(prim.GetVerticalApertureAttr().Get()),
        }

    def _external_view(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Frame the table and the robot from the side the robot is facing."""
        import numpy as np

        root = np.asarray(self.sequence.trajectory["root_pos_w"][0], dtype=np.float64)
        centre = (0.5 * (float(root[0]) - self._shift[0]), 0.5 * (float(root[1]) - self._shift[1]))
        side = 1.0 if centre[1] >= 0.0 else -1.0
        return (centre[0], centre[1] - side * 3.4, 1.5), (centre[0], centre[1], 0.9)

    def _spawn_table(self, stage: Any, env_path: str) -> None:
        from pxr import Gf, UsdGeom

        position, quaternion, size = (
            self.sequence.table_pos,
            self.sequence.table_quat_wxyz,
            self.sequence.table_size,
        )
        width, depth, thickness = size
        table = UsdGeom.Xform.Define(stage, f"{env_path}/Table")
        table.AddTranslateOp().Set(
            Gf.Vec3d(position[0] - self._shift[0], position[1] - self._shift[1], position[2])
        )
        table.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*quaternion))
        slab = UsdGeom.Cube.Define(stage, f"{env_path}/Table/slab")
        slab.CreateSizeAttr(1.0)
        slab.GetDisplayColorAttr().Set([Gf.Vec3f(0.62, 0.47, 0.33)])
        UsdGeom.Xformable(slab.GetPrim()).AddScaleOp().Set(Gf.Vec3f(width, depth, thickness))
        leg_height = position[2] - thickness / 2.0
        if leg_height <= 0.0:
            return
        leg_width, inset = 0.04, 0.03
        offset_x = width / 2.0 - inset - leg_width / 2.0
        offset_y = depth / 2.0 - inset - leg_width / 2.0
        for index, (sign_x, sign_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            leg = UsdGeom.Cube.Define(stage, f"{env_path}/Table/leg_{index}")
            leg.CreateSizeAttr(1.0)
            leg.GetDisplayColorAttr().Set([Gf.Vec3f(0.40, 0.28, 0.15)])
            xformable = UsdGeom.Xformable(leg.GetPrim())
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(sign_x * offset_x, sign_y * offset_y, -(thickness + leg_height) / 2.0)
            )
            xformable.AddScaleOp().Set(Gf.Vec3f(leg_width, leg_width, leg_height))

    def _spawn_object(self, stage: Any, env_path: str) -> Any:
        from pxr import UsdGeom

        # A raw reference keeps the USD's relative texture paths intact and lets
        # the per-frame xform ops below win over the asset's own ops.
        prim = stage.DefinePrim(f"{env_path}/Object", "Xform")
        prim.GetReferences().AddReference(str(self.sequence.object_usd))
        self._object_ops = _xform_ops(UsdGeom.Xformable(prim))
        _set_xform_value(self._object_ops[2], self.sequence.trajectory["object_scale"])
        return prim

    # ------------------------------------------------------------------ render

    def _write_frame(self, frame: int) -> tuple[Any, Any]:
        import numpy as np
        import torch

        sample = interpolated_frame(self.sequence.trajectory, frame)
        dof = joint_positions(sample["dof_pos"], self._joint_map)
        joint_pos = torch.from_numpy(dof).unsqueeze(0).to(self._robot.device)
        joint_vel = torch.zeros_like(joint_pos)
        root_position = np.asarray(sample["root_pos_w"], dtype=np.float32).copy()
        root_position[0] -= self._shift[0]
        root_position[1] -= self._shift[1]
        root_state = torch.zeros(1, 13, device=self._robot.device)
        root_state[0, :3] = torch.from_numpy(root_position)
        root_state[0, 3:7] = torch.from_numpy(np.asarray(sample["root_quat_w"], dtype=np.float32))
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.write_root_state_to_sim(root_state)

        if self._object_ops is not None:
            object_position = np.asarray(sample["object_pos_w"], dtype=np.float32).copy()
            translate, orient, _ = self._object_ops
            _set_xform_value(
                translate,
                (object_position[0] - self._shift[0], object_position[1] - self._shift[1], object_position[2]),
            )
            _set_xform_value(orient, sample["object_quat_w"])

        # Fabric is disabled on this CPU articulation, so forward() does not
        # propagate child-link transforms to USD. A zero-gravity physics step
        # updates the rendered articulation after the recorded state is written.
        self._sim.step(render=False)
        self._sim.render()
        # Read PhysX directly rather than comparing against ArticulationData,
        # whose buffers are populated by write_*_to_sim and would be tautological.
        physx_joint_pos = self._robot.root_physx_view.get_dof_positions()[0]
        physx_root_pos = self._robot.root_physx_view.get_root_transforms()[0, :3]
        self._joint_tracking_error = max(
            self._joint_tracking_error,
            float((physx_joint_pos - joint_pos[0]).abs().max()),
        )
        self._root_tracking_error = max(
            self._root_tracking_error,
            float((physx_root_pos - root_state[0, :3]).abs().max()),
        )
        return self._read_cameras(frame)

    def _read_cameras(self, frame: int | None = None) -> tuple[Any, Any]:
        self._head_camera.update(dt=0.0)
        self._external_camera.update(dt=0.0)
        if self._visibility is not None and frame is not None:
            self._visibility_rows.append(
                self._visibility.read(
                    frame,
                    render_fps=self.sequence.render_fps,
                    grasp_time_seconds=self.sequence.grasp_time_seconds,
                )
            )
        return (
            self._head_camera.data.output["rgb"][0].cpu().numpy()[..., :3],
            self._external_camera.data.output["rgb"][0].cpu().numpy()[..., :3],
        )

    def _render_clip(self) -> None:
        import imageio

        writers: dict[str, Any] = {}
        if not self.show_ui:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            writers = {
                name: imageio.get_writer(
                    self.output_dir / f"{name}.mp4",
                    fps=self.sequence.render_fps,
                    codec="libx264",
                    quality=5,
                    pixelformat="yuv420p",
                    macro_block_size=1,
                )
                for name in ("external", "head")
            }
        period = 1.0 / self.sequence.render_fps
        digests: set[bytes] = set()
        started_at = time.monotonic()
        completed = False
        try:
            while True:
                for frame in range(self.sequence.render_frames):
                    if self.show_ui and not self.app.is_running():
                        return
                    started = time.monotonic()
                    head, external = self._write_frame(frame)
                    digests.add(hashlib.blake2b(head.tobytes(), digest_size=8).digest())
                    if writers:
                        writers["external"].append_data(external)
                        writers["head"].append_data(head)
                    else:
                        time.sleep(max(0.0, period - (time.monotonic() - started)))
                    if frame % 25 == 0:
                        print(
                            f"[replay] frame {frame}/{self.sequence.render_frames} "
                            f"({time.monotonic() - started_at:.0f}s)",
                            flush=True,
                        )
                completed = True
                if not self.show_ui or not self.app.is_running():
                    break
        finally:
            for writer in writers.values():
                writer.close()
            if writers and not completed:
                # Never leave a truncated clip that could pass for a good run.
                for name in writers:
                    (self.output_dir / f"{name}.mp4").unlink(missing_ok=True)
            self.head_camera_distinct_frames = len(digests)
            elapsed = time.monotonic() - started_at
            print(
                f"[replay] rendered {len(digests)} distinct head frames over "
                f"{self.sequence.render_frames} frames in {elapsed:.1f}s "
                f"({self.sequence.render_frames / max(elapsed, 1e-6):.1f} fps)",
                flush=True,
            )

    def _open_ui(self) -> None:
        from omni import ui
        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Sdf

        panels = (
            ("GRAIL Replay Head Camera", self._head_camera_path, 120),
            ("GRAIL Replay External Camera", "/World/ExternalCamera", 500),
        )
        for name, camera_path, position_y in panels:
            window = create_viewport_window(
                name=name,
                width=640,
                height=400,
                position_x=900,
                position_y=position_y,
                camera_path=Sdf.Path(camera_path),
            )
            if window is None:
                raise ReplayError(f"failed to create the {name} viewport")
            self._viewport_windows.append(window)
        self._window = ui.Window("GRAIL Replay", width=340, height=80, position_x=900, position_y=40)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label(
                    f"{self.sequence.key}: {self.sequence.frames} source frames @ "
                    f"{self.sequence.fps:g} Hz, rendered @ {self.sequence.render_fps:g} Hz"
                )
                ui.Label("Kinematic replay, no physics.")
