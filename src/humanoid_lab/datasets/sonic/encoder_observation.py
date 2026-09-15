"""Exact 1751D G1 observation layout for the pinned SONIC v1.1 encoder.

Semantics are taken from the pinned upstream source
(``NVlabs/GR00T-WholeBodyControl`` at the commit in ``versions.lock.yaml``):

* layout — ``observation_config.yaml`` lists the encoder observations in input
  order; the deployment computes each offset from that order
  (``g1_deploy_onnx_ref.cpp``: ``InitializeObservationFunctions``) and asserts
  the total against the ONNX input dimension.
* mode block — ``GatherEncoderMode`` writes the motion's mode id into the first
  slot and zero-fills the rest, so G1 (mode id 0) is ``[0, 0, 0, 0]``, not a
  one-hot vector.
* mode filtering — for mode id 0 only the four observations listed under
  ``encoder_modes.g1.required_observations`` are computed; every other slot in
  the 1751D input stays zero (``GatherEncoderObservations``).
* look-ahead — 10 frames at step 5 control ticks (20 ms) span 0.9 s, and frames
  past the end of the motion clamp to the last frame (``Gather...MultiFrame``).
* orientation — the encoder consumes ``motion_anchor_orientation_heading``,
  i.e. the reference root rotation made relative to the *robot's* heading
  (:mod:`humanoid_lab.datasets.sonic.heading`).  Raw world orientation is not
  the encoder contract.

Two different things share the word "heading" and must not be conflated:

* live C++ semantics — the base quaternion of the robot at control time
  (``state_logger_`` history) plus the operator's ``apply_delta_heading``;
* offline dataset production — no robot state is recorded, so the conversion
  has to assume a base orientation.  :class:`OrientationPolicy` names the
  assumption, :func:`require_conversion_policy` refuses every policy that is not
  a documented, deterministic offline choice, and callers must record the chosen
  policy in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from .heading import heading_relative_anchor_6d, world_quaternion_to_6d
from .schema import CanonicalEpisode

ENCODER_INPUT_DIM = 1751
ENCODER_TOKEN_DIM = 64
CONTROL_DT_S = 0.02
FRAME_COUNT = 10
FRAME_STEP = 5
FUTURE_OFFSETS = np.arange(0, FRAME_COUNT * FRAME_STEP, FRAME_STEP, dtype=np.int64)

#: Encoder observation names and dimensions of the official registry, in the
#: order the pinned ``observation_config.yaml`` enables them.
PINNED_ENCODER_OBSERVATIONS: tuple[tuple[str, int], ...] = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_anchor_orientation_heading_10frame_step5", 60),
    ("motion_anchor_orientation_heading", 6),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_heading_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)

#: Mode-filtered observations of ``encoder_modes.g1``; everything else is zero.
G1_REQUIRED_OBSERVATIONS: tuple[str, ...] = (
    "encoder_mode_4",
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_anchor_orientation_heading_10frame_step5",
)
G1_MODE_ID = 0


class OrientationPolicy(str, Enum):
    """Which quaternion stands in for the robot's live base orientation."""

    #: Offline default: the robot is assumed to track the reference heading, so
    #: the base heading is the reference root heading at the *current* frame.
    #: Algebraically identical to upstream's ``refheading`` variant
    #: (``motion_anchor_orientation_refheading_*``) with an identity
    #: ``apply_delta_heading``, which is why it needs no robot state.
    REFERENCE_ROOT_CURRENT_FRAME = "reference_root_current_frame"
    #: Live C++ semantics: an explicit measured base quaternion per frame.
    #: Requires ``base_quat_wxyz``; the datasets have no such recording.
    MEASURED_BASE = "measured_base"
    #: Raw world reference orientation in the orientation slots.  This is the
    #: historical Python behaviour and it encodes the world yaw the heading
    #: normalization exists to remove; diagnostic comparisons only.
    LEGACY_RAW_WORLD = "legacy_raw_world"


#: Policies allowed to produce training data.
CONVERSION_POLICIES: frozenset[OrientationPolicy] = frozenset({OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME})


@dataclass(frozen=True)
class EncoderLayout:
    """Offsets of the pinned encoder input, derived from the enabled observations."""

    observations: tuple[tuple[str, int], ...]
    offsets: dict[str, int]
    total_dim: int

    @classmethod
    def from_observations(cls, observations: tuple[tuple[str, int], ...]) -> "EncoderLayout":
        offsets: dict[str, int] = {}
        offset = 0
        for name, dim in observations:
            if name in offsets:
                raise ValueError(f"encoder observation {name!r} is listed twice")
            offsets[name] = offset
            offset += dim
        return cls(observations=observations, offsets=offsets, total_dim=offset)

    def offset(self, name: str) -> int:
        try:
            return self.offsets[name]
        except KeyError as error:
            raise ValueError(f"encoder observation {name!r} is not part of this layout") from error

    def span(self, name: str) -> slice:
        width = dict(self.observations)[name]
        return slice(self.offset(name), self.offset(name) + width)


PINNED_LAYOUT = EncoderLayout.from_observations(PINNED_ENCODER_OBSERVATIONS)


def load_encoder_layout(config_path: Path) -> EncoderLayout:
    """Read ``observation_config.yaml`` and fail closed on any deviation.

    The layout is the official contract, so a mismatch between our pinned copy
    and the file shipped next to the pinned ONNX model must stop conversion
    rather than silently shift every offset.
    """
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    encoder = document.get("encoder") or {}
    if int(encoder.get("dimension", 0)) != ENCODER_TOKEN_DIM:
        raise ValueError("observation config encoder dimension must be 64")
    names = [item["name"] for item in encoder.get("encoder_observations", []) if item.get("enabled")]
    dims = dict(PINNED_ENCODER_OBSERVATIONS)
    unknown = [name for name in names if name not in dims]
    if unknown:
        raise ValueError(f"observation config enables unregistered observations: {unknown}")
    observations = tuple((name, dims[name]) for name in names)
    if observations != PINNED_ENCODER_OBSERVATIONS:
        raise ValueError("observation config encoder observations differ from the pinned layout")
    modes = {str(mode["name"]): mode for mode in encoder.get("encoder_modes", [])}
    g1 = modes.get("g1")
    if g1 is None or int(g1.get("mode_id", -1)) != G1_MODE_ID:
        raise ValueError("observation config must define encoder mode 'g1' with mode_id 0")
    required = tuple(str(name) for name in g1.get("required_observations", []))
    if sorted(required) != sorted(G1_REQUIRED_OBSERVATIONS):
        raise ValueError(f"G1 mode requires {required}, expected {G1_REQUIRED_OBSERVATIONS}")
    layout = EncoderLayout.from_observations(observations)
    if layout.total_dim != ENCODER_INPUT_DIM:
        raise ValueError(f"encoder layout totals {layout.total_dim}, expected {ENCODER_INPUT_DIM}")
    return layout


def quaternion_to_rotation_6d(wxyz: np.ndarray) -> np.ndarray:
    """Raw world orientation 6D; kept for diagnostics, not the encoder contract."""
    return world_quaternion_to_6d(wxyz)


def require_conversion_policy(policy: OrientationPolicy) -> OrientationPolicy:
    """Refuse orientation policies that cannot be reproduced offline."""
    policy = OrientationPolicy(policy)
    if policy not in CONVERSION_POLICIES:
        raise ValueError(
            f"orientation policy {policy.value!r} must not produce training data; "
            f"allowed: {sorted(item.value for item in CONVERSION_POLICIES)} "
            "(diagnostic comparisons stay in tests and QC tooling)"
        )
    return policy


def future_indices(frames: int, offsets: np.ndarray = FUTURE_OFFSETS) -> tuple[np.ndarray, np.ndarray]:
    """Look-ahead indices plus the per-row fraction of clamped look-ahead samples."""
    if frames <= 0:
        raise ValueError("episode must contain at least one frame")
    unclamped = np.arange(frames, dtype=np.int64)[:, None] + np.asarray(offsets, dtype=np.int64)[None, :]
    indices = np.minimum(unclamped, frames - 1)
    clamp_fraction = np.mean(unclamped >= frames, axis=1)
    return indices, clamp_fraction


def orientation_block(
    reference_root_quat: np.ndarray,
    indices: np.ndarray,
    policy: OrientationPolicy,
    base_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Heading-normalised anchor orientation for every row and look-ahead frame."""
    reference = np.asarray(reference_root_quat, dtype=np.float64)
    policy = OrientationPolicy(policy)
    if policy is OrientationPolicy.LEGACY_RAW_WORLD:
        return world_quaternion_to_6d(reference[indices]).reshape(len(indices), FRAME_COUNT * 6)
    if policy is OrientationPolicy.MEASURED_BASE:
        if base_quat_wxyz is None:
            raise ValueError("measured_base policy needs a measured base quaternion per frame")
        base = np.asarray(base_quat_wxyz, dtype=np.float64)
        if base.shape != reference.shape:
            raise ValueError("measured base quaternions must match the reference frame count")
    else:
        # Deterministic offline assumption: the robot tracks the reference
        # heading at the current frame, so the current reference root heading
        # stands in for the measured base heading.
        base = reference[:, None, :]
    reference_future = reference[indices]
    if base.ndim == 2:
        base = base[:, None, :]
    return heading_relative_anchor_6d(
        np.broadcast_to(base, reference_future.shape), reference_future
    ).reshape(len(indices), FRAME_COUNT * 6)


def build_g1_encoder_observation(
    episode: CanonicalEpisode,
    *,
    policy: OrientationPolicy = OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME,
    base_quat_wxyz: np.ndarray | None = None,
    layout: EncoderLayout = PINNED_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``[frames, 1751]`` encoder input and the tail-clamp fraction.

    The policy is only checked, not defaulted: a caller that wants the raw world
    orientation has to ask for :attr:`OrientationPolicy.LEGACY_RAW_WORLD`
    explicitly, and conversion entry points reject it.
    """
    episode.validate()
    policy = OrientationPolicy(policy)
    if layout.total_dim != ENCODER_INPUT_DIM:
        raise ValueError(f"encoder layout totals {layout.total_dim}, expected {ENCODER_INPUT_DIM}")
    frames = len(episode.timestamps)
    indices, clamp_fraction = future_indices(frames)
    output = np.zeros((frames, ENCODER_INPUT_DIM), dtype=np.float32)
    mode = np.zeros(4, dtype=np.float32)
    mode[0] = float(G1_MODE_ID)
    output[:, layout.span("encoder_mode_4")] = mode
    output[:, layout.span("motion_joint_positions_10frame_step5")] = episode.joint_pos[indices].reshape(frames, -1)
    output[:, layout.span("motion_joint_velocities_10frame_step5")] = episode.joint_vel[indices].reshape(frames, -1)
    block = orientation_block(episode.body_quat_wxyz, indices, policy, base_quat_wxyz)
    if block.shape != (frames, FRAME_COUNT * 6):
        raise ValueError(f"orientation block has shape {block.shape}")
    output[:, layout.span("motion_anchor_orientation_heading_10frame_step5")] = block
    if not np.isfinite(output).all():
        raise ValueError("encoder observation contains NaN or Inf")
    return output, clamp_fraction.astype(np.float32)
