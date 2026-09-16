"""AppleToPlate adapter: verified body order, blocked normalized hand action.

``info.json`` names no channels; ``modality.json`` gives the block boundaries
(left_leg 6, right_leg 6, waist 3, left_arm 7, right_arm 7, left_hand 7,
right_hand 7) and two different 43D fields carry different semantics:

* ``action`` — desired joint *commands*.  The hand block is a quantized
  **normalized open/close signal**: zero is open, the first four channels go
  negative to close, the closed tuple is about ``[-1, -1, -1, -1, 0, 0.4, 0.7]``,
  the right hand is exactly zero in all 402 episodes, and the mapping from that
  signal to Dex3 actuator radians is unproven.  The 78D final action therefore
  stays closed (:func:`require_hand_schema`); the raw commands are preserved as a
  pilot artifact instead.
* ``observation.state`` — measured joint qpos in **radians**.  Its hand blocks
  are already in canonical Dex3 motor order (left and right), which the state
  and joint-limit analysis confirms: the left block's one-sided channels line up
  with the Dex3 ranges (``thumb_2`` non-negative, index/middle non-positive), and
  the right block mirrors them.  This field is what
  :func:`load_state_reference` decodes for frame-exact visualisation; it does not
  open the normalized action hand.

The body block interior is the Unitree SDK/MuJoCo order for both fields, now
verified by joint-limit fit, flat-footed standing in the recordings and the
camera view, so the body no longer needs an assumption flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER

from ..joints import hand_order, sonic_reference_body
from ..reference import StandingPose
from ..schema import HAND_SCHEMA_UNRESOLVED, HAND_SCHEMA_VERIFIED, CanonicalEpisode, CanonicalEpisodeBuild
from ..state_reference import StateReference, build_state_reference
from ..timeline import finite_difference, linear_resample, uniform_timeline

APPLE_BLOCK_ORDER = ("left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand")
BODY_BLOCKS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")
BODY_ORDER = "unitree_mujoco_order_within_each_modality_block"
BODY_ORDER_EVIDENCE = (
    "joint-limit fit on observation.state, flat-footed standing in the recordings, camera view",
)
BODY_ORDER_VERIFIED = True
SOURCE_FPS = 30.0
HAND_SCHEMA_STATUS = "unresolved"

HAND_BLOCK_SLICES = {"left": slice(29, 36), "right": slice(36, 43)}
#: Both hand blocks use the canonical Dex3 motor order per side.
LEFT_HAND_ORDER = ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "middle_0_joint", "middle_1_joint", "index_0_joint", "index_1_joint")
RIGHT_HAND_ORDER = ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint")
#: Source slot -> canonical slot; the block interior already is canonical order.
HAND_BLOCK_IDENTITY_MAP = (0, 1, 2, 3, 4, 5, 6)
#: Channels the normalized *action* drives in equal pairs (index pair, middle pair).
LEFT_EQUAL_PAIRS = ((0, 1), (2, 3))
#: Documented closed-hand tuple of the normalized action command.
DOCUMENTED_CLOSED_TUPLE = (-1.0, -1.0, -1.0, -1.0, 0.0, 0.4, 0.7)
#: Observed per-channel envelope of the normalized action command (not radians).
LEFT_NORMALIZED_ENVELOPE = (
    (-1.0, 0.0),
    (-1.0, 0.0),
    (-1.0, 0.0),
    (-1.0, 0.0),
    (-0.5, 0.5),
    (0.0, 0.4),
    (0.0, 0.7),
)
ENVELOPE_TOLERANCE = 1e-3
CLOSED_THRESHOLD = -0.9

HAND_BLOCK_REASON = (
    "left and right action hand blocks are quantized normalized open/close commands in canonical Dex3 "
    "motor order, but the normalized-command -> Dex3 radian actuator mapping is unproven and the right "
    "action hand is exactly zero in all 402 episodes; the recorded radians live in observation.state"
)


def require_hand_schema() -> None:
    raise RuntimeError(
        "AppleToPlate hand schema is unresolved: the channels are normalized open/close commands "
        "with an unproven radian mapping and an all-zero right hand; the 78D final action stays blocked"
    )


def body_block_slices(dataset: Path) -> dict[str, slice]:
    modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))["action"]
    # The joint blocks come first; the effort/navigate entries that follow reuse
    # index 0 because they address their own parquet columns.
    blocks = tuple(modality)[: len(APPLE_BLOCK_ORDER)]
    if blocks != APPLE_BLOCK_ORDER:
        raise ValueError(f"unexpected AppleToPlate action block order: {blocks}")
    slices = {name: slice(int(modality[name]["start"]), int(modality[name]["end"])) for name in APPLE_BLOCK_ORDER}
    if slices["right_hand"].stop != 43:
        raise ValueError("AppleToPlate action must be 43D")
    state = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8")).get("state") or {}
    for name in APPLE_BLOCK_ORDER:
        if name in state and (int(state[name]["start"]), int(state[name]["end"])) != (slices[name].start, slices[name].stop):
            raise ValueError(f"AppleToPlate state and action disagree on block {name!r}")
    return slices


def body_names(slices: dict[str, slice]) -> tuple[str, ...]:
    """Body channel names implied by the declared block boundaries."""
    return tuple(
        name
        for block in BODY_BLOCKS
        for name in BODY_JOINT_ORDER[slices[block]]
    )


#: Kept for callers that referenced the old name.
assumed_body_names = body_names


def hand_command_evidence(
    dataset: Path,
    episode_index: int | None = None,
    *,
    episode_files: list[Path] | None = None,
) -> dict[str, object]:
    """Measure the normalized *action* hand blocks of one episode or the collection."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("AppleToPlate conversion requires pyarrow") from exc

    slices = body_block_slices(dataset)
    if slices["left_hand"] != HAND_BLOCK_SLICES["left"] or slices["right_hand"] != HAND_BLOCK_SLICES["right"]:
        raise ValueError(
            f"AppleToPlate hand blocks moved: left {slices['left_hand']}, right {slices['right_hand']}"
        )
    if episode_files is None:
        if episode_index is None:
            raise ValueError("hand evidence needs an episode index")
        episode_files = [dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"]
    left_blocks: list[np.ndarray] = []
    right_blocks: list[np.ndarray] = []
    for path in episode_files:
        table = pq.read_table(path, columns=["action"])
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
        left_blocks.append(action[:, slices["left_hand"]])
        right_blocks.append(action[:, slices["right_hand"]])
    left = np.concatenate(left_blocks, axis=0)
    right = np.concatenate(right_blocks, axis=0)
    pair_error = max(
        float(np.abs(left[:, first] - left[:, second]).max()) for first, second in LEFT_EQUAL_PAIRS
    )
    closed_mask = left[:, 0] <= CLOSED_THRESHOLD
    return {
        "episodes": len(episode_files),
        "frames": int(left.shape[0]),
        "left_min": [float(value) for value in left.min(axis=0)],
        "left_max": [float(value) for value in left.max(axis=0)],
        "left_non_finite": int((~np.isfinite(left)).sum()),
        "left_pair_max_abs_error": pair_error,
        "left_negative_counts": [int(np.count_nonzero(left[:, index] < 0.0)) for index in range(7)],
        "right_nonzero_frames": int(np.count_nonzero(right)),
        "right_max_abs": float(np.abs(right).max()),
        "closed_frames": int(closed_mask.sum()),
        "closed_mean_tuple": [float(value) for value in left[closed_mask].mean(axis=0)] if closed_mask.any() else None,
        "documented_closed_tuple": list(DOCUMENTED_CLOSED_TUPLE),
        "interior_order": list(LEFT_HAND_ORDER),
        "source_to_canonical_slot": list(HAND_BLOCK_IDENTITY_MAP),
    }


def validate_hand_schema(evidence: dict[str, object]) -> None:
    """Fail closed when the documented normalized action hand schema changes."""
    if int(evidence["left_non_finite"]) != 0 or int(evidence["right_nonzero_frames"]) != 0:
        raise ValueError(
            "AppleToPlate hand schema changed: non-finite left values or a nonzero right hand "
            f"({evidence['left_non_finite']} non-finite, {evidence['right_nonzero_frames']} nonzero right frames)"
        )
    if float(evidence["right_max_abs"]) != 0.0:
        raise ValueError("AppleToPlate right action hand is no longer identically zero")
    if float(evidence["left_pair_max_abs_error"]) != 0.0:
        raise ValueError(
            "AppleToPlate left hand channel pairs are no longer equal "
            f"(max pair error {evidence['left_pair_max_abs_error']}); the documented interior drive is gone"
        )
    for index, (lower, upper) in enumerate(LEFT_NORMALIZED_ENVELOPE):
        observed_low = float(evidence["left_min"][index])
        observed_high = float(evidence["left_max"][index])
        if observed_low < lower - ENVELOPE_TOLERANCE or observed_high > upper + ENVELOPE_TOLERANCE:
            raise ValueError(
                f"AppleToPlate left hand channel {index} left the documented normalized envelope "
                f"[{lower}, {upper}]: observed [{observed_low}, {observed_high}]"
            )
    if int(evidence["closed_frames"]) == 0:
        raise ValueError(
            "AppleToPlate episode never closes the left hand; the documented negative-is-close "
            "direction cannot be confirmed on this episode"
        )
    for index in (0, 1, 2, 3):
        if int(evidence["left_negative_counts"][index]) == 0:
            raise ValueError(f"AppleToPlate close channel {index} is never negative in this selection")


def _read_episode(dataset: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("AppleToPlate conversion requires pyarrow") from exc

    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    if float(info["fps"]) != SOURCE_FPS:
        raise ValueError(f"unexpected AppleToPlate fps: {info['fps']}")
    slices = body_block_slices(dataset)
    path = dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"AppleToPlate episode parquet missing: {path}")
    table = pq.read_table(path, columns=["action", "observation.state", "timestamp"])
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
    if len(action) < 2 or len(state) != len(action) or len(timestamps) != len(action):
        raise ValueError(f"AppleToPlate episode {episode_index} has inconsistent rows")
    return action, state, timestamps, slices


def load_state_reference(
    dataset: Path,
    episode_index: int,
    *,
    limits: dict[str, tuple[float, float]],
) -> StateReference:
    """Measured ``observation.state`` at 30 Hz: canonical body29 and Dex3 radians."""
    _, state, timestamps, slices = _read_episode(dataset, episode_index)
    body_source = np.concatenate([state[:, slices[block]] for block in BODY_BLOCKS], axis=1)
    body = sonic_reference_body(body_source, body_names(slices))
    left = state[:, slices["left_hand"]]
    right = state[:, slices["right_hand"]]
    if left.shape[1] != 7 or right.shape[1] != 7:
        raise ValueError("AppleToPlate state hand blocks must be 7D per side")
    return build_state_reference(
        timestamps,
        body,
        left,
        right,
        source_field="observation.state",
        source_fps=SOURCE_FPS,
        limits=limits,
        mapping={
            "body_joint_order_source": "verified Unitree SDK/MuJoCo order within each modality block",
            "body_joint_order_evidence": list(BODY_ORDER_EVIDENCE),
            "hand_joint_order_source": "canonical Dex3 motor order (verified against state and joint limits)",
            "hand_units": "radians (measured qpos)",
            "blocks": {name: [slices[name].start, slices[name].stop] for name in APPLE_BLOCK_ORDER},
        },
    )


def load_action_source_reference(dataset: Path, episode_index: int) -> StateReference:
    """The body action at 30 Hz (row i <-> frame i); hands stay blocked.

    The action hand block is a normalized open/close command, so it is not
    applied: the replay writes the body from the action and leaves the hands at
    their zero (open) state, exactly as the 78D action remains blocked.
    """
    action, _, timestamps, slices = _read_episode(dataset, episode_index)
    body_source = np.concatenate([action[:, slices[block]] for block in BODY_BLOCKS], axis=1)
    body = sonic_reference_body(body_source, body_names(slices))
    velocity = np.zeros_like(body)
    velocity[:-1] = np.diff(body, axis=0) * SOURCE_FPS
    zeros = np.zeros((len(body), 7), dtype=np.float32)
    reference = StateReference(
        timestamps=timestamps,
        joint_pos=body.astype(np.float32),
        joint_vel=velocity.astype(np.float32),
        left_hand_joints=zeros,
        right_hand_joints=zeros.copy(),
        hands_applied=False,
        provenance={
            "source_field": "action",
            "semantics": "desired body command, 30 Hz, row i <-> frame i; hand block is normalized and not applied",
            "source_fps": SOURCE_FPS,
            "velocity_source": "finite difference of the action trajectory at source rate",
            "body_joint_order": "official SONIC reference / IsaacLab",
            "hand_units": "not applied (normalized open/close command, 78D blocked)",
            "hands_applied_in_replay": False,
        },
    )
    reference.validate()
    return reference


def load_body_pilot(
    dataset: Path,
    episode_index: int,
    *,
    standing: StandingPose,
    assume_unitree_mujoco_body_order: bool = True,
    hands_from_state: bool = False,
) -> CanonicalEpisodeBuild:
    """Load the 29-channel body trajectory; the normalized hands stay blocked.

    The body block interior order is verified (see module docstring), so the flag
    is accepted for backwards compatibility only.  The raw normalized hand
    commands are kept in ``raw_hand_command`` and never enter the canonical hand
    slots.

    ``hands_from_state`` optionally fills the canonical hand slots from
    ``observation.state``, which *is* measured qpos in radians in canonical Dex3
    motor order (see :func:`load_state_reference`).  It is off by default because
    it changes the meaning of the emitted 78D action from "commanded" to
    "measured", so a caller has to ask for it explicitly and the manifest records
    which field supplied the hands.
    """
    action, state, timestamps, slices = _read_episode(dataset, episode_index)
    evidence = hand_command_evidence(dataset, episode_index)
    validate_hand_schema(evidence)
    body_source = np.concatenate([action[:, slices[block]] for block in BODY_BLOCKS], axis=1)
    body = sonic_reference_body(body_source, body_names(slices))
    target = uniform_timeline(timestamps)
    body_resampled = linear_resample(timestamps, body, target)
    zeros = np.zeros((len(target), 7), dtype=np.float32)
    if hands_from_state:
        # Measured radians, already in canonical Dex3 motor order per side.
        left_hands = linear_resample(timestamps, state[:, slices["left_hand"]], target)
        right_hands = linear_resample(timestamps, state[:, slices["right_hand"]], target)
        hand_source = "observation.state"
    else:
        left_hands, right_hands = zeros, zeros.copy()
        hand_source = "zeros (placeholder)"
    episode: CanonicalEpisode = CanonicalEpisode(
        timestamps=target,
        joint_pos=body_resampled.astype(np.float32),
        joint_vel=finite_difference(body_resampled).astype(np.float32),
        body_quat_wxyz=np.tile(standing.body_quat_wxyz.astype(np.float32), (len(target), 1)),
        body_pos=np.tile(standing.body_pos.astype(np.float32), (len(target), 1)),
        # Without ``hands_from_state`` the hand slots stay zero because the
        # normalized action command has no proven radian mapping; the build status
        # keeps them out of any 78D action.
        left_hand_joints=left_hands.astype(np.float32),
        right_hand_joints=right_hands.astype(np.float32),
    )
    episode.validate()
    provenance = {
        "source_collection": "nvidia_gr00t_n1.7_apple_to_plate",
        "source_kind": "lerobot_v2.1",
        "source_fps": SOURCE_FPS,
        "processed_fps": 50.0,
        "source_semantics": "43D action = desired joint commands; observation.state = measured joint qpos",
        "source_modality_blocks": {name: [slices[name].start, slices[name].stop] for name in APPLE_BLOCK_ORDER},
        "body_joint_order_source": "verified",
        "body_joint_order_assumption": BODY_ORDER,
        "body_joint_order_verified": BODY_ORDER_VERIFIED,
        "body_joint_order_evidence": list(BODY_ORDER_EVIDENCE),
        "hand_joint_order_source": "canonical Dex3 motor order (action block is normalized, state block is radian)",
        "hand_schema_status": HAND_SCHEMA_STATUS,
        "hand_block_boundaries": {
            name: [HAND_BLOCK_SLICES[name].start, HAND_BLOCK_SLICES[name].stop] for name in ("left", "right")
        },
        "hand_order_left": list(LEFT_HAND_ORDER),
        "hand_order_right": list(RIGHT_HAND_ORDER),
        "hand_source_to_canonical_slot": list(HAND_BLOCK_IDENTITY_MAP),
        "hand_command_semantics": "action: quantized normalized open/close command; state: measured radians",
        "hand_command_is_radians": False,
        "hand_command_evidence": evidence,
        "hand_values_written": hand_source,
        "hand_values_are_measured": bool(hands_from_state),
        "raw_hand_command_artifact": "raw_hand_command.npz (source rate, normalized action)",
        "visual_state_artifact": "reference_state.npz (observation.state, radians, canonical Dex3 order)",
        "velocity_source": "re-derived by finite difference of the resampled 50 Hz positions",
        "resampling": "linear interpolation of the action trajectory onto a uniform 50 Hz timeline",
        "root_assumption": "synthetic_static_standing_root",
        "root_assumption_detail": {
            "body_pos": [float(value) for value in standing.body_pos],
            "body_quat_wxyz": [float(value) for value in standing.body_quat_wxyz],
            "source": standing.source,
        },
        "episode_frames": int(action.shape[0]),
        "name_check": {
            "body": {
                "mode": "name_round_trip",
                "source": list(body_names(slices)),
                "target": list(SONIC_REFERENCE_JOINT_ORDER),
            },
            "left_hand": f"canonical order, identity map to {list(hand_order('left'))}",
            "right_hand": f"canonical order, identity map to {list(hand_order('right'))}",
        },
    }
    raw_hand_command = {
        "timestamps": timestamps,
        "left": np.ascontiguousarray(action[:, slices["left_hand"]], dtype=np.float64),
        "right": np.ascontiguousarray(action[:, slices["right_hand"]], dtype=np.float64),
    }
    if hands_from_state:
        notes = (
            f"body channel order is verified ({BODY_ORDER})",
            "hand channels are the measured observation.state radians in canonical Dex3 order "
            "(hands_from_state); the normalized action hand block is NOT used",
            "the right hand is nearly static in most episodes, so the right-hand channels carry "
            "little motion even though they are real measurements",
            "raw normalized hand commands are preserved in raw_hand_command.npz (source rate)",
        )
        return CanonicalEpisodeBuild(
            episode=episode,
            provenance=provenance,
            hand_schema_status=HAND_SCHEMA_VERIFIED,
            hand_schema_reason=(
                "hand radians come from observation.state (measured qpos) in canonical Dex3 motor order, "
                "not from the unresolved normalized action command; requested explicitly via hands_from_state"
            ),
            notes=notes,
            raw_hand_command=raw_hand_command,
        )
    notes = (
        f"body channel order is verified ({BODY_ORDER})",
        "hand action is a normalized open/close signal without a proven Dex3 radian mapping: "
        "this pilot may only produce a 64D body latent",
        "raw normalized hand commands are preserved in raw_hand_command.npz (source rate)",
    )
    return CanonicalEpisodeBuild(
        episode=episode,
        provenance=provenance,
        hand_schema_status=HAND_SCHEMA_UNRESOLVED,
        hand_schema_reason=HAND_BLOCK_REASON,
        notes=notes,
        raw_hand_command=raw_hand_command,
    )
