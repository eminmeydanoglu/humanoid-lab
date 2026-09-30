"""Gate 0 contract for the GR00T N1.7 Unitree Dex3 / SONIC v1.1 training pack.

The decisions frozen here are the ones a converter must never make again: the
output row identity (retained SONIC 50 Hz samples), the official 43D state
layout, the two hand orders that must not be confused, the 45-row clamped tail,
the repair policy for grossly corrupted measured hand samples, and the 13 task
instructions.

Two hand vocabularies exist and neither may be inferred from the other:

* ``observation.state`` stores the hands in the official ``unitree_g1_sonic``
  order, ``index, middle, thumb`` for both sides;
* the action hand fields store them in the official SONIC actuated order,
  ``thumb, index, middle`` for both sides;
* the SONIC corpus stores the left block in the *source motor* order
  (``thumb, middle, index``), which is why ``action[:, 64:78]`` is never copied.

Every channel is placed by name through a permutation derived from the source's
declared joint names, so a re-ordered upstream release fails loudly instead of
filling the wrong joint.

The sources under ``raw_root``/``sonic_root`` are frozen inputs; nothing in this
package writes under them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from humanoid_lab.controllers.sonic import (
    BODY_JOINT_ORDER,
    DEFAULT_STANDING_POSE_RAD,
    LEFT_HAND_JOINT_ORDER,
    RIGHT_HAND_JOINT_ORDER,
)
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import UNITREE_ACTION_NAMES

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = ROOT / "configs/datasets/groot/unitree_dex3_sonic_v1.yaml"

#: Width of the raw Unitree ``observation.state``/``action`` vectors.
SOURCE_STATE_DIM = 28
#: One hand of the official schema.
HAND_DIM = 7
#: Both hands, the width of one action hand pair.
HAND_CHANNELS = 2 * HAND_DIM
#: Stored ``observation.state`` width (legs, waist, arms, hands).
STATE_DIM = 43
#: Model-side state width: the 43 stored channels plus 3D projected gravity.
MODEL_STATE_DIM = 46
#: SONIC v1.1 body latent.
MOTION_TOKEN_DIM = 64
#: body latent + both action hands.
ACTION_DIM = MOTION_TOKEN_DIM + HAND_CHANNELS
#: Rows GR00T's ``unitree_g1_sonic`` action config reads per sample at 50 Hz.
HORIZON = 40
#: Trailing corpus rows whose future window is upstream-clamped, not supervision.
TAIL_ROWS = 45
#: Frame rate of the pack, which is the SONIC corpus frame rate.
FPS = 50
#: Synthetic upright gravity for every frame (no IMU in the source collection).
PROJECTED_GRAVITY = (0.0, 0.0, -1.0)
#: Gross measured hand corruption: a Dex3 motor reading beyond this is not a
#: reachable hand configuration, while its mechanical range is far inside it.
REPAIR_INVALID_ABS_RAD = 3.0

#: Pre-registered embodiment tag the pack is written for. It is the member name
#: of the pinned ``EmbodimentTag`` enum, which is what the official statistics
#: tool's ``tyro`` CLI accepts (its lower-case value is not a valid choice).
EMBODIMENT_TAG = "UNITREE_G1_SONIC"

STATE_FIELD = "observation.state"
GRAVITY_FIELD = "observation.projected_gravity"
MOTION_TOKEN_FIELD = "action.motion_token"
LEFT_HAND_FIELD = "teleop.left_hand_joints"
RIGHT_HAND_FIELD = "teleop.right_hand_joints"
#: Language annotation: the instruction is the task table's string, reached
#: through the frame's ``task_index``.
INSTRUCTION_KEY = "human.task_description"
INSTRUCTION_ORIGINAL_KEY = "task_index"
EGO_VIEW_KEY = "ego_view"

#: Official ``unitree_g1_sonic`` state hand order, both sides.
STATE_HAND_ORDER: dict[str, tuple[str, ...]] = {
    "left": (
        "index_0_joint", "index_1_joint", "middle_0_joint",
        "middle_1_joint", "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
    ),
    "right": (
        "index_0_joint", "index_1_joint", "middle_0_joint",
        "middle_1_joint", "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
    ),
}

#: Official SONIC actuated hand order used by the action hand fields, both sides.
ACTION_HAND_ORDER: dict[str, tuple[str, ...]] = {
    "left": (
        "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
        "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint",
    ),
    "right": (
        "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
        "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint",
    ),
}

#: Per-side Dex3 motor order the SONIC v1.1 corpus stores in ``action[:, 64:78]``.
CORPUS_HAND_ORDER: dict[str, tuple[str, ...]] = {
    "left": tuple(LEFT_HAND_JOINT_ORDER),
    "right": tuple(RIGHT_HAND_JOINT_ORDER),
}

SIDES = ("left", "right")
#: Stored block order of ``observation.state``, which is the official exporter's:
#: the lower body, then each arm followed by its hand.
STATE_BLOCKS: dict[str, slice] = {
    "left_leg": slice(0, 6),
    "right_leg": slice(6, 12),
    "waist": slice(12, 15),
    "left_arm": slice(15, 22),
    "left_hand": slice(22, 29),
    "right_arm": slice(29, 36),
    "right_hand": slice(36, 43),
}
#: The stored block order above, left to right.
STATE_STORAGE_KEYS: tuple[str, ...] = tuple(STATE_BLOCKS)
#: State keys in the order GR00T concatenates them, gravity last. This is the
#: pre-registered ``unitree_g1_sonic`` key order, which is *not* the stored order:
#: the loader addresses every key through its slice, so ``right_arm`` (29:36) is
#: read before ``left_hand`` (22:29) and the model still sees arms then hands.
STATE_MODEL_KEYS: tuple[str, ...] = (
    "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand", "projected_gravity",
)
ACTION_MODEL_KEYS: tuple[str, ...] = ("motion_token", "left_hand_joints", "right_hand_joints")


def _hand_names(side: str, order: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_hand_{stem}" for stem in order)


#: Canonical 43D state order: 12 leg joints, 3 waist joints, the left arm and
#: hand, then the right arm and hand, with the 14 hand channels in the official
#: state hand order.
CANONICAL_STATE_NAMES: tuple[str, ...] = (
    tuple(BODY_JOINT_ORDER[:22])
    + _hand_names("left", STATE_HAND_ORDER["left"])
    + tuple(BODY_JOINT_ORDER[22:29])
    + _hand_names("right", STATE_HAND_ORDER["right"])
)

#: Names of one action hand pair in the official actuated order.
CANONICAL_ACTION_HAND_NAMES: tuple[str, ...] = tuple(
    f"{side}_hand_{stem}" for side in SIDES for stem in ACTION_HAND_ORDER[side]
)


def _unitree_field_name(canonical_name: str) -> str:
    """``left_hand_index_0_joint`` -> ``kLeftHandIndex0``.

    The raw collection names every channel in camel case with a ``k`` prefix;
    this is the inverse of the map in
    :mod:`humanoid_lab.datasets.sonic.adapters.unitree_dex3`.
    """
    parts = canonical_name.replace("_joint", "").split("_")
    return "k" + "".join(part.title() for part in parts)


#: Raw source field of every measured channel (arms and hands, 15:43 by name).
MEASURED_STATE_FIELDS: tuple[str, ...] = tuple(_unitree_field_name(name) for name in CANONICAL_STATE_NAMES[15:43])

#: Source field of each action hand channel, in the official actuated order.
ACTION_HAND_FIELDS: tuple[str, ...] = tuple(
    _unitree_field_name(name) for name in CANONICAL_ACTION_HAND_NAMES
)


def assert_state_layout() -> None:
    """The 43D order and both hand vocabularies, checked by name."""
    if len(CANONICAL_STATE_NAMES) != STATE_DIM or len(set(CANONICAL_STATE_NAMES)) != STATE_DIM:
        raise ValueError("the canonical state must be 43 distinct channels")
    if tuple(CANONICAL_STATE_NAMES[:15]) != tuple(BODY_JOINT_ORDER[:15]):
        raise ValueError("the lower body must be the first 15 channels in hardware order")
    if STATE_STORAGE_KEYS != ("left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"):
        raise ValueError("the stored blocks must be the official exporter order: lower body, arm, hand, arm, hand")
    if set(STATE_STORAGE_KEYS) != set(STATE_MODEL_KEYS[:7]):
        raise ValueError("the stored blocks and the registered modality keys must name the same seven blocks")
    cursor = 0
    for key in STATE_STORAGE_KEYS:
        block = STATE_BLOCKS[key]
        if block.start != cursor or block.stop - block.start != len(CANONICAL_STATE_NAMES[block]):
            raise ValueError(f"state block {key!r} does not tile the 43D state in stored order")
        cursor = block.stop
    if CANONICAL_STATE_NAMES[22:29] != _hand_names("left", STATE_HAND_ORDER["left"]) or CANONICAL_STATE_NAMES[
        36:43
    ] != _hand_names("right", STATE_HAND_ORDER["right"]):
        raise ValueError("the stored hand blocks are not the declared official state hand order")
    if CANONICAL_STATE_NAMES[15:22] != tuple(BODY_JOINT_ORDER[15:22]) or CANONICAL_STATE_NAMES[
        29:36
    ] != tuple(BODY_JOINT_ORDER[22:29]):
        raise ValueError("the stored arm blocks are not the pinned Unitree arm joints")
    for side in SIDES:
        state_order = STATE_HAND_ORDER[side]
        action_order = ACTION_HAND_ORDER[side]
        pinned = LEFT_HAND_JOINT_ORDER if side == "left" else RIGHT_HAND_JOINT_ORDER
        for label, order in (("state", state_order), ("action", action_order)):
            if sorted(order) != sorted(pinned):
                raise ValueError(f"the {label} {side} hand order is not the pinned Dex3 joint set")
    if ACTION_HAND_ORDER["left"] == CORPUS_HAND_ORDER["left"]:
        raise ValueError(
            "the official left actuated order must differ from the corpus motor order: copying "
            "action[:, 64:78] instead of rebuilding the hands is the failure this contract prevents"
        )
    unknown = sorted(set(MEASURED_STATE_FIELDS) - set(UNITREE_ACTION_NAMES))
    if unknown:
        raise ValueError(f"the state map names channels the source does not declare: {unknown}")
    if len(MEASURED_STATE_FIELDS) != 28 or MEASURED_STATE_FIELDS[0] != "kLeftShoulderPitch":
        raise ValueError("the measured block must be the raw Unitree arm and hand field order")
    if MEASURED_STATE_FIELDS[14] != "kRightShoulderPitch":
        raise ValueError("the right arm must follow the left hand in the stored measured block")


def assert_source_names(names: tuple[str, ...]) -> None:
    """Fail closed unless the raw 28D fields carry exactly the declared names."""
    if tuple(names) != UNITREE_ACTION_NAMES:
        raise ValueError("the raw observation.state/action are not the declared 28D Unitree Dex3 layout")


def _positions(names: tuple[str, ...], fields: tuple[str, ...], label: str) -> tuple[int, ...]:
    assert_source_names(names)
    position = {name: index for index, name in enumerate(names)}
    missing = [field for field in fields if field not in position]
    if missing:
        raise ValueError(f"{label}: the source is missing declared channels {missing}")
    return tuple(position[field] for field in fields)


def source_state_permutation(names: tuple[str, ...]) -> tuple[int, ...]:
    """Indices lifting the raw 28D measured state into stored channels 15:43."""
    return _positions(names, MEASURED_STATE_FIELDS, "measured state")


def source_action_hand_permutation(names: tuple[str, ...]) -> tuple[int, ...]:
    """Indices lifting the raw 28D desired action into the official 14D hand order."""
    return _positions(names, ACTION_HAND_FIELDS, "action hands")


def corpus_hand_permutation() -> tuple[int, ...]:
    """Indices mapping the corpus hand block (``action[:, 64:78]``) into actuated order.

    The left corpus block is the source motor order, so this is a real
    permutation and not the identity; a validator compares the rebuilt hands
    against the corpus only through it.
    """
    indices: list[int] = []
    for side_index, side in enumerate(SIDES):
        order = CORPUS_HAND_ORDER[side]
        for stem in ACTION_HAND_ORDER[side]:
            try:
                offset = order.index(stem)
            except ValueError as error:  # pragma: no cover - guarded by assert_state_layout
                raise ValueError(f"the corpus {side} hand order is missing {stem!r}") from error
            indices.append(side_index * HAND_DIM + offset)
    return tuple(indices)


def standing_lower_body() -> np.ndarray:
    """The 15D legs+waist proxy in stored order.

    The official SONIC deployment standing targets, not a measured trajectory:
    the collection records no leg or waist observation, so this is the same
    constant the Psi0 pack uses and it is recorded as synthetic in provenance.
    """
    return np.asarray(
        [DEFAULT_STANDING_POSE_RAD[name] for name in CANONICAL_STATE_NAMES[:15]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CollectionContract:
    """One source collection, its exclusions and its canonical instruction."""

    name: str
    excluded: tuple[int, ...]
    instruction: str
    exclusion_reason: str | None = None

    def usable_episodes(self, total_episodes: int) -> tuple[int, ...]:
        dropped = set(self.excluded)
        return tuple(index for index in range(total_episodes) if index not in dropped)


@dataclass(frozen=True)
class ConversionConfig:
    """The Gate 0 contract, loaded from the machine-readable YAML."""

    path: Path
    sha256: str
    name: str
    raw: dict[str, Any]
    raw_root: Path
    sonic_root: Path
    output_root: Path
    train_repo: str
    val_repo: str
    split_name: str
    split_manifest: Path
    camera_source_key: str
    camera_target_key: str
    camera_excluded_keys: tuple[str, ...]
    camera_width: int
    camera_height: int
    source_fps: int
    sonic_fps: int
    dataset_fps: int
    trailing_invalid_rows: int
    min_rows: int
    horizon: int
    repair_invalid_abs_rad: float
    repair_max_invalid_fraction: float
    repair_max_gap_frames: int
    repair_fail_on_action: bool
    corpus_hand_tolerance_rad: float
    video: dict[str, Any]
    stats_official_tool: str
    stats_own_file: str
    stats_official_file: str
    stats_embodiment_tag: str
    collections: tuple[CollectionContract, ...]
    tasks: dict[str, str]

    @property
    def collection_names(self) -> tuple[str, ...]:
        return tuple(collection.name for collection in self.collections)

    def collection(self, name: str) -> CollectionContract:
        for collection in self.collections:
            if collection.name == name:
                return collection
        raise KeyError(f"{name} is not one of the declared collections")

    def source_total_episodes(self, name: str) -> int:
        """Episode count declared by the frozen source collection metadata."""
        info_path = self.raw_root / name / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"missing source collection metadata: {info_path}")
        return int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])

    def usable_episodes(self, name: str) -> tuple[int, ...]:
        return self.collection(name).usable_episodes(self.source_total_episodes(name))

    def assert_contract(self) -> None:
        """Fail closed if the contract is internally inconsistent."""
        if self.dataset_fps != FPS or self.dataset_fps != self.sonic_fps:
            raise ValueError(
                f"the pack timeline is the SONIC {FPS} Hz timeline, got dataset={self.dataset_fps} "
                f"sonic={self.sonic_fps}"
            )
        if self.source_fps >= self.dataset_fps:
            raise ValueError("the source collection must be slower than the pack timeline")
        if self.trailing_invalid_rows != TAIL_ROWS:
            raise ValueError(f"exactly the {TAIL_ROWS} clamped tail rows are dropped")
        if self.horizon != HORIZON:
            raise ValueError(f"the GR00T horizon is {HORIZON} rows")
        if self.min_rows != HORIZON:
            raise ValueError("the shortest admissible episode is one horizon")
        if tuple(self.raw["state"]["projected_gravity"]) != PROJECTED_GRAVITY:
            raise ValueError("projected gravity is the synthetic upright [0, 0, -1]")
        if self.camera_width != 640 or self.camera_height != 480:
            raise ValueError("the ego view is 640x480")
        if self.camera_target_key in self.camera_excluded_keys:
            raise ValueError("the training camera cannot also be an excluded camera")
        if str(self.video["codec"]) != "h264" or str(self.video["pix_fmt"]) != "yuv420p":
            raise ValueError("the ego video is H.264 yuv420p")
        if str(self.video["resample"]) != "causal_last_frame_hold":
            raise ValueError(f"unsupported video policy {self.video['resample']!r}")
        if not 0.0 < self.repair_max_invalid_fraction < 1.0:
            raise ValueError("repair.max_invalid_fraction_per_channel must be in (0, 1)")
        if self.repair_invalid_abs_rad <= 0.0 or self.repair_max_gap_frames < 1:
            raise ValueError("the repair threshold and gap bound must be positive")
        if self.corpus_hand_tolerance_rad <= 0.0:
            raise ValueError("the corpus hand cross-check tolerance must be positive")
        if set(self.tasks) != set(self.collection_names):
            missing = sorted(set(self.collection_names) - set(self.tasks))
            extra = sorted(set(self.tasks) - set(self.collection_names))
            raise ValueError(f"task instructions must cover every collection: missing={missing} extra={extra}")
        for instruction in self.tasks.values():
            if not instruction or not instruction[0].isupper() or not instruction.endswith("."):
                raise ValueError(f"task instruction {instruction!r} is not a canonical sentence")
        if tuple(self.raw["state"]["model_key_order"]) != STATE_MODEL_KEYS:
            raise ValueError("state.model_key_order disagrees with the registered key order")
        if tuple(self.raw["action"]["model_key_order"]) != ACTION_MODEL_KEYS:
            raise ValueError("action.model_key_order disagrees with the registered key order")
        if self.stats_embodiment_tag != EMBODIMENT_TAG:
            raise ValueError(
                f"stats.embodiment_tag must be the pinned registry member {EMBODIMENT_TAG!r}, "
                f"got {self.stats_embodiment_tag!r} (the official tool rejects the lower-case value)"
            )
        if int(self.raw["state"]["dim"]) != STATE_DIM or int(self.raw["action"]["motion_token_dim"]) != MOTION_TOKEN_DIM:
            raise ValueError("the state and motion token widths are 43 and 64")
        for side in SIDES:
            declared_state = tuple(str(stem) for stem in self.raw["state"]["hand_order"][side])
            declared_action = tuple(str(stem) for stem in self.raw["action"]["hand_order"][side])
            if declared_state != STATE_HAND_ORDER[side] or declared_action != ACTION_HAND_ORDER[side]:
                raise ValueError(f"the declared {side} hand orders disagree with the official schema")
        assert_state_layout()


def _as_slice(value: Any, key: str) -> slice:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a [start, end] pair, got {value!r}")
    start, end = int(value[0]), int(value[1])
    if not 0 <= start < end:
        raise ValueError(f"{key} must be an increasing range, got {value!r}")
    return slice(start, end)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ConversionConfig:
    """Load and fully validate the conversion contract."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported conversion config schema in {path}")

    source, camera, fps = raw["source"], raw["camera"], raw["fps"]
    state, action, trim, repair = raw["state"], raw["action"], raw["trim"], raw["repair"]
    split, output, video, stats = raw["split"], raw["output"], raw["video"], raw["stats"]

    declared = tuple(str(name) for name in source["collections"])
    if len(set(declared)) != len(declared):
        raise ValueError("the collection list contains duplicates")
    exclusions = source.get("excluded_episodes") or {}
    undeclared = sorted(set(exclusions) - set(declared))
    if undeclared:
        raise ValueError(f"exclusions name undeclared collections: {undeclared}")

    layout = state["joint_layout"]
    # The slices are the contract; the stored order itself is asserted in
    # `assert_state_layout`, so a YAML that merely lists the blocks differently
    # still fails if any slice moved.
    if {key: _as_slice(value, key) for key, value in layout.items()} != STATE_BLOCKS:
        raise ValueError("state.joint_layout disagrees with the official 43D block layout")
    if str(raw["resample"]["method"]) != "linear_timestamp_interpolation":
        raise ValueError(f"unsupported 30->50 Hz policy {raw['resample']['method']!r}")
    if str(trim["method"]) != "drop_trailing_invalid_rows":
        raise ValueError(f"unsupported tail policy {trim['method']!r}")
    if str(repair["scope"]) != "measured_hand_channels" or str(repair["method"]) != (
        "linear_interpolation_over_valid_source_samples"
    ):
        raise ValueError("the repair policy is interpolation of measured hand channels only")
    if float(repair["invalid_abs_rad"]) != REPAIR_INVALID_ABS_RAD:
        raise ValueError(f"the corruption threshold is the declared {REPAIR_INVALID_ABS_RAD} rad")

    collections = tuple(
        CollectionContract(
            name=name,
            excluded=tuple(sorted(int(index) for index in (exclusions.get(name) or {}).get("episodes", ()))),
            instruction=str(raw["tasks"][name]),
            exclusion_reason=(exclusions.get(name) or {}).get("reason"),
        )
        for name in declared
    )
    config = ConversionConfig(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        name=str(raw["name"]),
        raw=raw,
        raw_root=Path(source["raw_root"]),
        sonic_root=Path(source["sonic_root"]),
        output_root=Path(output["root"]),
        train_repo=str(output["train_repo"]),
        val_repo=str(output["val_repo"]),
        split_name=str(split["name"]),
        split_manifest=Path(split["manifest"]),
        camera_source_key=str(camera["source_key"]),
        camera_target_key=str(camera["target_key"]),
        camera_excluded_keys=tuple(str(key) for key in camera.get("excluded_keys", ())),
        camera_width=int(camera["source_width"]),
        camera_height=int(camera["source_height"]),
        source_fps=int(fps["source"]),
        sonic_fps=int(fps["sonic"]),
        dataset_fps=int(fps["dataset"]),
        trailing_invalid_rows=int(trim["trailing_invalid_rows"]),
        min_rows=int(trim["min_rows"]),
        horizon=int(action["horizon"]),
        repair_invalid_abs_rad=float(repair["invalid_abs_rad"]),
        repair_max_invalid_fraction=float(repair["max_invalid_fraction_per_channel"]),
        repair_max_gap_frames=int(repair["max_gap_source_frames"]),
        repair_fail_on_action=bool(repair["fail_on_action_hand_corruption"]),
        corpus_hand_tolerance_rad=float(action["corpus_hand_tolerance_rad"]),
        video=dict(video),
        stats_official_tool=str(stats["official_tool"]),
        stats_own_file=str(stats["own_file"]),
        stats_official_file=str(stats["official_file"]),
        stats_embodiment_tag=str(stats["embodiment_tag"]),
        collections=collections,
        tasks={str(key): str(value) for key, value in raw["tasks"].items()},
    )
    config.assert_contract()
    return config


def official_stats_command(config: ConversionConfig, split_dir: Path, python: str = "python3") -> list[str]:
    """Argv of the official statistics tool for one produced split.

    The loader-facing ``meta/stats.json`` and ``meta/relative_stats.json`` are
    produced by the pinned Isaac-GR00T tool, not by this pipeline; this is the
    command to run them, in the training image where ``/opt/src/isaac-groot`` is
    mounted.

    The tool's interface is ``tyro.cli(main)`` over
    ``main(dataset_path, embodiment_tag, modality_config_path=None)`` at the
    pinned commit, whose own usage line is
    ``python gr00t/data/stats.py --dataset-path <path> --embodiment-tag <tag>``.
    The tag is a plain ``Enum`` member, so the accepted spelling is the member
    name ``UNITREE_G1_SONIC``; the lower-case value ``unitree_g1_sonic`` is
    rejected as an invalid choice. Omitting the flag is not an option: the tool
    resolves the modality config from the tag before it writes anything.
    """
    return [
        python,
        config.stats_official_tool,
        "--dataset-path",
        str(split_dir),
        "--embodiment-tag",
        config.stats_embodiment_tag,
    ]


__all__ = [
    "ACTION_DIM",
    "ACTION_HAND_FIELDS",
    "ACTION_HAND_ORDER",
    "ACTION_MODEL_KEYS",
    "CANONICAL_ACTION_HAND_NAMES",
    "CANONICAL_STATE_NAMES",
    "CORPUS_HAND_ORDER",
    "CollectionContract",
    "ConversionConfig",
    "DEFAULT_CONFIG_PATH",
    "EGO_VIEW_KEY",
    "EMBODIMENT_TAG",
    "FPS",
    "GRAVITY_FIELD",
    "HAND_CHANNELS",
    "HAND_DIM",
    "HORIZON",
    "INSTRUCTION_KEY",
    "INSTRUCTION_ORIGINAL_KEY",
    "LEFT_HAND_FIELD",
    "MEASURED_STATE_FIELDS",
    "MODEL_STATE_DIM",
    "MOTION_TOKEN_DIM",
    "MOTION_TOKEN_FIELD",
    "PROJECTED_GRAVITY",
    "REPAIR_INVALID_ABS_RAD",
    "RIGHT_HAND_FIELD",
    "SOURCE_STATE_DIM",
    "STATE_BLOCKS",
    "STATE_DIM",
    "STATE_FIELD",
    "STATE_HAND_ORDER",
    "STATE_MODEL_KEYS",
    "STATE_STORAGE_KEYS",
    "TAIL_ROWS",
    "assert_source_names",
    "assert_state_layout",
    "corpus_hand_permutation",
    "load_config",
    "official_stats_command",
    "source_action_hand_permutation",
    "source_state_permutation",
    "standing_lower_body",
]
