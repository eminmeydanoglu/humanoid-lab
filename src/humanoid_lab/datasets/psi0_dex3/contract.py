"""Gate 0 contract for the Psi0 Unitree Dex3 / SONIC v1.1 conversion.

The decisions frozen here are the ones a converter must never make again: the
source collections and their exclusions, the single left camera, the 30 Hz
dataset timeline, the canonical 43D state order and the 13 canonical task
instructions. Every name lookup is fail-closed: a channel is placed by name,
and a missing or duplicated name raises instead of silently shifting the rest
of a vector.

The sources are frozen inputs. Nothing in this package writes under them.

The field names and widths of the produced pack are not declared here: they are
the loader-facing interop contract and live once, in
``humanoid_lab.datasets.psi0.contract``. This module imports them and asserts
that the YAML agrees, so the converter cannot drift away from what the Psi0
loader reads.
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
from humanoid_lab.datasets.psi0.contract import (
    ACTION_CHUNK,
    ACTION_MODEL_DIM,
    ANCHOR_MASK_KEY,
    BODY_TOKEN_KEY,
    BODY_TOKEN_DIM,
    DATASET_DIR,
    FPS,
    HAND_DIM,
    HAND_KEY,
    IMAGE_KEY,
    INSTRUCTION_KEY,
    MASK_KEY,
    STATE_DIM,
    STATE_KEY,
    STATE_MODEL_DIM,
    TRAIN_REPO_ID,
    VAL_REPO_ID,
)
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import UNITREE_ACTION_NAMES

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = ROOT / "configs/datasets/psi0/unitree_dex3_sonic_v1.yaml"

#: Source `observation.state`/`action` width of the raw Unitree Dex3 collection.
SOURCE_STATE_DIM = 28
#: Unused neck padding the Psi0 model contract appends; both state and action
#: carry the same two channels, so the loader contract declares them once.
NECK_PADDING_DIM = STATE_MODEL_DIM - STATE_DIM


def _hand_names(side: str) -> tuple[str, ...]:
    order = LEFT_HAND_JOINT_ORDER if side == "left" else RIGHT_HAND_JOINT_ORDER
    return tuple(f"{side}_hand_{name}" for name in order)


#: Canonical 43D state order: legs+waist (15), arms (14), Dex3 hands (14).
#:
#: This is the order Psi0's SONIC packer produces
#: (``scripts/data/raw_sonic_to_psi_lerobot.py::pack_psi_state``): the 29 body
#: qpos channels in hardware order followed by the 14 hand channels. The three
#: contiguous groups are addressable as :data:`LEGS_WAIST_SLICE`,
#: :data:`ARMS_SLICE` and :data:`HANDS_SLICE`.
CANONICAL_STATE_NAMES: tuple[str, ...] = BODY_JOINT_ORDER + _hand_names("left") + _hand_names("right")

LEGS_WAIST_SLICE = slice(0, 15)
ARMS_SLICE = slice(15, 29)
HANDS_SLICE = slice(29, 43)

#: Canonical 14D hand action order, which is also the raw Unitree field order.
CANONICAL_HAND_NAMES: tuple[str, ...] = _hand_names("left") + _hand_names("right")


def _unitree_field_name(canonical_name: str) -> str:
    """``left_shoulder_pitch_joint`` -> ``kLeftShoulderPitch``.

    The raw collection names every channel in camel case with a ``k`` prefix;
    this is the inverse of the hand field map in
    :mod:`humanoid_lab.datasets.sonic.adapters.unitree_dex3`.
    """
    parts = canonical_name.replace("_joint", "").split("_")
    return "k" + "".join(part.title() for part in parts)


#: Raw source field for each measured state channel, in canonical order 15:43.
SOURCE_ARM_HAND_FIELDS: tuple[str, ...] = tuple(
    _unitree_field_name(name) for name in CANONICAL_STATE_NAMES[15:43]
)


def standing_lower_body() -> np.ndarray:
    """The 15D synthetic legs+waist proxy in canonical order.

    These are the official SONIC deployment standing targets, not a measured
    qpos trajectory: the collection records no leg or waist observation. They
    are constant for every frame of every episode, so normalization maps them
    to zero and the model never learns lower-body motion from invented data.
    """
    return np.asarray(
        [DEFAULT_STANDING_POSE_RAD[name] for name in CANONICAL_STATE_NAMES[:15]],
        dtype=np.float64,
    )


def assert_state_layout() -> None:
    """The canonical 43D order, checked by name rather than by position."""
    if len(CANONICAL_STATE_NAMES) != 43:
        raise ValueError(f"canonical state must be 43D, got {len(CANONICAL_STATE_NAMES)}")
    if len(set(CANONICAL_STATE_NAMES)) != 43:
        raise ValueError("canonical state names contain duplicates")
    if CANONICAL_STATE_NAMES[0] != "left_hip_pitch_joint":
        raise ValueError(f"first state channel must be a leg joint, got {CANONICAL_STATE_NAMES[0]!r}")
    if CANONICAL_STATE_NAMES[ARMS_SLICE.start] != "left_shoulder_pitch_joint":
        raise ValueError("arms must start at index 15")
    if CANONICAL_STATE_NAMES[HANDS_SLICE.start] != "left_hand_thumb_0_joint":
        raise ValueError(
            f"hands must start at index {HANDS_SLICE.start}, "
            f"got {CANONICAL_STATE_NAMES[HANDS_SLICE.start]!r}"
        )
    if tuple(CANONICAL_STATE_NAMES[:15]) != tuple(BODY_JOINT_ORDER[:15]):
        raise ValueError("legs+waist must be the first 15 channels in hardware order")
    if len(set(CANONICAL_HAND_NAMES)) != HAND_DIM:
        raise ValueError("canonical hand action must be 14 distinct joints")
    unknown = sorted(set(SOURCE_ARM_HAND_FIELDS) - set(UNITREE_ACTION_NAMES))
    if unknown:
        raise ValueError(f"raw field map names channels the source does not declare: {unknown}")
    if len(SOURCE_ARM_HAND_FIELDS) != SOURCE_STATE_DIM or SOURCE_ARM_HAND_FIELDS[0] != "kLeftShoulderPitch":
        raise ValueError("the measured 28D block must be the raw Unitree arm and hand field order")


def assert_source_state_names(names: tuple[str, ...]) -> None:
    """Fail closed unless the raw 28D state is exactly the declared layout."""
    if tuple(names) != UNITREE_ACTION_NAMES:
        raise ValueError("raw observation.state is not the declared 28D Unitree Dex3 layout")


def source_state_permutation(names: tuple[str, ...]) -> tuple[int, ...]:
    """Indices lifting the raw 28D measured state into canonical channels 15:43.

    The current source order already matches, but the converter derives the
    permutation from the declared names so a re-ordered upstream release fails
    loudly instead of filling the wrong joint.
    """
    assert_source_state_names(names)
    positions = {name: index for index, name in enumerate(names)}
    return tuple(positions[name] for name in SOURCE_ARM_HAND_FIELDS)


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
    camera_source_key: str
    camera_target_key: str
    camera_excluded_keys: tuple[str, ...]
    source_fps: int
    sonic_fps: int
    dataset_fps: int
    resample_method: str
    max_timestamp_error_s: float
    tail_timestamp_error_s: float
    action_field: str
    body_token_field: str
    state_field: str
    state_semantics: dict[str, str]
    video: dict[str, Any]
    split_seed: int
    val_fraction: float
    min_val_per_collection: int
    collections: tuple[CollectionContract, ...]
    tasks: dict[str, str]

    @property
    def collection_names(self) -> tuple[str, ...]:
        return tuple(collection.name for collection in self.collections)

    @property
    def action_chunk_size(self) -> int:
        return int(self.raw["model_contract"]["action_chunk_size"])

    @property
    def camera_height(self) -> int:
        return int(self.raw["camera"]["source_height"])

    @property
    def camera_width(self) -> int:
        return int(self.raw["camera"]["source_width"])

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
        if NECK_PADDING_DIM != ACTION_MODEL_DIM - (BODY_TOKEN_DIM + HAND_DIM):
            raise ValueError("the loader contract pads the action by a different width than the state")
        if self.dataset_fps != self.source_fps:
            raise ValueError(
                f"dataset fps {self.dataset_fps} must equal the source fps {self.source_fps}: "
                "the camera and the measured state stay on the source timeline"
            )
        if self.sonic_fps <= self.dataset_fps:
            raise ValueError("the SONIC corpus must be faster than the dataset timeline")
        if self.resample_method != "nearest_timestamp":
            raise ValueError(f"unsupported 50->30 Hz policy {self.resample_method!r}")
        if self.action_field == self.body_token_field:
            raise ValueError("the hand action and the body token must be separate fields")
        if self.camera_target_key in self.camera_excluded_keys:
            raise ValueError("the training camera cannot also be an excluded camera")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction {self.val_fraction} must be in (0, 1)")
        if not 0.0 < self.max_timestamp_error_s < self.tail_timestamp_error_s:
            raise ValueError(
                "the supervised 50->30 Hz bound must be positive and below the tail bound"
            )
        if set(self.tasks) != set(self.collection_names):
            missing = sorted(set(self.collection_names) - set(self.tasks))
            extra = sorted(set(self.tasks) - set(self.collection_names))
            raise ValueError(f"task instructions must cover every collection: missing={missing} extra={extra}")
        for instruction in self.tasks.values():
            if not instruction or not instruction[0].isupper() or not instruction.endswith("."):
                raise ValueError(f"task instruction {instruction!r} is not a canonical sentence")
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

    source = raw["source"]
    camera, fps = raw["camera"], raw["fps"]
    action, state, split, output = raw["action"], raw["state"], raw["split"], raw["output"]

    declared = tuple(str(name) for name in source["collections"])
    if len(set(declared)) != len(declared):
        raise ValueError("the collection list contains duplicates")
    exclusions = source.get("excluded_episodes") or {}
    undeclared = sorted(set(exclusions) - set(declared))
    if undeclared:
        raise ValueError(f"exclusions name undeclared collections: {undeclared}")

    body_token_slice = _as_slice(action["source_body_token_slice"], "action.source_body_token_slice")
    hand_slice = _as_slice(action["source_hand_slice"], "action.source_hand_slice")
    if body_token_slice != slice(0, BODY_TOKEN_DIM):
        raise ValueError("the body token is the first 64 channels of the frozen 78D action")
    if hand_slice != slice(BODY_TOKEN_DIM, BODY_TOKEN_DIM + HAND_DIM):
        raise ValueError("the hand action is the last 14 channels of the frozen 78D action")
    layout = state["joint_layout"]
    if (
        _as_slice(layout["legs_and_waist"], "legs_and_waist") != LEGS_WAIST_SLICE
        or _as_slice(layout["arms"], "arms") != ARMS_SLICE
        or _as_slice(layout["hands"], "hands") != HANDS_SLICE
    ):
        raise ValueError("state.joint_layout disagrees with the canonical 43D order")
    if int(state["dim"]) != len(CANONICAL_STATE_NAMES):
        raise ValueError("state.dim must be 43")
    if int(action["body_token_dim"]) != BODY_TOKEN_DIM or int(action["hand_dim"]) != HAND_DIM:
        raise ValueError("the frozen action must split into a 64D body token and a 14D hand action")
    model = raw["model_contract"]
    if int(model["state_dim"]) != STATE_MODEL_DIM or int(model["action_dim"]) != ACTION_MODEL_DIM:
        raise ValueError("model_contract must be 45D state / 80D action")
    if int(model["action_chunk_size"]) != int(model["action_exec_horizon"]):
        raise ValueError("the strict validity horizon is the action chunk size")
    # The pack is consumed by the Psi0 loader, whose field names and widths are
    # declared once in `humanoid_lab.datasets.psi0.contract`; a YAML that
    # disagrees would produce a pack the loader cannot read.
    for label, declared_value, canonical in (
        ("name", str(raw["name"]), DATASET_DIR),
        ("fps.dataset", int(fps["dataset"]), int(FPS)),
        ("model_contract.action_chunk_size", int(model["action_chunk_size"]), ACTION_CHUNK),
        ("state.field", str(state["field"]), STATE_KEY),
        ("state.dim", int(state["dim"]), STATE_DIM),
        ("action.field", str(action["field"]), HAND_KEY),
        ("action.body_token_field", str(action["body_token_field"]), BODY_TOKEN_KEY),
        ("camera.target_key", str(camera["target_key"]), IMAGE_KEY),
        ("output.train_repo", str(output["train_repo"]), TRAIN_REPO_ID),
        ("output.val_repo", str(output["val_repo"]), VAL_REPO_ID),
    ):
        if declared_value != canonical:
            raise ValueError(
                f"{label} is {declared_value!r} but the Psi0 loader contract requires {canonical!r}"
            )

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
        camera_source_key=str(camera["source_key"]),
        camera_target_key=str(camera["target_key"]),
        camera_excluded_keys=tuple(str(key) for key in camera.get("excluded_keys", ())),
        source_fps=int(fps["source"]),
        sonic_fps=int(fps["sonic"]),
        dataset_fps=int(fps["dataset"]),
        resample_method=str(raw["resample"]["method"]),
        max_timestamp_error_s=float(raw["resample"]["max_timestamp_error_s"]),
        tail_timestamp_error_s=float(raw["resample"]["tail_timestamp_error_s"]),
        action_field=str(action["field"]),
        body_token_field=str(action["body_token_field"]),
        state_field=str(state["field"]),
        state_semantics={str(k): str(v) for k, v in state["semantics"].items()},
        video=dict(raw["video"]),
        split_seed=int(split["seed"]),
        val_fraction=float(split["val_fraction"]),
        min_val_per_collection=int(split.get("min_val_per_collection", 1)),
        collections=collections,
        tasks={str(key): str(value) for key, value in raw["tasks"].items()},
    )
    config.assert_contract()
    return config
