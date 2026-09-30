"""Fail-closed, independent validation of a produced GR00T Unitree Dex3 / SONIC pack.

The validator re-derives every episode from the frozen raw collection and the
frozen 50 Hz SONIC corpus using its own literal name tables, its own corruption
scan and its own interpolation, then compares the result field by field against
what is on disk. It shares no mapping, ordering or repair code with the
converter, so a wrong channel order, a shifted timeline, a retained clamped row
or a truncated clip is caught rather than confirmed.

Checked here: the Psi0 split identity, the declared schema (info/modality), one
row per retained corpus sample, the exact timestamp and token copy, the
resampled state, the rebuilt hands, the repaired-sample ledger, the ego clip
(row count, geometry, frame rate, codec, and that its first picture really is the
segment's), and -- when the training image is mounted -- the pre-registered
embodiment entry, read from ``MODALITY_CONFIGS`` in the pinned registry source
with :mod:`ast`, plus the LeRobot loader when it is installed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import GRAVITY_FIELD, ConversionConfig, standing_lower_body
from .convert import read_raw_episode, sonic_episode_path
from .modality import assert_modality_payload

#: Restated literals: a change in the contract module cannot relax the validator.
EXPECTED_FPS = 50
EXPECTED_TAIL_ROWS = 45
EXPECTED_HORIZON = 40
EXPECTED_STATE_DIM = 43
EXPECTED_MODEL_STATE_DIM = 46
EXPECTED_TOKEN_DIM = 64
EXPECTED_HAND_DIM = 7
EXPECTED_ACTION_DIM = 78
EXPECTED_REPAIR_ABS_RAD = 3.0
EXPECTED_CAMERA = (640, 480)
#: Seconds; a 50 Hz row is 0.02 s, and the stored column is float32.
TIMESTAMP_TOLERANCE_S = 1e-4
#: Grey levels (0-255): a re-encode of the same picture differs by a few levels,
#: a clip holding another segment differs by tens. ``MARGIN`` is how much better
#: the probe frame must match before the clip is judged to hold the wrong part of
#: the recording; it keeps a still scene, where both matches are near zero, from
#: failing on encoder noise alone.
VIDEO_ALIGNMENT_MAX_DIFF = 10.0
VIDEO_ALIGNMENT_MARGIN = 1.0

CAMERA_KEY = "observation.images.ego_view"

#: Official state hand order: index, middle, thumb, both sides.
STATE_HAND_ORDER = {
    "left": (
        "index_0_joint", "index_1_joint", "middle_0_joint",
        "middle_1_joint", "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
    ),
    "right": (
        "index_0_joint", "index_1_joint", "middle_0_joint",
        "middle_1_joint", "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
    ),
}
#: Official actuated action hand order: thumb, index, middle, both sides.
ACTUATED_HAND_ORDER = (
    "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
    "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint",
)
#: Per-side motor order the SONIC corpus stores in ``action[:, 64:78]``.
CORPUS_HAND_ORDER = {
    "left": (
        "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
        "middle_0_joint", "middle_1_joint", "index_0_joint", "index_1_joint",
    ),
    "right": (
        "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
        "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint",
    ),
}

#: The 29 body channels, restated literally.
_BODY_CHANNELS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
#: The 14 measured arm fields, restated literally in official order.
_ARM_SOURCE_FIELDS = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll",
    "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
)
#: Raw Unitree field of every hand joint.
_HAND_FIELDS = {
    "left": {
        "thumb_0_joint": "kLeftHandThumb0", "thumb_1_joint": "kLeftHandThumb1", "thumb_2_joint": "kLeftHandThumb2",
        "index_0_joint": "kLeftHandIndex0", "index_1_joint": "kLeftHandIndex1",
        "middle_0_joint": "kLeftHandMiddle0", "middle_1_joint": "kLeftHandMiddle1",
    },
    "right": {
        "thumb_0_joint": "kRightHandThumb0", "thumb_1_joint": "kRightHandThumb1", "thumb_2_joint": "kRightHandThumb2",
        "index_0_joint": "kRightHandIndex0", "index_1_joint": "kRightHandIndex1",
        "middle_0_joint": "kRightHandMiddle0", "middle_1_joint": "kRightHandMiddle1",
    },
}
#: The 43 official channel names, in the stored order of the official exporter:
#: lower body, then each arm followed by its hand.
STATE_CHANNELS: tuple[str, ...] = (
    _BODY_CHANNELS[:22]
    + tuple(f"left_hand_{stem}" for stem in STATE_HAND_ORDER["left"])
    + _BODY_CHANNELS[22:29]
    + tuple(f"right_hand_{stem}" for stem in STATE_HAND_ORDER["right"])
)
#: The 28 raw measured fields, in the same stored order (left arm, left hand,
#: right arm, right hand); the lower body is synthetic and has no source field.
MEASURED_SOURCE_FIELDS: tuple[str, ...] = (
    _ARM_SOURCE_FIELDS[:7]
    + tuple(_HAND_FIELDS["left"][stem] for stem in STATE_HAND_ORDER["left"])
    + _ARM_SOURCE_FIELDS[7:]
    + tuple(_HAND_FIELDS["right"][stem] for stem in STATE_HAND_ORDER["right"])
)
#: The 14 raw desired-action hand fields, in the official actuated order.
ACTION_SOURCE_FIELDS: tuple[str, ...] = tuple(
    _HAND_FIELDS[side][stem] for side in ("left", "right") for stem in ACTUATED_HAND_ORDER
)
#: The action hand channel names the pack must declare, in actuated order.
EXPECTED_LEFT_HAND_NAMES: tuple[str, ...] = tuple(f"left_hand_{stem}" for stem in ACTUATED_HAND_ORDER)
EXPECTED_RIGHT_HAND_NAMES: tuple[str, ...] = tuple(f"right_hand_{stem}" for stem in ACTUATED_HAND_ORDER)

STATE_BLOCKS = {
    "left_leg": slice(0, 6), "right_leg": slice(6, 12), "waist": slice(12, 15), "left_arm": slice(15, 22),
    "left_hand": slice(22, 29), "right_arm": slice(29, 36), "right_hand": slice(36, 43),
}
#: Each hand as ``(state channel slice, measured source slice)``.
_HAND_BLOCKS = (
    (slice(22, 29), slice(7, 14)),
    (slice(36, 43), slice(21, 28)),
)
#: The modality boxes, restated literally so a change in the writer cannot pass.
#: Keys are in the registered order; the slices are the stored order, which the
#: loader is free to reorder.
EXPECTED_MODALITY: dict[str, dict[str, Any]] = {
    "state": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "right_arm": {"start": 29, "end": 36},
        "left_hand": {"start": 22, "end": 29},
        "right_hand": {"start": 36, "end": 43},
        "projected_gravity": {"start": 0, "end": 3, "original_key": GRAVITY_FIELD},
    },
    "action": {
        "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
        "left_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.left_hand_joints"},
        "right_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.right_hand_joints"},
    },
    "video": {"ego_view": {"original_key": CAMERA_KEY}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
#: The registered key order, restated so the box above is checked as a whole.
EXPECTED_STATE_KEY_ORDER = (
    "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand", "projected_gravity",
)
#: The pinned member name the official statistics tool accepts.
EXPECTED_EMBODIMENT_TAG = "UNITREE_G1_SONIC"

#: The pre-registered registry of embodiment configs, as it is written in the
#: pinned checkout: a module-level ``MODALITY_CONFIGS`` dict keyed by the *tag
#: value* (``EmbodimentTag.UNITREE_G1_SONIC.value``), not by the member name the
#: statistics tool takes on its command line.
REGISTRY_RELATIVE_PATH = "gr00t/configs/data/embodiment_configs.py"
REGISTRY_CONFIG_VARIABLE = "MODALITY_CONFIGS"
REGISTRY_KEY = "unitree_g1_sonic"
#: Modality keys of the pinned ``unitree_g1_sonic`` entry, restated literally.
REGISTRY_STATE_KEYS = (
    "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand", "projected_gravity",
)
REGISTRY_ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
REGISTRY_VIDEO_KEYS = ("ego_view",)
REGISTRY_LANGUAGE_KEYS = ("annotation.human.task_description",)
#: Every modality that observes the current frame only.
REGISTRY_SINGLE_FRAME_MODALITIES = ("video", "state", "language")


def _literal(node: ast.AST) -> Any:
    """Evaluate a registry literal without importing the package.

    Only the shapes the pinned registry is written in are understood: literals,
    and ``list``/``tuple``/``range`` calls over them (``delta_indices=[0]``,
    ``delta_indices=list(range(40))``). Anything else is refused rather than
    guessed at, so a registry that computes its keys elsewhere fails the check
    instead of passing it silently.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        values = [_literal(argument) for argument in node.args]
        if name == "range":
            return list(range(*values))
        if name in ("list", "tuple") and len(values) == 1:
            return list(values[0]) if name == "list" else tuple(values[0])
    raise ValidationError(f"the pinned registry uses an expression this check does not read: {ast.dump(node)[:120]}")


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _modality_configs_dict(tree: ast.Module) -> ast.Dict:
    for statement in tree.body:
        targets = statement.targets if isinstance(statement, ast.Assign) else (
            [statement.target] if isinstance(statement, ast.AnnAssign) else []
        )
        if not any(isinstance(target, ast.Name) and target.id == REGISTRY_CONFIG_VARIABLE for target in targets):
            continue
        if not isinstance(statement.value, ast.Dict):
            raise ValidationError(f"{REGISTRY_CONFIG_VARIABLE} is not a dict literal in the pinned registry")
        return statement.value
    raise ValidationError(f"the pinned registry declares no {REGISTRY_CONFIG_VARIABLE} dict literal")


def parse_registry_entry(source: str, key: str = REGISTRY_KEY) -> dict[str, Any]:
    """Read the pinned ``MODALITY_CONFIGS[key]`` entry from its source text.

    Returns the modality keys and delta indices of the entry, with every value
    read from the AST, so the check runs under an interpreter that cannot import
    ``gr00t`` (its import pulls torch and the model stack) and without executing
    the file.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValidationError(f"the pinned registry is not parsable Python: {error}") from error
    configs = _modality_configs_dict(tree)
    available = [entry.value for entry in configs.keys if isinstance(entry, ast.Constant)]
    for name, entry in zip(configs.keys, configs.values, strict=True):
        if not (isinstance(name, ast.Constant) and name.value == key):
            continue
        if not isinstance(entry, ast.Dict):
            raise ValidationError(f"{REGISTRY_CONFIG_VARIABLE}[{key!r}] is not a dict literal")
        modalities: dict[str, Any] = {}
        for modality, call in zip(entry.keys, entry.values, strict=True):
            if not (isinstance(modality, ast.Constant) and isinstance(modality.value, str) and isinstance(call, ast.Call)):
                continue
            keys_node = _keyword(call, "modality_keys")
            delta_node = _keyword(call, "delta_indices")
            modalities[modality.value] = {
                "keys": tuple(_literal(keys_node)) if keys_node is not None else None,
                "delta_indices": _literal(delta_node) if delta_node is not None else None,
            }
            if modality.value == "action":
                configs_node = _keyword(call, "action_configs")
                modalities["action"]["action_configs"] = (
                    len(configs_node.elts) if isinstance(configs_node, (ast.List, ast.Tuple)) else None
                )
                representations: list[str | None] = []
                if isinstance(configs_node, (ast.List, ast.Tuple)):
                    for element in configs_node.elts:
                        rep = _keyword(element, "rep") if isinstance(element, ast.Call) else None
                        representations.append(rep.attr if isinstance(rep, ast.Attribute) else None)
                modalities["action"]["action_representations"] = (
                    tuple(representations) if representations else None
                )
        return modalities
    raise ValidationError(
        f"the pinned registry has no {key!r} entry; it registers {sorted(str(name) for name in available)}"
    )


def assert_registry_matches_contract(entry: dict[str, Any], horizon: int) -> str:
    """Fail closed unless the pinned entry describes exactly this pack."""
    if entry.get("state", {}).get("keys") != REGISTRY_STATE_KEYS:
        raise ValidationError(
            f"the registry state keys are {entry.get('state', {}).get('keys')}, expected {REGISTRY_STATE_KEYS}"
        )
    if entry.get("action", {}).get("keys") != REGISTRY_ACTION_KEYS:
        raise ValidationError(
            f"the registry action keys are {entry.get('action', {}).get('keys')}, expected {REGISTRY_ACTION_KEYS}"
        )
    if entry.get("video", {}).get("keys") != REGISTRY_VIDEO_KEYS:
        raise ValidationError(
            f"the registry video keys are {entry.get('video', {}).get('keys')}, expected {REGISTRY_VIDEO_KEYS}"
        )
    if entry.get("language", {}).get("keys") != REGISTRY_LANGUAGE_KEYS:
        raise ValidationError(
            f"the registry language keys are {entry.get('language', {}).get('keys')}, expected {REGISTRY_LANGUAGE_KEYS}"
        )
    expected_delta = list(range(horizon))
    if entry.get("action", {}).get("delta_indices") != expected_delta:
        raise ValidationError(
            f"the registry action delta indices are {entry.get('action', {}).get('delta_indices')}, "
            f"expected the {horizon}-row horizon {expected_delta[0]}..{expected_delta[-1]}"
        )
    for modality in REGISTRY_SINGLE_FRAME_MODALITIES:
        if entry.get(modality, {}).get("delta_indices") != [0]:
            raise ValidationError(
                f"the registry observes {modality} at {entry.get(modality, {}).get('delta_indices')}; "
                "this pack carries the current frame only"
            )
    representations = entry.get("action", {}).get("action_representations")
    if representations is not None and set(representations) != {"ABSOLUTE"}:
        raise ValidationError(f"the registry action representations are {representations}, expected all ABSOLUTE")
    return (
        f"{len(REGISTRY_STATE_KEYS)} state / {len(REGISTRY_ACTION_KEYS)} action keys, "
        f"action horizon {horizon}, all ABSOLUTE"
    )

REQUIRED_META = (
    "meta/info.json", "meta/modality.json", "meta/tasks.jsonl", "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)


class ValidationError(RuntimeError):
    """Raised when a check cannot even be evaluated."""


def _positions(names: tuple[str, ...]) -> dict[str, int]:
    positions = {name: index for index, name in enumerate(names)}
    if len(positions) != len(names):
        raise ValidationError("the source declares duplicate joint names")
    return positions


def _interpolate(source: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Column-wise linear interpolation onto the target timeline."""
    source = np.asarray(source, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if float(target.min()) < float(source[0]) or float(target.max()) > float(source[-1]):
        raise ValidationError("the pack timeline leaves the source span; the pack cannot be re-derived")
    return np.stack([np.interp(target, source, values[:, index]) for index in range(values.shape[1])], axis=1)


def invalid_mask(samples: np.ndarray) -> np.ndarray:
    """The declared gross-corruption predicate, restated independently."""
    values = np.asarray(samples, dtype=np.float64)
    return ~np.isfinite(values) | (np.abs(values) > EXPECTED_REPAIR_ABS_RAD)


def repaired_measured(state: np.ndarray, names: tuple[str, ...], label: str) -> tuple[np.ndarray, dict[str, int]]:
    """Independent re-implementation of the measured-hand repair, with its ledger."""
    repaired = np.asarray(state, dtype=np.float64).copy()
    positions = _positions(names)
    ledger: dict[str, int] = {}
    for state_slice, source_slice in _HAND_BLOCKS:
        for channel, field in zip(STATE_CHANNELS[state_slice], MEASURED_SOURCE_FIELDS[source_slice], strict=True):
            column = positions[field]
            invalid = invalid_mask(repaired[:, column])
            if not bool(invalid.any()):
                continue
            valid = np.flatnonzero(~invalid)
            if valid.size == 0 or bool(invalid[0]) or bool(invalid[-1]):
                raise ValidationError(f"{label}: {field} cannot be repaired by interpolation as the pack claims")
            repaired[:, column] = np.interp(np.arange(repaired.shape[0]), valid, repaired[valid, column])
            ledger[channel] = int(invalid.sum())
    return repaired, ledger


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"missing metadata file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_parquet(path: Path, columns: list[str]) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return {name: table.column(name).to_pylist() for name in columns}


def _decode_video(path: Path) -> dict[str, Any]:
    """Decode the whole clip: the frame count is counted, not read from metadata."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return {
            "frames": frames,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": float(stream.average_rate) if stream.average_rate else 0.0,
            "codec": str(stream.codec_context.name),
            "pix_fmt": str(stream.codec_context.format.name) if stream.codec_context.format else "",
        }


def _gray(frame: Any) -> np.ndarray:
    return np.asarray(frame.to_ndarray(format="gray"), dtype=np.float64)


def first_frame_content(path: Path, *, seek_s: float | None = None) -> np.ndarray | None:
    """The first picture of a clip, or the first one at or after ``seek_s``."""
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if seek_s:
                container.seek(int(seek_s * av.time_base), backward=True)
            for frame in container.decode(stream):
                if seek_s is None or (frame.time or 0.0) >= seek_s - 1e-3:
                    return _gray(frame)
    except Exception:
        return None
    return None


def video_alignment(
    episode_clip: Path,
    source_clip: Path,
    source_start_s: float,
    *,
    probe_offset_s: float = 0.5,
) -> dict[str, float] | None:
    """Compare a clip's first frame with the source, at the episode start and later.

    The first retained row covers the segment start, so the clip's first picture
    must be the source picture at ``source_start_s`` and, by construction, must
    resemble the source at ``source_start_s + probe_offset_s`` less than it
    resembles the start. A clip holding another part of the recording matches the
    probe better, which the caller turns into a failure; a re-encode of the right
    picture is a few grey levels off and matches the start better.
    """
    produced = first_frame_content(episode_clip)
    expected = first_frame_content(source_clip, seek_s=source_start_s)
    later = first_frame_content(source_clip, seek_s=source_start_s + probe_offset_s)
    if produced is None or expected is None or later is None:
        return None
    if produced.shape != expected.shape or produced.shape != later.shape:
        return None
    return {
        "expected": float(np.abs(produced - expected).mean()),
        "probe": float(np.abs(produced - later).mean()),
    }


class DatasetValidator:
    """Collects pass/fail checks for one produced dataset root."""

    def __init__(
        self,
        config: ConversionConfig,
        root: Path,
        manifest: dict[str, Any],
        *,
        require_complete: bool = False,
        groot_source: Path | None = None,
    ):
        self.config = config
        self.root = Path(root)
        self.manifest = manifest
        self.require_complete = require_complete
        self.groot_source = Path(groot_source) if groot_source else Path("/opt/src/isaac-groot")
        self.checks: list[dict[str, str]] = []
        self.episodes_checked = 0
        self.frames_checked = 0
        self.repaired_samples = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        return ok

    def skip(self, name: str, detail: str) -> None:
        self.checks.append({"check": name, "status": "SKIP", "detail": detail})

    @property
    def status(self) -> str:
        return "PASS" if all(entry["status"] in ("PASS", "SKIP") for entry in self.checks) else "FAIL"

    # -- contract ----------------------------------------------------------
    def check_contract_literals(self) -> None:
        """The validator's own restatement of the frozen numbers."""
        self.check("contract.fps", self.config.dataset_fps == EXPECTED_FPS, str(self.config.dataset_fps))
        self.check(
            "contract.tail_rows",
            self.config.trailing_invalid_rows == EXPECTED_TAIL_ROWS,
            str(self.config.trailing_invalid_rows),
        )
        self.check("contract.horizon", self.config.horizon == EXPECTED_HORIZON, str(self.config.horizon))
        self.check(
            "contract.repair_threshold",
            self.config.repair_invalid_abs_rad == EXPECTED_REPAIR_ABS_RAD,
            str(self.config.repair_invalid_abs_rad),
        )
        self.check(
            "contract.camera",
            (self.config.camera_width, self.config.camera_height) == EXPECTED_CAMERA,
            f"{self.config.camera_width}x{self.config.camera_height}",
        )
        # The trap this pack exists to avoid: the corpus left block is not the
        # official actuated order, so the action hands are rebuilt, never copied.
        self.check(
            "contract.corpus_left_differs_from_actuated",
            CORPUS_HAND_ORDER["left"] != ACTUATED_HAND_ORDER,
        )
        self.check(
            "contract.state_hand_order_differs_from_actuated",
            STATE_HAND_ORDER["left"] != ACTUATED_HAND_ORDER,
        )

    # -- the split ---------------------------------------------------------
    def check_split(self) -> None:
        manifest = self.manifest
        self.check("split.name", str(manifest.get("name")) == self.config.split_name, str(manifest.get("name")))
        declared: dict[str, set[tuple[str, int]]] = {"train": set(), "val": set()}
        for name in self.config.collection_names:
            entry = manifest["collections"][name]
            self.check(f"split.{name}.instruction", str(entry.get("instruction", "")) == self.config.tasks[name])
            self.check(
                f"split.{name}.excluded_match_contract",
                tuple(sorted(int(index) for index in entry.get("excluded", ())))
                == tuple(sorted(self.config.collection(name).excluded)),
            )
            for split in ("train", "val"):
                self.check(f"split.{name}.{split}_non_empty", bool(entry[split]))
                self.check(
                    f"split.{name}.{split}_excludes_nothing_declared",
                    not (set(int(index) for index in entry[split]) & set(self.config.collection(name).excluded)),
                )
                declared[split] |= {(name, int(index)) for index in entry[split]}
        overlap = declared["train"] & declared["val"]
        self.check("split.train_val_disjoint", not overlap, f"{len(overlap)} shared episodes")

    # -- schema ------------------------------------------------------------
    def check_info(self, split: str) -> dict[str, Any]:
        root = self.root / split
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        features = info["features"]
        self.check(f"{split}.info.codebase_version", info["codebase_version"] == "v2.1", str(info["codebase_version"]))
        self.check(f"{split}.info.fps", float(info["fps"]) == EXPECTED_FPS, str(info["fps"]))
        cameras = [key for key, spec in features.items() if spec["dtype"] in ("video", "image")]
        self.check(f"{split}.info.single_ego_camera", cameras == [CAMERA_KEY], str(cameras))
        for key, width in (
            ("observation.state", EXPECTED_STATE_DIM),
            ("observation.projected_gravity", 3),
            ("action.motion_token", EXPECTED_TOKEN_DIM),
            ("teleop.left_hand_joints", EXPECTED_HAND_DIM),
            ("teleop.right_hand_joints", EXPECTED_HAND_DIM),
        ):
            self.check(
                f"{split}.info.{key}.shape",
                [int(value) for value in features[key]["shape"]] == [width],
                str(features[key]["shape"]),
            )
        self.check(
            f"{split}.info.state_names",
            tuple(features["observation.state"]["names"]) == STATE_CHANNELS,
            str(features["observation.state"]["names"][:2]),
        )
        self.check(
            f"{split}.info.hand_names",
            tuple(features["teleop.left_hand_joints"]["names"]) == EXPECTED_LEFT_HAND_NAMES
            and tuple(features["teleop.right_hand_joints"]["names"]) == EXPECTED_RIGHT_HAND_NAMES,
            "the action hands must be named in the official actuated order",
        )
        video = features[CAMERA_KEY]["info"]
        self.check(
            f"{split}.info.video_geometry",
            (int(video["video.width"]), int(video["video.height"])) == EXPECTED_CAMERA,
            f"{video['video.width']}x{video['video.height']}",
        )
        self.check(f"{split}.info.video_codec", str(video["video.codec"]) == "h264", str(video["video.codec"]))
        self.check(f"{split}.info.video_pix_fmt", str(video["video.pix_fmt"]) == "yuv420p", str(video["video.pix_fmt"]))
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            self.check(f"{split}.info.frame_key[{key}]", key in features)
        self.check(f"{split}.info.own_stats_present", (root / f"meta/{self.config.stats_own_file}").is_file())
        self.check_stats(split)
        self.check_provenance(split)
        official = root / f"meta/{self.config.stats_official_file}"
        if official.is_file():
            self.check(
                f"{split}.official_stats",
                True,
                "meta/stats.json is present; this pipeline does not produce it, so its provenance is the "
                "official tool's record, not this check's",
            )
        else:
            self.skip(
                f"{split}.official_stats",
                f"meta/{self.config.stats_official_file} is produced later by the official GR00T tool",
            )
        return info

    def check_stats(self, split: str) -> None:
        """This pipeline's own statistics file, checked in the loader's shape.

        The official loader-facing file is ``meta/stats.json`` and is produced by
        GR00T's own tool; this one only has to be a truthful summary of what was
        written, so it is checked for coverage and shape, not for parity.
        """
        path = self.root / split / f"meta/{self.config.stats_own_file}"
        stats = json.loads(path.read_text(encoding="utf-8"))
        widths = {
            "observation.state": EXPECTED_STATE_DIM,
            "observation.projected_gravity": 3,
            "action.motion_token": EXPECTED_TOKEN_DIM,
            "teleop.left_hand_joints": EXPECTED_HAND_DIM,
            "teleop.right_hand_joints": EXPECTED_HAND_DIM,
        }
        self.check(f"{split}.stats_keys", sorted(stats) == sorted(widths), str(sorted(stats)))
        for key, width in widths.items():
            block = stats.get(key) or {}
            self.check(
                f"{split}.stats.{key}.shape",
                all(len(block.get(field, [])) == width for field in ("min", "max", "mean", "std"))
                and len(block.get("count", [])) == 1,
                str({field: len(block.get(field, [])) for field in ("min", "max", "mean", "std", "count")}),
            )
            if len(block.get("min", [])) == width and len(block.get("max", [])) == width:
                self.check(
                    f"{split}.stats.{key}.ordered",
                    all(high >= low for low, high in zip(block["min"], block["max"], strict=True)),
                )

    def check_provenance(self, split: str) -> None:
        """The recorded official-statistics command must carry both required flags.

        The command is what the pack's own provenance tells the operator to run;
        the pinned tool takes ``--dataset-path`` and ``--embodiment-tag``, and the
        tag is the enum member name, so a recorded command missing either would
        send the run nowhere.
        """
        path = self.root / split / "meta/provenance.json"
        if not path.is_file():
            self.check(f"{split}.provenance_present", False, str(path))
            return
        provenance = json.loads(path.read_text(encoding="utf-8"))
        statistics = provenance.get("statistics") or {}
        command = [str(part) for part in statistics.get("official_command") or []]
        self.check(f"{split}.provenance_official_command", "--dataset-path" in command and "--embodiment-tag" in command)
        if "--embodiment-tag" in command:
            self.check(
                f"{split}.provenance_embodiment_tag",
                command[command.index("--embodiment-tag") + 1] == EXPECTED_EMBODIMENT_TAG,
                command[command.index("--embodiment-tag") + 1],
            )
        self.check(
            f"{split}.provenance_official_not_produced",
            statistics.get("official_produced") is False,
            "this pipeline must not claim to have written the official statistics",
        )

    def check_modality(self, split: str) -> None:
        payload = json.loads((self.root / split / "meta/modality.json").read_text(encoding="utf-8"))
        self.check(f"{split}.modality_exact", payload == EXPECTED_MODALITY, str(tuple(payload)))
        try:
            assert_modality_payload(payload, self.config)
            self.check(f"{split}.modality_contract", True)
        except Exception as error:
            self.check(f"{split}.modality_contract", False, f"{type(error).__name__}: {error}")
        self.check(
            f"{split}.modality_state_is_46d",
            sum(entry["end"] - entry["start"] for entry in payload["state"].values()) == EXPECTED_MODEL_STATE_DIM,
        )
        self.check(
            f"{split}.modality_state_key_order",
            tuple(payload["state"]) == EXPECTED_STATE_KEY_ORDER,
            str(tuple(payload["state"])),
        )
        self.check(
            f"{split}.modality_state_stored_blocks",
            [tuple(payload["state"][key].items()) for key in ("left_arm", "left_hand", "right_arm", "right_hand")]
            == [
                (("start", 15), ("end", 22)),
                (("start", 22), ("end", 29)),
                (("start", 29), ("end", 36)),
                (("start", 36), ("end", 43)),
            ],
            "the stored order is lower body, left arm, left hand, right arm, right hand",
        )
        self.check(
            f"{split}.modality_action_is_78d",
            sum(entry["end"] - entry["start"] for entry in payload["action"].values()) == EXPECTED_ACTION_DIM,
        )

    def check_metadata(self, split: str, info: dict[str, Any]) -> list[dict[str, Any]]:
        root = self.root / split
        tasks = _read_jsonl(root / "meta/tasks.jsonl")
        by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
        self.check(f"{split}.task_count", len(tasks) == len(self.config.collections), str(len(tasks)))
        self.check(
            f"{split}.task_instructions",
            [by_index.get(index) for index in range(len(self.config.collections))]
            == [self.config.tasks[name] for name in self.config.collection_names],
        )
        episodes = _read_jsonl(root / "meta/episodes.jsonl")
        self.check(f"{split}.episode_count", len(episodes) == int(info["total_episodes"]), str(len(episodes)))
        self.check(f"{split}.total_frames", sum(int(row["length"]) for row in episodes) == int(info["total_frames"]))
        self.check(
            f"{split}.episode_index_contiguous",
            [int(row["episode_index"]) for row in episodes] == list(range(len(episodes))),
        )
        declared = {(str(row["source_collection"]), int(row["source_episode_index"])) for row in episodes}
        allowed = {
            (name, int(index))
            for name in self.config.collection_names
            for index in self.manifest["collections"][name][split]
        }
        self.check(
            f"{split}.episodes_within_split_manifest",
            declared <= allowed,
            f"{len(declared - allowed)} of {len(declared)} outside the manifest",
        )
        if self.require_complete:
            self.check(
                f"{split}.covers_the_whole_split",
                declared == allowed,
                f"{len(allowed - declared)} manifest episodes missing",
            )
        excluded = {
            (name, int(index))
            for name in self.config.collection_names
            for index in self.config.collection(name).excluded
        }
        self.check(f"{split}.no_excluded_episode", not (declared & excluded))
        stats = _read_jsonl(root / "meta/episodes_stats.jsonl")
        self.check(
            f"{split}.episodes_stats_cover_episodes",
            [int(row["episode_index"]) for row in stats] == list(range(len(episodes))),
        )
        return episodes

    # -- one episode -------------------------------------------------------
    def check_episode(self, split: str, row: dict[str, Any]) -> None:
        root = self.root / split
        collection, index = str(row["source_collection"]), int(row["source_episode_index"])
        label = f"{split}.{collection}.ep{index:06d}"
        episode = int(row["episode_index"])
        chunk = episode // 1000
        parquet = root / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"
        video = root / f"videos/chunk-{chunk:03d}/{CAMERA_KEY}/episode_{episode:06d}.mp4"
        if not parquet.is_file():
            self.check(f"{label}.parquet_exists", False, str(parquet))
            return

        columns = _read_parquet(
            parquet,
            [
                "observation.state", "observation.projected_gravity", "action.motion_token",
                "teleop.left_hand_joints", "teleop.right_hand_joints", "timestamp", "frame_index",
                "episode_index", "index", "task_index",
            ],
        )
        state = np.asarray(columns["observation.state"], dtype=np.float64)
        gravity = np.asarray(columns["observation.projected_gravity"], dtype=np.float64)
        token = np.asarray(columns["action.motion_token"], dtype=np.float32)
        stored_hands = np.concatenate(
            [
                np.asarray(columns["teleop.left_hand_joints"], dtype=np.float64),
                np.asarray(columns["teleop.right_hand_joints"], dtype=np.float64),
            ],
            axis=1,
        )
        timestamp = np.asarray(columns["timestamp"], dtype=np.float64)

        corpus_path = sonic_episode_path(self.config.sonic_root, collection, index)
        if not corpus_path.is_file():
            raise ValidationError(f"missing frozen SONIC episode: {corpus_path}")
        with np.load(corpus_path, allow_pickle=False) as payload:
            corpus_action = np.asarray(payload["action"], dtype=np.float32)
            corpus_timestamp = np.asarray(payload["timestamp"], dtype=np.float64)
            corpus_mask = np.asarray(payload["training_valid_mask"], dtype=bool)
        raw = read_raw_episode(self.config, collection, index)

        rows_declared = int(row["length"])
        invalid = np.flatnonzero(~corpus_mask)
        self.check(
            f"{label}.corpus_tail_is_45",
            invalid.size == EXPECTED_TAIL_ROWS
            and np.array_equal(invalid, np.arange(corpus_mask.size - EXPECTED_TAIL_ROWS, corpus_mask.size)),
            f"{invalid.size} invalid rows",
        )
        rows = int(corpus_mask.size) - EXPECTED_TAIL_ROWS
        self.check(f"{label}.train_row_count", rows_declared == rows, f"{rows_declared} vs {rows}")
        self.check(f"{label}.horizon", rows >= EXPECTED_HORIZON, str(rows))
        self.check(f"{label}.state_shape", state.shape == (rows, EXPECTED_STATE_DIM), str(state.shape))
        self.check(f"{label}.token_shape", token.shape == (rows, EXPECTED_TOKEN_DIM), str(token.shape))
        self.check(f"{label}.hands_shape", stored_hands.shape == (rows, 2 * EXPECTED_HAND_DIM), str(stored_hands.shape))
        finite = bool(
            np.isfinite(state).all() and np.isfinite(token).all() and np.isfinite(stored_hands).all()
        )
        self.check(f"{label}.finite", finite)
        if not finite:
            return

        # The stored timeline is the frozen corpus timeline, first row included.
        self.check(
            f"{label}.timestamps_are_corpus_timestamps",
            bool(np.allclose(timestamp, corpus_timestamp[:rows], rtol=0, atol=TIMESTAMP_TOLERANCE_S)),
            f"max deviation {float(np.abs(timestamp - corpus_timestamp[:rows]).max()):.2e}s",
        )
        self.check(
            f"{label}.timeline_is_uniform_50hz",
            bool(np.allclose(np.diff(corpus_timestamp[:rows]), 1.0 / EXPECTED_FPS, rtol=0, atol=1e-6)),
        )
        self.check(f"{label}.token_is_the_corpus_token", bool(np.array_equal(token, corpus_action[:rows, :64])))
        self.check(f"{label}.frame_index_contiguous", list(columns["frame_index"]) == list(range(rows)))
        self.check(f"{label}.episode_index", set(columns["episode_index"]) == {episode})
        # The instruction reaches the loader through the frame's task_index, so the
        # index is re-derived from the collection name rather than read back.
        expected_task = self.config.collection_names.index(collection)
        self.check(f"{label}.task_index", set(columns["task_index"]) == {expected_task})
        self.check(f"{label}.task_instruction", list(row.get("tasks") or []) == [self.config.tasks[collection]])
        first = int(columns["index"][0])
        self.check(f"{label}.index_contiguous", list(columns["index"]) == list(range(first, first + rows)))

        # Independent re-derivation of the measured state and the action hands.
        #
        # The grid is the frozen corpus timeline in float64, which is what the
        # converter interpolated on. The stored `timestamp` column is float32
        # (LeRobot v2.1 stores it that way) and rounds the grid by up to about
        # 2e-6 s, which a fast hand turns into tens of micro-radians: re-deriving
        # on the rounded column would flag a numerically perfect pack. That
        # column is checked against the corpus on its own, above.
        grid = corpus_timestamp[:rows]
        measured, expected_ledger = repaired_measured(raw.state, raw.state_names, label)
        state_positions = _positions(raw.state_names)
        source_order = [state_positions[field] for field in MEASURED_SOURCE_FIELDS]
        expected_state = np.concatenate(
            [
                np.tile(standing_lower_body(), (rows, 1)),
                _interpolate(raw.timestamp, measured[:, source_order], grid),
            ],
            axis=1,
        )
        self.check(
            f"{label}.state_matches_raw_by_name",
            bool(np.allclose(state, expected_state, rtol=0, atol=1e-6)),
            f"max deviation {float(np.abs(state - expected_state).max()):.2e} rad",
        )
        self.check(f"{label}.lower_body_is_constant", bool(np.all(state[:, :15] == state[0, :15])))
        self.check(
            f"{label}.gravity_is_synthetic_upright",
            bool(np.array_equal(gravity, np.tile(np.array([0.0, 0.0, -1.0]), (rows, 1)))),
        )

        action_positions = _positions(raw.action_names)
        desired = raw.action[:, [action_positions[field] for field in ACTION_SOURCE_FIELDS]]
        expected_hands = _interpolate(raw.timestamp, desired, grid)
        self.check(
            f"{label}.hands_match_rebuilt_desired_action",
            bool(np.allclose(stored_hands, expected_hands, rtol=0, atol=1e-6)),
            f"max deviation {float(np.abs(stored_hands - expected_hands).max()):.2e} rad",
        )
        corpus_columns = [CORPUS_HAND_ORDER["left"].index(stem) for stem in ACTUATED_HAND_ORDER]
        corpus_columns += [7 + CORPUS_HAND_ORDER["right"].index(stem) for stem in ACTUATED_HAND_ORDER]
        self.check(
            f"{label}.hands_match_name_permuted_corpus",
            bool(
                np.allclose(
                    stored_hands,
                    np.asarray(corpus_action[:rows, 64:], dtype=np.float64)[:, corpus_columns],
                    rtol=0,
                    atol=self.config.corpus_hand_tolerance_rad,
                )
            ),
        )

        reported = {str(key): int(value) for key, value in ((row.get("hand_repair") or {}).get("channels") or {}).items()}
        self.check(
            f"{label}.repair_ledger_matches_source",
            reported == expected_ledger,
            f"reported {reported}, recount {expected_ledger}",
        )
        self.check(f"{label}.trimmed_rows", int(row.get("trimmed_rows", 0)) == EXPECTED_TAIL_ROWS)
        self.repaired_samples += int(sum(expected_ledger.values()))

        if not video.is_file():
            self.check(f"{label}.video_exists", False, str(video))
            return
        decoded = _decode_video(video)
        self.check(f"{label}.video_frames", decoded["frames"] == rows_declared, f"{decoded['frames']} vs {rows_declared}")
        self.check(
            f"{label}.video_geometry",
            (decoded["width"], decoded["height"]) == EXPECTED_CAMERA,
            f"{decoded['width']}x{decoded['height']}",
        )
        self.check(f"{label}.video_fps", abs(decoded["fps"] - EXPECTED_FPS) < 0.01, str(decoded["fps"]))
        self.check(f"{label}.video_codec", decoded["codec"] == "h264", decoded["codec"])
        self.check(f"{label}.video_pix_fmt", decoded["pix_fmt"] == "yuv420p", decoded["pix_fmt"])
        alignment = video_alignment(video, raw.video_file, raw.video_start_s)
        if alignment is None:
            self.skip(f"{label}.video_alignment", "the source segment could not be decoded for comparison")
        else:
            self.check(
                f"{label}.video_alignment",
                alignment["expected"] <= VIDEO_ALIGNMENT_MAX_DIFF
                and alignment["probe"] >= alignment["expected"] - VIDEO_ALIGNMENT_MARGIN,
                f"first frame differs from the segment start by {alignment['expected']:.2f} grey levels "
                f"and from a later source frame by {alignment['probe']:.2f}",
            )

        self.episodes_checked += 1
        self.frames_checked += rows

    # -- consumers ---------------------------------------------------------
    def check_lerobot_load(self, split: str) -> None:
        """Open the split with the pinned LeRobot loader, as the pack's consumer does."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as error:  # pragma: no cover - environment diagnostic
            self.skip(f"{split}.lerobot_load", f"lerobot is not installed here: {type(error).__name__}")
            return
        try:
            dataset = LeRobotDataset(repo_id=split, root=self.root / split, video_backend="pyav")
            item = dataset[0]
        except Exception as error:
            self.check(f"{split}.lerobot_load", False, f"{type(error).__name__}: {error}")
            return
        self.check(f"{split}.lerobot_load", True, f"{len(dataset)} frames")
        image = item.get(CAMERA_KEY)
        shape = tuple(image.shape) if hasattr(image, "shape") else None
        self.check(
            f"{split}.lerobot_video_tensor",
            shape is not None and shape[1:] == (EXPECTED_CAMERA[1], EXPECTED_CAMERA[0]),
            str(shape),
        )

    def check_groot_registry(self, split: str) -> None:
        """Compare the pack's modality keys with the pre-registered embodiment.

        The registry is read from its source with :mod:`ast`: the pinned file keys
        ``MODALITY_CONFIGS`` by the tag *value* (``unitree_g1_sonic``), while the
        statistics command line takes the enum *member name*, and the training
        image's interpreter cannot import the package for a check that must run
        before the model stack is up. A missing entry, a different key list or a
        different action horizon all fail; only an unmounted checkout is skipped.
        """
        config_path = self.groot_source / REGISTRY_RELATIVE_PATH
        if not config_path.is_file():
            self.skip(f"{split}.groot_registry", f"the pinned Isaac-GR00T source is not mounted at {self.groot_source}")
            return
        try:
            entry = parse_registry_entry(config_path.read_text(encoding="utf-8"))
            detail = assert_registry_matches_contract(entry, EXPECTED_HORIZON)
        except ValidationError as error:
            self.check(f"{split}.groot_registry", False, str(error))
            return
        self.check(f"{split}.groot_registry", True, f"{REGISTRY_KEY}: {detail}")

    def report(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "status": self.status,
            "episodes_checked": self.episodes_checked,
            "frames_checked": self.frames_checked,
            "repaired_source_samples": self.repaired_samples,
            "failed": [entry for entry in self.checks if entry["status"] == "FAIL"],
            "skipped": [entry for entry in self.checks if entry["status"] == "SKIP"],
            "checks": self.checks,
        }


def validate_dataset(
    config: ConversionConfig,
    root: Path,
    manifest: dict[str, Any],
    *,
    splits: tuple[str, ...] | None = None,
    check_lerobot: bool = True,
    check_groot: bool = True,
    require_complete: bool = False,
    groot_source: Path | None = None,
) -> dict[str, Any]:
    """Validate a produced dataset root and return a machine-readable report."""
    validator = DatasetValidator(
        config, Path(root), manifest, require_complete=require_complete, groot_source=groot_source
    )
    validator.check_contract_literals()
    validator.check_split()
    for split in splits or (config.train_repo, config.val_repo):
        split_root = validator.root / split
        for relative in REQUIRED_META:
            if not (split_root / relative).is_file():
                raise ValidationError(f"missing {split_root / relative}")
        info = validator.check_info(split)
        validator.check_modality(split)
        for row in validator.check_metadata(split, info):
            validator.check_episode(split, row)
        if check_lerobot:
            validator.check_lerobot_load(split)
        if check_groot:
            validator.check_groot_registry(split)
    return validator.report()


__all__ = [
    "ACTION_SOURCE_FIELDS",
    "DatasetValidator",
    "EXPECTED_MODALITY",
    "MEASURED_SOURCE_FIELDS",
    "REGISTRY_KEY",
    "REGISTRY_RELATIVE_PATH",
    "STATE_CHANNELS",
    "ValidationError",
    "assert_registry_matches_contract",
    "invalid_mask",
    "parse_registry_entry",
    "repaired_measured",
    "validate_dataset",
]
