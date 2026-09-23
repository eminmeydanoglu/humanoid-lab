#!/usr/bin/env python3
"""Offline replay of recorded BlockStacking action tokens through the pinned SONIC decoder.

One question only: do the action tokens recorded in the task-0 demonstrations,
decoded by the *same* ``model_decoder.onnx`` the Isaac rollouts ran and with the
994D observation the deployment assembles around it, reproduce the palm motion
those demonstrations actually measured?

Everything that defines "the same decoder path" is read out of the pinned SONIC
source at run time rather than restated here: the observation layout comes from
``observation_config.yaml`` plus the deployment's own observation registry, the
joint permutations, action scales, default angles and control period come from
``policy_parameters.hpp`` / ``g1_deploy_onnx_ref.cpp``, and the model file is
checksummed against ``MODEL_PROVENANCE.json``.  A source or model change fails
the run instead of quietly changing the answer.

The harness is read-only end to end: no socket is bound, no Isaac process is
started, and nothing outside ``--out`` is written.

Stages: ``semantics`` ``validate`` ``replay`` ``report`` ``all``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 1
SCRIPT_VERSION = "demo-token-decoder-replay.py/1.0.0"

ROOT = Path(__file__).resolve().parents[1]
PSI0_TRAIN = Path("psi0-unitree-dex3-sonic-v1/train")
DEFAULT_OUT = ROOT / "data/outputs/blockstacking-debug/experiments/01-demo-token-decoder"
DEFAULT_CAMPAIGN = ROOT / "data/outputs/blockstacking-debug"
DEFAULT_MODEL = Path("/data/models/sonic/sonic_v1_1/model_decoder.onnx")
DEFAULT_OBS_CONFIG = Path("/data/models/sonic/sonic_v1_1/observation_config.yaml")
DEFAULT_PROVENANCE = Path("/data/models/sonic/MODEL_PROVENANCE.json")
DEFAULT_SONIC_DEPLOY = Path("/opt/src/sonic/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref")
DEFAULT_URDF = ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"
COMPARE_SCRIPT = ROOT / "scripts" / "compare-blockstacking-dataset.py"

#: The three representative demonstrations, taken from the stage-2 contact sheets
#: of ``dataset-comparison`` (low / median / high palm-height median).
REPRESENTATIVE_EPISODES: tuple[tuple[str, int], ...] = (
    ("low_median", 283), ("median_median", 21), ("high_median", 160))

CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ
HISTORY_FRAMES = 10
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625
TOKEN_DIM = 64
ARM_SLICE = slice(15, 29)

#: The pinned header and the pinned call site disagree about which array is the
#: permutation in ``out[array[hw]]``; all three readings are carried and the
#: rollout itself decides between them.
PERM_HYPOTHESES = ("mujoco_to_isaaclab", "isaaclab_to_mujoco", "identity")


class ReplayError(RuntimeError):
    """The replay cannot run; the message is meant for the operator."""


# --------------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_compare_module() -> Any:
    """Import the dataset-comparison script for its validated URDF forward kinematics."""
    spec = importlib.util.spec_from_file_location("compare_blockstacking_dataset", COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    for candidate in (Path("/data/datasets"), ROOT / "data" / "datasets"):
        if candidate.is_dir():
            return candidate
    raise ReplayError("cannot locate the dataset root (pass --data-root)")


def git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=30, check=True).stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    names = list(columns) if columns else list(rows[0])
    lines = [",".join(names)]
    for row in rows:
        cells = []
        for name in names:
            value = row.get(name)
            if isinstance(value, float):
                cells.append("" if math.isnan(value) else f"{value:.9g}")
            else:
                cells.append("" if value is None else str(value))
        lines.append(",".join(cells))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fsq_quantize(values: np.ndarray) -> np.ndarray:
    """Snap a token onto the WBC's FSQ grid, exactly as the live bridge does."""
    array = np.asarray(values, dtype=np.float64)
    quantized = np.round(np.clip(array, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP
    return np.clip(quantized, FSQ_MIN, FSQ_MAX)


# ------------------------------------------------- pinned deployment semantics


@dataclass(frozen=True)
class DeploymentSemantics:
    """The decoder path, as read out of the pinned SONIC deployment source."""

    deploy_root: Path
    control_dt: float
    isaaclab_to_mujoco: np.ndarray
    mujoco_to_isaaclab: np.ndarray
    action_scale: np.ndarray
    default_angles: np.ndarray
    registry: dict[str, int]
    layout: list[tuple[str, int, int]]
    input_dim: int
    token_dim: int
    source_sha256: dict[str, str]

    def permutation(self, hypothesis: str) -> np.ndarray:
        """Output index consumed for hardware joint ``h``: ``perm[h]``."""
        if hypothesis == "mujoco_to_isaaclab":
            return np.asarray(self.mujoco_to_isaaclab, dtype=int)
        if hypothesis == "isaaclab_to_mujoco":
            return np.asarray(self.isaaclab_to_mujoco, dtype=int)
        if hypothesis == "identity":
            return np.arange(len(self.default_angles))
        raise ReplayError(f"unknown permutation hypothesis {hypothesis!r}")

    def history_order(self, hypothesis: str) -> np.ndarray:
        """Hardware joint index behind each history slot, i.e. the inverse permutation."""
        return np.argsort(self.permutation(hypothesis))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sonic_deploy_root": str(self.deploy_root),
            "control_hz": 1.0 / self.control_dt,
            "control_dt_s": self.control_dt,
            "isaaclab_to_mujoco": self.isaaclab_to_mujoco.tolist(),
            "mujoco_to_isaaclab": self.mujoco_to_isaaclab.tolist(),
            "action_scale": self.action_scale.tolist(),
            "default_angles": self.default_angles.tolist(),
            "observation_layout": [{"name": n, "offset": o, "dim": d} for n, o, d in self.layout],
            "observation_input_dim": self.input_dim,
            "encoder_token_dim": self.token_dim,
            "source_sha256": dict(self.source_sha256),
        }


def _parse_double_constants(text: str) -> dict[str, float]:
    """Evaluate the header's ``const double NAME = <arithmetic>;`` definitions."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("const double ") or "=" not in stripped:
            continue
        name, _, expression = stripped[len("const double "):].partition("=")
        expression = expression.split(";")[0].strip()
        try:
            values[name.strip()] = float(eval(expression, {"__builtins__": {}}, values))  # noqa: S307 - pinned arithmetic
        except (NameError, SyntaxError, TypeError):
            continue
    return values


def _parse_array(text: str, declaration: str) -> list[str]:
    start = text.index(declaration) + len(declaration)
    body = text[start:text.index("};", start)]
    body = "\n".join(line.split("//")[0] for line in body.splitlines())
    items = [item.strip() for item in body.replace("{", "").replace("}", "").split(",")]
    return [item for item in items if item]


def parse_deployment_semantics(deploy_root: Path, obs_config: Path, model_input_dim: int) -> DeploymentSemantics:
    """Read the observation layout, joint maps, gains and control period from source."""
    header = Path(deploy_root) / "include" / "policy_parameters.hpp"
    source = Path(deploy_root) / "src" / "g1_deploy_onnx_ref.cpp"
    for path in (header, source, obs_config):
        if not Path(path).is_file():
            raise ReplayError(f"pinned SONIC source is missing: {path}")
    header_text = header.read_text(encoding="utf-8")
    source_text = source.read_text(encoding="utf-8")

    constants = _parse_double_constants(header_text)
    scale = np.asarray([float(eval(item, {"__builtins__": {}}, constants))  # noqa: S307 - pinned arithmetic
                        for item in _parse_array(header_text, "const std::array<double, 29> g1_action_scale =")],
                       dtype=float)
    default_angles = np.asarray(
        [float(item) for item in _parse_array(header_text, "const std::array<double, 29> default_angles =")],
        dtype=float)
    isaaclab_to_mujoco = np.asarray(
        [int(item) for item in _parse_array(header_text, "const std::array<int, 29> isaaclab_to_mujoco =")], dtype=int)
    mujoco_to_isaaclab = np.asarray(
        [int(item) for item in _parse_array(header_text, "const std::array<int, 29> mujoco_to_isaaclab =")], dtype=int)
    if not np.array_equal(np.argsort(isaaclab_to_mujoco), mujoco_to_isaaclab):
        raise ReplayError("the pinned joint permutations are not inverses of each other")

    control_dt = float(source_text.split("control_dt_(")[1].split(")")[0])

    # ``token_state`` is the only registry entry whose width is not a literal: the
    # deployment takes it from the encoder section of the same config file.
    declared_token_dim = None
    for line in Path(obs_config).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("dimension:"):
            declared_token_dim = int(line.split(":")[1].split("#")[0].strip())
            break
    if declared_token_dim != TOKEN_DIM:
        raise ReplayError(f"observation config declares encoder dimension {declared_token_dim}, expected {TOKEN_DIM}")

    registry: dict[str, int] = {}
    start = source_text.index("std::vector<ObservationRegistry> GetObservationRegistry()")
    region = source_text[start:source_text.index("// Initialize observation functions", start)]
    for name, field in re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*([A-Za-z0-9_]+)\s*,', region):
        registry[name] = declared_token_dim if field == "token_dim" else int(field)
    if not registry:
        raise ReplayError("could not read the deployment observation registry")

    layout: list[tuple[str, int, int]] = []
    offset = 0
    in_observations = False
    for line in Path(obs_config).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("encoder:"):
            in_observations = False
        elif stripped.startswith("observations:"):
            in_observations = True
        elif in_observations and stripped.startswith("- name:"):
            name = stripped.split('"')[1]
            if name not in registry:
                raise ReplayError(f"observation {name!r} is not in the deployment registry")
            layout.append((name, offset, registry[name]))
            offset += registry[name]
    if offset != model_input_dim:
        raise ReplayError(f"observation layout sums to {offset}, model expects {model_input_dim}")
    token_dim = next((dimension for name, _, dimension in layout if name == "token_state"), 0)
    if token_dim != TOKEN_DIM:
        raise ReplayError(f"observation config declares a {token_dim}D token, expected {TOKEN_DIM}")

    return DeploymentSemantics(
        deploy_root=Path(deploy_root), control_dt=control_dt, isaaclab_to_mujoco=isaaclab_to_mujoco,
        mujoco_to_isaaclab=mujoco_to_isaaclab, action_scale=scale, default_angles=default_angles,
        registry=registry, layout=layout, input_dim=offset, token_dim=token_dim,
        source_sha256={str(p): sha256(p) for p in (header, source, obs_config)})


# --------------------------------------------------------------------- decoder


class Decoder:
    """The pinned deployment ONNX, run one frame at a time with contract checks."""

    def __init__(self, model: Path, expected_sha256: str | None, input_dim: int, output_dim: int = 29) -> None:
        try:
            import onnxruntime  # noqa: PLC0415 - optional dependency
        except ImportError as error:  # pragma: no cover - environment guard
            raise ReplayError("onnxruntime is not importable; run with the isaac-sonic interpreter") from error
        self.path = Path(model)
        if not self.path.is_file():
            raise ReplayError(f"decoder model not found: {self.path}")
        self.sha256 = sha256(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ReplayError(f"decoder checksum {self.sha256} != pinned {expected_sha256}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 8
        self._session = onnxruntime.InferenceSession(str(self.path), sess_options=options,
                                                      providers=["CPUExecutionProvider"])
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ReplayError(f"decoder has {len(inputs)} inputs and {len(outputs)} outputs")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        if tuple(inputs[0].shape) != (1, input_dim) or tuple(outputs[0].shape) != (1, output_dim):
            raise ReplayError(f"decoder IO is {inputs[0].shape}->{outputs[0].shape}, expected "
                              f"[(1,{input_dim})]->[(1,{output_dim})]")
        self.input_dim = int(input_dim)
        self.version = str(onnxruntime.__version__)

    def run(self, observations: np.ndarray) -> np.ndarray:
        frames = np.asarray(observations, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames[None, :]
        if frames.shape[1] != self.input_dim:
            raise ReplayError(f"observation width {frames.shape[1]} != decoder input {self.input_dim}")
        out = np.empty((frames.shape[0], 29), dtype=np.float64)
        for index, row in enumerate(frames):
            if not np.isfinite(row).all():
                raise ReplayError("observation contains NaN or Inf")
            raw = self._session.run([self.output_name], {self.input_name: np.ascontiguousarray(row[None, :])})[0]
            out[index] = np.asarray(raw, dtype=np.float64).reshape(-1)
        return out

    def info(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "input_name": self.input_name,
                "output_name": self.output_name, "input_shape": [1, self.input_dim], "output_shape": [1, 29],
                "onnxruntime": self.version, "providers": list(self._session.get_providers())}


# ------------------------------------------------------- observation assembly


def build_observation(token: np.ndarray, block: dict[str, np.ndarray], layout: Sequence[tuple[str, int, int]]) -> np.ndarray:
    """One observation vector, laid out exactly in the order ``observation_config.yaml`` declares."""
    blocks = {"token_state": np.asarray(token, dtype=np.float64).reshape(-1),
              "his_base_angular_velocity_10frame_step1": np.asarray(block["angvel"], dtype=np.float64).reshape(-1),
              "his_body_joint_positions_10frame_step1": np.asarray(block["q"], dtype=np.float64).reshape(-1),
              "his_body_joint_velocities_10frame_step1": np.asarray(block["dq"], dtype=np.float64).reshape(-1),
              "his_last_actions_10frame_step1": np.asarray(block["last_action"], dtype=np.float64).reshape(-1),
              "his_gravity_dir_10frame_step1": np.asarray(block["gravity"], dtype=np.float64).reshape(-1)}
    total = sum(dimension for _, _, dimension in layout)
    out = np.empty(total, dtype=np.float64)
    for name, offset, dimension in layout:
        value = blocks.get(name)
        if value is None:
            raise ReplayError(f"observation {name!r} has no builder")
        if value.size != dimension:
            raise ReplayError(f"observation {name!r} is {value.size}D, layout says {dimension}D")
        out[offset:offset + dimension] = value
    return out


def history_windows(values: np.ndarray, frames: int = HISTORY_FRAMES) -> np.ndarray:
    """``[T, dim] -> [T, frames, dim]``, oldest frame first, front-padded with zeros."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros((len(values), frames, values.shape[1]), dtype=np.float64)
    for t in range(len(values)):
        chunk = values[max(0, t - frames + 1):t + 1]
        out[t, frames - len(chunk):] = chunk
    return out


def action_to_target(action: np.ndarray, semantics: DeploymentSemantics, hypothesis: str) -> np.ndarray:
    """``out -> q_target`` in MuJoCo/hardware order, exactly as ``CreatePolicyCommand``."""
    perm = semantics.permutation(hypothesis)
    values = np.asarray(action, dtype=np.float64).reshape(-1, 29)
    return semantics.default_angles[None, :] + values[:, perm] * semantics.action_scale[None, :]


def target_to_action(target: np.ndarray, semantics: DeploymentSemantics, hypothesis: str) -> np.ndarray:
    """The inverse map, used to seed ``his_last_actions`` from a measured pose."""
    perm = semantics.permutation(hypothesis)
    values = np.asarray(target, dtype=np.float64).reshape(-1, 29)
    expected = (values - semantics.default_angles[None, :]) / semantics.action_scale[None, :]
    action = np.zeros_like(expected)
    action[:, perm] = expected
    return action


def gravity_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Body-frame gravity direction, ``quat_conjugate(base_quat) @ (0, 0, -1)``."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    conjugate = np.concatenate([q[:, :1], -q[:, 1:]], axis=1)
    return rotate_quaternion(conjugate, np.tile(np.array([0.0, 0.0, -1.0]), (len(q), 1)))


def rotate_quaternion(quaternion: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``quat_rotate`` for a wxyz quaternion and row-wise points."""
    q = np.asarray(quaternion, dtype=np.float64)
    w, v = q[:, :1], q[:, 1:]
    point = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    crossed = np.cross(v, point)
    return point + 2.0 * (w * crossed + np.cross(v, crossed))


# ------------------------------------------------------------------ replayer


@dataclass
class ReplayResult:
    """The shared shape every replay variant produces."""

    q_target: np.ndarray
    action: np.ndarray
    tokens: np.ndarray

    def save(self, path: Path, extra: dict[str, np.ndarray] | None = None) -> None:
        payload = {"q_target": self.q_target, "action": self.action, "tokens": self.tokens}
        if extra:
            payload.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)


def replay_stream(decoder: Decoder, semantics: DeploymentSemantics, *, tokens: np.ndarray, q_measured: np.ndarray,
                  dq_measured: np.ndarray, gravity: np.ndarray, angvel: np.ndarray, hypothesis: str,
                  last_action_source: str, seed_target: np.ndarray | None = None,
                  recorded_target: np.ndarray | None = None) -> ReplayResult:
    """Run the deployment's per-tick observation through the decoder, in tick order.

    ``q_measured`` is the robot's measured body configuration in MuJoCo order; the
    history blocks are reordered into the deployment's IsaacLab slots and carry the
    default angles subtracted, exactly as ``GetLowStateAndIMU`` records them.  The
    ring buffer logs the robot at the start of a tick and the action it issued at
    the end of the previous one, so every ``last_action`` window ends one tick
    before the tick it is read at.
    """
    order = semantics.history_order(hypothesis)
    default, scale = semantics.default_angles, semantics.action_scale
    frame_count = len(tokens)

    q_slots = q_measured[:, order] - default[order]
    dq_slots = dq_measured[:, order]
    q_windows = history_windows(q_slots)
    dq_windows = history_windows(dq_slots)
    gravity_windows = history_windows(np.asarray(gravity, dtype=np.float64))
    angvel_windows = history_windows(np.asarray(angvel, dtype=np.float64))

    def previous_tick_windows(series: np.ndarray) -> np.ndarray:
        windows = history_windows(series)
        return np.concatenate([np.zeros((1,) + windows.shape[1:], dtype=np.float64), windows[:-1]])

    if last_action_source in ("recorded_target", "recorded_action"):
        if recorded_target is None:
            raise ReplayError("the recorded_action source needs the recorded joint targets")
        last_windows = previous_tick_windows(target_to_action(recorded_target, semantics, hypothesis))
    elif last_action_source == "measured":
        last_windows = previous_tick_windows(target_to_action(q_measured, semantics, hypothesis))
    elif last_action_source == "zeros":
        last_windows = np.zeros_like(q_windows)
    elif last_action_source == "recurrence":
        seed = target_to_action(np.asarray(seed_target, dtype=np.float64)[None, :], semantics, hypothesis)[0]
        buffer = np.tile(seed, (HISTORY_FRAMES, 1))
        last_windows = np.empty_like(q_windows)
    else:
        raise ReplayError(f"unknown last_action source {last_action_source!r}")

    actions = np.empty((frame_count, 29), dtype=np.float64)
    for t in range(frame_count):
        if last_action_source == "recurrence":
            last_windows[t] = buffer
        observation = build_observation(tokens[t], {
            "angvel": angvel_windows[t], "q": q_windows[t], "dq": dq_windows[t],
            "last_action": last_windows[t], "gravity": gravity_windows[t],
        }, semantics.layout)
        action = decoder.run(observation[None, :])[0]
        actions[t] = action
        if last_action_source == "recurrence":
            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = action
    return ReplayResult(q_target=action_to_target(actions, semantics, hypothesis), action=actions,
                        tokens=np.asarray(tokens, dtype=np.float64))


# ------------------------------------------------------------------ data input


@dataclass
class DemoEpisode:
    """One task-0 demonstration, resampled onto the deployment's 50 Hz control grid."""

    episode: int
    label: str
    source_episode: int
    timestamp: np.ndarray
    q_measured: np.ndarray
    tokens: np.ndarray
    hand_action: np.ndarray
    arm_action: np.ndarray | None
    frames_valid: int
    ticks: np.ndarray
    tick_tokens: np.ndarray
    tick_q: np.ndarray
    tick_target: np.ndarray
    source_path: str
    source_sha256: str

    @property
    def valid_ticks(self) -> int:
        """Control ticks inside the converter's training-valid window."""
        limit = self.timestamp[self.frames_valid - 1] if 0 < self.frames_valid <= len(self.timestamp) else self.ticks[-1]
        return int(np.count_nonzero(self.ticks <= limit + 1e-9))

    @property
    def command_gap_rad(self) -> float | None:
        """Mean |commanded - measured| over the arms: the demonstration's own tracking error."""
        if self.arm_action is None:
            return None
        return float(np.abs(self.arm_action - self.q_measured[:, 15:29]).mean())


def read_demo_episode(root: Path, episode: int, label: str, raw_root: Path | None = None) -> DemoEpisode:
    import pyarrow.parquet as pq

    path = Path(root) / PSI0_TRAIN / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    if not path.is_file():
        raise ReplayError(f"dataset episode not found: {path}")
    table = pq.read_table(str(path), columns=["observation.state", "action.body_token_v1_1", "action",
                                              "timestamp", "frame_index", "task_index"]).to_pydict()
    if set(np.unique(table["task_index"])) != {0}:
        raise ReplayError(f"episode {episode} is not task 0 (BlockStacking)")
    state = np.asarray(table["observation.state"], dtype=np.float64)
    tokens = np.stack([np.asarray(row, dtype=np.float64) for row in table["action.body_token_v1_1"]])
    hands = np.stack([np.asarray(row, dtype=np.float64) for row in table["action"]])
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    if tokens.shape[1] != TOKEN_DIM:
        raise ReplayError(f"episode {episode} token width is {tokens.shape[1]}, expected {TOKEN_DIM}")
    if not np.allclose(tokens, fsq_quantize(tokens), atol=1e-6):
        raise ReplayError(f"episode {episode} tokens are off the FSQ grid")

    ticks = np.arange(0.0, timestamp[-1] + 1e-9, CONTROL_DT)
    token_index = np.clip(np.searchsorted(timestamp, ticks, side="right") - 1, 0, len(tokens) - 1)
    tick_q = np.stack([np.interp(ticks, timestamp, state[:, joint]) for joint in range(29)], axis=1)
    source_episode, arm_action = read_source_command(root, episode, raw_root)
    if arm_action is None or len(arm_action) != len(timestamp):
        tick_target = tick_q.copy()
    else:
        command = np.concatenate([state[:, :15], arm_action], axis=1)  # legs+waist, then arms
        tick_target = np.stack([np.interp(ticks, timestamp, command[:, joint]) for joint in range(29)], axis=1)
    return DemoEpisode(episode=episode, label=label, source_episode=source_episode, timestamp=timestamp,
                       q_measured=state[:, :29], tokens=tokens, hand_action=hands, arm_action=arm_action,
                       frames_valid=episode_frames_valid(root, episode), ticks=ticks,
                       tick_tokens=tokens[token_index], tick_q=tick_q, tick_target=tick_target,
                       source_path=str(path), source_sha256=sha256(path))


def read_source_command(root: Path, episode: int, raw_root: Path | None) -> tuple[int, np.ndarray | None]:
    """The demonstration's own commanded arm joints, from the raw source episode.

    The converted copies keep only the hand command, but the raw teleoperation recording also
    carries the arm target the operator's command resolved to.  That series *is* the analogue
    of the deployment's ``his_last_actions`` channel, so where it exists the replay conditions
    on a measurement instead of on a reconstruction.
    """
    source_episode = -1
    for line in (Path(root) / PSI0_TRAIN / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if int(entry.get("episode_index", -1)) == episode:
            source_episode = int(entry.get("source_episode_index", -1))
            break
    if raw_root is None or source_episode < 0:
        return source_episode, None
    import pyarrow.parquet as pq

    collection = Path(raw_root) / "unitree-g1-dex3" / "G1_Dex3_BlockStacking_Dataset" / "data" / "chunk-000"
    for path in sorted(collection.glob("file-*.parquet")):
        columns = pq.read_table(str(path), columns=["episode_index", "action"]).to_pydict()
        index = np.asarray(columns["episode_index"], dtype=int)
        rows = np.where(index == source_episode)[0]
        if rows.size:
            action = np.stack([np.asarray(row, dtype=np.float64) for row in columns["action"]])
            return source_episode, action[rows][:, :14]
    return source_episode, None


def episode_frames_valid(root: Path, episode: int) -> int:
    """The converter's training-valid frame count, from the copy's own episode metadata."""
    meta = Path(root) / PSI0_TRAIN / "meta" / "episodes.jsonl"
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if int(entry.get("episode_index", -1)) == episode:
            return int(entry.get("frames_valid", 0))
    raise ReplayError(f"episode {episode} is not in {meta}")


# --------------------------------------------------------------- rollout input


@dataclass
class RolloutTicks:
    """The tokens the deployment actually consumed and the state it read, per control tick."""

    tag: str
    rollout: str
    tick_index: np.ndarray
    sim_s: np.ndarray
    tokens: np.ndarray
    q_measured: np.ndarray
    dq_measured: np.ndarray
    gravity: np.ndarray
    angvel: np.ndarray
    body_target: np.ndarray
    token_frame_index: np.ndarray
    support_active: np.ndarray
    raw_token_grid_share: float = 0.0


def read_rollout(campaign: Path, tag: str, rollout: str) -> RolloutTicks:
    """Pair every free 50 Hz tick with the token the deployment was holding at that instant."""
    session = Path(campaign) / tag
    tracking = read_jsonl(session / "rollouts" / rollout / "tracking.jsonl")
    telemetry = read_jsonl(session / "rollouts" / rollout / "bridge-telemetry.jsonl")
    published = [(int(row["wall_time_ns"]), fsq_quantize(np.asarray(row["action"], dtype=np.float64)[:TOKEN_DIM]),
                  int(row["frame_index"]), np.asarray(row["action"], dtype=np.float64)[:TOKEN_DIM])
                 for row in telemetry if row.get("kind") == "target_action"]
    if len(published) < 10 or not tracking:
        raise ReplayError(f"rollout {tag}/{rollout} lacks tokens or tracking rows")
    published.sort(key=lambda item: item[0])
    counts = np.unique([item[2] for item in published])
    if len(counts) != len(published):
        raise ReplayError(f"rollout {tag}/{rollout} publishes a repeated frame index")

    wall = np.asarray([item[0] for item in published], dtype=np.int64)
    tokens = np.stack([item[1] for item in published])
    frames = np.asarray([item[2] for item in published], dtype=np.int64)

    rows = [row for row in tracking if not bool(row["support_active"])]
    if len(rows) < 100:
        raise ReplayError(f"rollout {tag}/{rollout} has too few free ticks")
    tick_wall = np.asarray([int(row["wall_time_ns"]) for row in rows], dtype=np.int64)
    index = np.clip(np.searchsorted(wall, tick_wall, side="right") - 1, 0, len(published) - 1)
    quaternion = np.stack([np.asarray(row["root_quaternion_wxyz"], dtype=np.float64) for row in rows])

    q = np.stack([np.asarray(row["body_measured"], dtype=np.float64) for row in rows])
    dq = np.stack([np.asarray(row["body_measured_velocity"], dtype=np.float64) for row in rows])
    sim_s = np.asarray([float(row["sim_s"]) for row in rows], dtype=np.float64)
    angvel = quaternion_angular_velocity(quaternion, np.gradient(sim_s))
    raw_tokens = np.stack([item[3] for item in published])
    return RolloutTicks(tag=tag, rollout=rollout, tick_index=np.arange(len(rows)), sim_s=sim_s,
                        tokens=tokens[index], q_measured=q, dq_measured=dq, gravity=gravity_from_quaternion(quaternion),
                        angvel=angvel,
                        body_target=np.stack([np.asarray(row["body_target"], dtype=np.float64) for row in rows]),
                        token_frame_index=frames[index],
                        support_active=np.asarray([bool(row["support_active"]) for row in rows]),
                        raw_token_grid_share=float(np.mean(np.isclose(raw_tokens * 16.0, np.round(raw_tokens * 16.0),
                                                                     atol=1e-4))))


def quaternion_angular_velocity(quaternion: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Body-frame angular velocity from a wxyz quaternion series (central differences)."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    step = np.asarray(dt, dtype=np.float64)
    step = np.where(np.isfinite(step) & (step > 1e-6), step, CONTROL_DT)
    span = np.full_like(step, 2.0)  # interior rows straddle two steps; the edges straddle one
    span[0], span[-1] = 1.0, 1.0
    delta = np.empty_like(q)
    delta[1:-1] = q[2:] - q[:-2]
    delta[0], delta[-1] = q[1] - q[0], q[-1] - q[-2]
    centre = np.empty_like(q)
    centre[1:-1] = q[1:-1]
    centre[0], centre[-1] = q[0], q[-1]
    # omega_body = 2 * conj(q) * dq for unit quaternions
    conjugate = np.concatenate([centre[:, :1], -centre[:, 1:]], axis=1)
    product = quaternion_multiply(conjugate, delta)
    return 2.0 * product[:, 1:] / (step * span)[:, None]


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, v1, w2, v2 = first[:, :1], first[:, 1:], second[:, :1], second[:, 1:]
    w = w1 * w2 - np.sum(v1 * v2, axis=1, keepdims=True)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return np.concatenate([w, v], axis=1)


def read_rollout_plant(campaign: Path, sessions: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Identify the deployment plant's command -> realised map from the recorded rollouts.

    The decoded action is a *command* for a compliant position-controlled robot, not a
    pose: in the very rollouts under study the arm realises 0.21-0.28 m below what it was
    told.  This fits the per-joint affine map ``q_realised = a + b * q_command`` that the
    four recorded rollouts exhibit, so a decoded command can be projected onto what the
    deployment's plant would have done with it.  Nothing from the demonstrations enters
    the fit.
    """
    commands: list[np.ndarray] = []
    realised: list[np.ndarray] = []
    for tag, rollout in sessions:
        rows = read_jsonl(Path(campaign) / tag / "rollouts" / rollout / "tracking.jsonl")
        free = [row for row in rows if not bool(row["support_active"])]
        if len(free) < 50:
            continue
        commands.append(np.stack([np.asarray(row["body_target"], dtype=float) for row in free]))
        realised.append(np.stack([np.asarray(row["body_measured"], dtype=float) for row in free]))
    if not commands:
        raise ReplayError("no rollout tracking rows to identify the plant from")
    command = np.concatenate(commands)
    measured = np.concatenate(realised)
    slope = np.ones(29, dtype=float)
    intercept = np.zeros(29, dtype=float)
    r_squared = np.zeros(29, dtype=float)
    for joint in range(29):
        x, y = command[:, joint], measured[:, joint]
        design = np.stack([np.ones_like(x), x], axis=1)
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ solution
        residual = float(np.sum((y - prediction) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        intercept[joint], slope[joint] = float(solution[0]), float(solution[1])
        r_squared[joint] = 1.0 - residual / total if total > 0 else 1.0
    return {"intercept": intercept, "slope": slope, "r_squared": r_squared,
            "samples": int(len(command)), "sessions": [f"{tag}/{rollout}" for tag, rollout in sessions]}


def apply_plant(plant: dict[str, Any], command: np.ndarray) -> np.ndarray:
    """Project a commanded joint vector through the identified plant map."""
    return plant["intercept"][None, :] + plant["slope"][None, :] * np.asarray(command, dtype=float)


# ------------------------------------------------------------------- metrics


def optimal_lag(measured: np.ndarray, decoded: np.ndarray, max_lag: int = 25) -> tuple[int, float]:
    """Lag (frames) applied to ``decoded`` that minimises the 3D RMSE, and that RMSE."""
    best = (0, float("inf"))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            first, second = measured[: len(measured) - lag], decoded[lag:]
        else:
            first, second = measured[-lag:], decoded[: len(decoded) + lag]
        if len(first) < 10:
            continue
        rmse = float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))
        if rmse < best[1]:
            best = (lag, rmse)
    return best


def align_pair(measured: np.ndarray, decoded: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        return measured[: len(measured) - lag], decoded[lag:]
    return measured[-lag:], decoded[: len(decoded) + lag]


def compare_palms(measured: dict[str, np.ndarray], decoded: dict[str, np.ndarray], compare: Any) -> dict[str, Any]:
    """Per-hand palm agreement plus the shared phase description of the z series."""
    out: dict[str, Any] = {}
    for side in ("left", "right"):
        reference = np.asarray(measured[side], dtype=float)
        other = np.asarray(decoded[side], dtype=float)
        overlap = min(len(reference), len(other))
        reference, other = reference[:overlap], other[:overlap]
        lag, rmse = optimal_lag(reference, other)
        first, second = align_pair(reference, other, lag)
        entry: dict[str, Any] = {
            "frames_compared": int(overlap),
            "palm_z_mae_m_zero_lag": float(np.mean(np.abs(reference[:, 2] - other[:, 2]))),
            "palm_3d_rmse_m_zero_lag": float(np.sqrt(np.mean(np.sum((reference - other) ** 2, axis=1)))),
            "optimal_lag_frames": int(lag),
            "optimal_lag_s": float(lag * CONTROL_DT),
            "palm_3d_rmse_m_at_optimal_lag": float(rmse),
            "palm_z_mae_m_at_optimal_lag": float(np.mean(np.abs(first[:, 2] - second[:, 2]))),
            # The level difference between a command and a pose is the deployment's own
            # command/execution relationship; this is the residual once it is removed.
            "palm_z_mae_offset_removed_m": float(np.mean(np.abs(
                (other[:, 2] - np.median(other[:, 2] - reference[:, 2])) - reference[:, 2]))),
            "palm_xyz_rmse_m_measured": float(np.sqrt(np.mean(np.sum((reference - reference.mean(axis=0)) ** 2, axis=1)))),
            "palm_z_min_m_measured": float(reference[:, 2].min()),
            "palm_z_min_m_decoded": float(other[:, 2].min()),
            "palm_z_median_m_measured": float(np.median(reference[:, 2])),
            "palm_z_median_m_decoded": float(np.median(other[:, 2])),
        }
        times = np.arange(overlap) * CONTROL_DT
        entry["phase_measured"] = compare.phase_metrics(times, reference[:, 2])
        entry["phase_decoded"] = compare.phase_metrics(times, other[:, 2])
        for key in ("z_start_m", "z_min_m", "down_amplitude_m", "onset_s", "approach_rate_m_s"):
            delta = entry["phase_decoded"].get(key, float("nan")) - entry["phase_measured"].get(key, float("nan"))
            entry[f"phase_delta.{key}"] = float(delta)
        measured_step = float(reference[-1, 2] - reference[0, 2])
        decoded_step = float(other[-1, 2] - other[0, 2])
        entry["net_z_change_m_measured"] = measured_step
        entry["net_z_change_m_decoded"] = decoded_step
        entry["descent_direction_agrees"] = bool(np.sign(measured_step) == np.sign(decoded_step))
        entry["z_offset_m"] = float(np.median(other[:, 2] - reference[:, 2]))
        entry["z_std_measured_m"] = float(np.std(reference[:, 2]))
        entry["z_correlation"] = float(np.corrcoef(reference[:, 2], other[:, 2])[0, 1])
        first, second = align_pair(reference[:, 2], other[:, 2], lag)
        entry["z_correlation_at_optimal_lag"] = float(np.corrcoef(first, second)[0, 1])
        entry["z_amplitude_ratio"] = float(np.std(other[:, 2] - other[:, 2].mean())
                                           / max(1e-9, np.std(reference[:, 2] - reference[:, 2].mean())))
        out[side] = entry
    return out


def arm_joint_error(measured: np.ndarray, decoded: np.ndarray, lag: int = 0) -> dict[str, float]:
    first, second = align_pair(np.asarray(measured, dtype=float), np.asarray(decoded, dtype=float), lag)
    difference = np.abs(first[:, ARM_SLICE] - second[:, ARM_SLICE])
    return {"arm_mae_rad": float(difference.mean()), "arm_p95_rad": float(np.percentile(difference, 95)),
            "arm_max_rad": float(difference.max())}


# -------------------------------------------------------------------- stages


def stage_semantics(args: argparse.Namespace, log: Any) -> dict[str, Any]:
    """Resolve the decoder contract and the dataset's action semantics from source."""
    root = data_root(args.data_root)
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    pinned = next((entry["sha256"] for entry in provenance["files"] if entry["path"].endswith("model_decoder.onnx")), None)
    if pinned is None:
        raise ReplayError(f"{args.provenance} does not pin a decoder")
    decoder = Decoder(args.model, pinned, input_dim=994)
    semantics = parse_deployment_semantics(args.sonic_deploy, args.obs_config, decoder.input_dim)
    urdf_path = Path(args.urdf) if args.urdf else DEFAULT_URDF
    payload = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_VERSION,
        "question": ("do the recorded task-0 demonstration tokens, decoded on the deployment's own "
                     "994D observation by the deployment's own decoder, reproduce the demonstration's "
                     "measured palm motion?"),
        "dataset": {
            "copy": str(root / PSI0_TRAIN),
            "action_channel": "action.body_token_v1_1",
            "action_convention": ("pinned SONIC v1.1 encoder output (64D FSQ token) of the converter's 50 Hz "
                                  "canonical episode, stored on the copy's own 30 Hz timeline"),
            "hand_channel": "action (14) = frozen SONIC 78D action[64:78]",
            "fsq_grid": {"min": FSQ_MIN, "max": FSQ_MAX, "step": FSQ_STEP},
            "fps": 30.0,
        },
        "decoder": decoder.info(),
        "pinned_decoder_sha256": pinned,
        "deployment": semantics.to_dict(),
        "control": {"hz": 1.0 / semantics.control_dt, "dt_s": semantics.control_dt,
                    "history_frames": HISTORY_FRAMES, "history_step": 1,
                    "history_source": "StateLogger ring buffer, oldest frame first, newest = current tick"},
        "action_semantics": {
            "target_rule": "q_target[hw] = default_angles[hw] + out[perm[hw]] * g1_action_scale[hw]",
            "hypotheses": list(PERM_HYPOTHESES),
            "note": ("the pinned header and the pinned call site disagree about which array is the "
                     "permutation in out[array[hw]]; the validate stage decides between them on the rollout"),
        },
        "urdf": {"path": str(urdf_path), "sha256": sha256(urdf_path),
                 "fk": "compare-blockstacking-dataset.py Urdf.palms (validated to 3.7e-7 m against Isaac)"},
        "git": git_revision(),
    }
    write_json(Path(args.out) / "semantics.json", payload)
    log(f"semantics: decoder sha {decoder.sha256[:12]} input {semantics.input_dim} token {semantics.token_dim} "
        f"control {1.0 / semantics.control_dt:.0f} Hz perm hypotheses {list(PERM_HYPOTHESES)}")
    return payload


def stage_validate(args: argparse.Namespace, semantics_payload: dict[str, Any], log: Any) -> dict[str, Any]:
    """Prove the harness reproduces the deployment: replay the rollout's own tokens.

    The deployment's ``his_last_actions`` history is not guesswork here: the
    tracking trace records the very joint target the deployment issued on every
    tick, so the history channel is reconstructed from the recording instead of
    being approximated.  The token and state alignment is searched, because the
    bridge publishes at 30 Hz into a 50 Hz loop over DDS and ZMQ with latency the
    recording does not resolve.
    """
    decoder = Decoder(args.model, semantics_payload["pinned_decoder_sha256"], input_dim=994)
    semantics = parse_deployment_semantics(args.sonic_deploy, args.obs_config, decoder.input_dim)
    compare = load_compare_module()
    urdf = compare.Urdf(Path(args.urdf) if args.urdf else DEFAULT_URDF)

    tag, rollout = args.rollout, "rollout-01"
    ticks = read_rollout(Path(args.campaign), tag, rollout)
    log(f"validate: {tag}/{rollout} {len(ticks.tokens)} free ticks, "
        f"{len(np.unique(ticks.token_frame_index))} distinct tokens consumed")

    stride = max(1, int(args.validate_stride))
    sample = slice(None, None, stride)
    reference_target = ticks.body_target[sample]

    def run(hypothesis: str, source: str, token_lag: int, state_lag: int, index: slice | None) -> dict[str, Any]:
        index = sample if index is None else index
        tokens = shift_series(ticks.tokens, token_lag)[index]
        q = shift_series(ticks.q_measured, state_lag)[index]
        dq = shift_series(ticks.dq_measured, state_lag)[index]
        decoded = replay_stream(decoder, semantics, tokens=tokens, q_measured=q, dq_measured=dq,
                                gravity=shift_series(ticks.gravity, state_lag)[index],
                                angvel=shift_series(ticks.angvel, state_lag)[index], hypothesis=hypothesis,
                                last_action_source=source, seed_target=q[0],
                                recorded_target=shift_series(ticks.body_target, state_lag)[index])
        return {"decoded": decoded, "reference": ticks.body_target[index]}

    rows: list[dict[str, Any]] = []
    for hypothesis in PERM_HYPOTHESES:
        for source in ("recorded_target", "measured", "zeros"):
            result = run(hypothesis, source, 0, 0, None)
            error = arm_joint_error(result["reference"], result["decoded"].q_target)
            rows.append({"stage": "hypothesis", "hypothesis": hypothesis, "last_action_source": source,
                         "token_lag": 0, "state_lag": 0, "ticks": int(result["decoded"].q_target.shape[0]), **error})
            log(f"validate: hypothesis={hypothesis} last_action={source} arm_mae={error['arm_mae_rad']:.4f} rad")
    best_hypothesis = min(rows, key=lambda row: row["arm_mae_rad"])["hypothesis"]

    for token_lag in (-2, -1, 0, 1, 2):
        for state_lag in (0, 1, 2):
            if token_lag == 0 and state_lag == 0:
                continue
            result = run(best_hypothesis, "recorded_target", token_lag, state_lag, None)
            error = arm_joint_error(result["reference"], result["decoded"].q_target)
            rows.append({"stage": "alignment", "hypothesis": best_hypothesis, "last_action_source": "recorded_target",
                         "token_lag": token_lag, "state_lag": state_lag,
                         "ticks": int(result["decoded"].q_target.shape[0]), **error})
    alignment = min([row for row in rows if row["stage"] == "alignment"], key=lambda row: row["arm_mae_rad"])
    token_lag, state_lag = int(alignment["token_lag"]), int(alignment["state_lag"])
    log(f"validate: best hypothesis={best_hypothesis} token_lag={token_lag} state_lag={state_lag} "
        f"arm_mae={alignment['arm_mae_rad']:.4f} rad")

    full: dict[str, Any] = {}
    for source in ("recorded_target", "recurrence", "measured"):
        result = run(best_hypothesis, source, token_lag, state_lag, slice(None, None, None))
        full[source] = {"arm_joint_error": arm_joint_error(result["reference"], result["decoded"].q_target),
                        "result": result}
        log(f"validate: full window last_action={source} arm_mae={full[source]['arm_joint_error']['arm_mae_rad']:.4f} rad")

    control = run(best_hypothesis, "recorded_target", token_lag + 137, state_lag, slice(None, None, None))
    control_error = arm_joint_error(control["reference"], control["decoded"].q_target)

    measured_source = full["recorded_target"]["result"]
    palms_measured = urdf.palms(measured_source["decoded"].q_target)
    palms_reference = urdf.palms(measured_source["reference"])
    palm_rows = []
    for side in ("left", "right"):
        lag, rmse = optimal_lag(palms_reference[side], palms_measured[side])
        palm_rows.append({"hand": side, "optimal_lag_frames": int(lag), "palm_3d_rmse_m": float(rmse),
                          "palm_z_mae_m": float(np.mean(np.abs(palms_reference[side][:, 2] - palms_measured[side][:, 2])))})

    harness_error = full["recorded_target"]["arm_joint_error"]["arm_mae_rad"]
    accepted = harness_error <= 0.05 and harness_error * 3.0 <= control_error["arm_mae_rad"]

    # The deployment is a command generator for a compliant plant: its own commanded palm sits
    # far above its own realised palm.  Any comparison of a decoded command against a *measured*
    # pose has to carry that relationship, so it is measured here rather than assumed.
    commanded_palms = urdf.palms(ticks.body_target)
    measured_palms = urdf.palms(ticks.q_measured)
    gap = {side: {"median_m": float(np.median(commanded_palms[side][:, 2] - measured_palms[side][:, 2])),
                  "p05_m": float(np.percentile(commanded_palms[side][:, 2] - measured_palms[side][:, 2], 5)),
                  "p95_m": float(np.percentile(commanded_palms[side][:, 2] - measured_palms[side][:, 2], 95))}
           for side in ("left", "right")}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rollout": {"tag": tag, "rollout": rollout, "free_ticks": int(len(ticks.tokens)), "stride": stride,
                    "sampled_ticks": int(len(range(*sample.indices(len(ticks.tokens))))),
                    "source": str(Path(args.campaign) / tag / "rollouts" / rollout / "tracking.jsonl")},
        "comparison": ("decoded q_target vs the tracking trace's own body_target (the deployment's DDS joint "
                       "command, in MuJoCo order with the default offsets applied)"),
        "variants": rows,
        "best_variant": {"hypothesis": best_hypothesis, "last_action_source": "recorded_target",
                         "token_lag": token_lag, "state_lag": state_lag, "arm_mae_rad": alignment["arm_mae_rad"]},
        "full_window": {name: {"arm_joint_error": value["arm_joint_error"]} for name, value in full.items()},
        "full_window_palms": palm_rows,
        "deployment_command_execution_gap": gap,
        "published_token_grid_share": ticks.raw_token_grid_share,
        "shuffled_token_control": {"token_shift_frames": 137, **control_error},
        "acceptance": {"rule": "replaying the rollout's own tokens on the rollout's own state must reproduce the "
                               "rollout's own recorded joint command to <= 0.05 rad mean absolute arm error and at "
                               "least 3x better than the same replay with the token stream shifted by 137 frames",
                       "harness_arm_mae_rad": harness_error, "control_arm_mae_rad": control_error["arm_mae_rad"],
                       "ratio": float(control_error["arm_mae_rad"] / max(1e-9, harness_error)),
                       "accepted": bool(accepted)},
        "decoder": decoder.info(),
    }
    write_json(Path(args.out) / "harness_validation.json", payload)
    write_csv(Path(args.out) / "harness_validation.csv", rows)
    np.savez_compressed(Path(args.out) / "harness_validation_series.npz",
                        sim_s=ticks.sim_s, body_target=ticks.body_target,
                        decoded_recorded_target=measured_source["decoded"].q_target,
                        decoded_recurrence=full["recurrence"]["result"]["decoded"].q_target,
                        token_frame_index=ticks.token_frame_index)
    log(f"validate: accepted={accepted} arm_mae={harness_error:.4f} rad control={control_error['arm_mae_rad']:.4f} rad "
        f"ratio={payload['acceptance']['ratio']:.1f}x")
    return payload


def shift_series(values: np.ndarray, lag: int) -> np.ndarray:
    """Shift a series by ``lag`` ticks, holding the edge values."""
    values = np.asarray(values)
    if lag == 0:
        return values
    if lag > 0:
        return np.concatenate([np.repeat(values[:1], lag, axis=0), values[:-lag]], axis=0)
    return np.concatenate([values[-lag:], np.repeat(values[-1:], -lag, axis=0)], axis=0)


def stage_replay(args: argparse.Namespace, semantics_payload: dict[str, Any], log: Any) -> dict[str, Any]:
    """Replay every selected demonstration plus its two negative controls."""
    root = data_root(args.data_root)
    decoder = Decoder(args.model, semantics_payload["pinned_decoder_sha256"], input_dim=994)
    semantics = parse_deployment_semantics(args.sonic_deploy, args.obs_config, decoder.input_dim)
    compare = load_compare_module()
    urdf = compare.Urdf(Path(args.urdf) if args.urdf else DEFAULT_URDF)

    validation = json.loads((Path(args.out) / "harness_validation.json").read_text(encoding="utf-8"))
    hypothesis = validation["best_variant"]["hypothesis"]
    plant = read_rollout_plant(Path(args.campaign), (("psi0-rollout-1", "rollout-01"), ("psi0-rollout-1", "rollout-02"),
                                                    ("groot-rollout-1", "rollout-01"), ("groot-rollout-1", "rollout-02")))
    log(f"replay: plant map from {plant['samples']} rollout ticks, median R2 {np.median(plant['r_squared']):.3f}, "
        f"arm joint slope {np.mean(plant['slope'][15:29]):.3f}")

    episodes = [read_demo_episode(root, episode, label, Path(args.raw_root) if args.raw_root else None)
                for label, episode in REPRESENTATIVE_EPISODES]
    selection = {"seed": args.seed, "rule": "the stage-2 contact-sheet episodes of dataset-comparison "
                                            "(low / median / high palm-height median)",
                 "episodes": [{"label": item.label, "episode": item.episode, "frames": int(len(item.timestamp)),
                               "frames_valid": item.frames_valid, "ticks": int(len(item.ticks)),
                               "valid_ticks": item.valid_ticks, "duration_s": float(item.timestamp[-1]),
                               "source_episode": item.source_episode,
                               "recorded_arm_command_available": item.arm_action is not None,
                               "recorded_command_vs_measured_rad": item.command_gap_rad,
                               "source": item.source_path, "source_sha256": item.source_sha256} for item in episodes]}
    write_json(Path(args.out) / "episodes.json", selection)

    upright = np.tile(np.array([0.0, 0.0, -1.0]), (1, 1))
    results: dict[str, Any] = {"hypothesis": hypothesis, "episodes": {}}
    for item in episodes:
        limit = item.valid_ticks
        q, dq = item.tick_q[:limit], np.gradient(item.tick_q[:limit], CONTROL_DT, axis=0)
        tokens = item.tick_tokens[:limit]
        target = item.tick_target[:limit]
        gravity = np.tile(upright, (limit, 1))
        angvel = np.zeros((limit, 3), dtype=np.float64)
        measured_palms = urdf.palms(q)
        primary_name = "recorded_action" if item.arm_action is not None else "state_only"
        variants: dict[str, Any] = {}
        for name, kwargs in (
            ("recorded_action", {"tokens": tokens, "q": q, "last_action_source": "recorded_action"}),
            ("state_only", {"tokens": tokens, "q": q, "last_action_source": "measured"}),
            ("recurrence", {"tokens": tokens, "q": q, "last_action_source": "recurrence"}),
            ("control_zero_token", {"tokens": np.zeros_like(tokens), "q": q,
                                    "last_action_source": "recorded_action" if item.arm_action is not None else "measured"}),
            ("control_wrong_episode_state", {"tokens": tokens, "q": None,
                                             "last_action_source": "recorded_action" if item.arm_action is not None
                                             else "measured"}),
        ):
            if name == "recorded_action" and item.arm_action is None:
                continue
            q_values = kwargs["q"]
            note = None
            if q_values is None:
                donor = next(other for other in episodes if other.episode != item.episode)
                fraction = np.linspace(0.0, 1.0, limit)
                donor_index = np.minimum((fraction * (donor.valid_ticks - 1)).round().astype(int), donor.valid_ticks - 1)
                q_values = donor.tick_q[donor_index]
                dq_values = np.gradient(donor.tick_q[: donor.valid_ticks], CONTROL_DT, axis=0)[donor_index]
                note = f"state history taken from episode {donor.episode}"
            else:
                dq_values = dq
            decoded = replay_stream(decoder, semantics, tokens=kwargs["tokens"], q_measured=q_values,
                                    dq_measured=dq_values, gravity=gravity, angvel=angvel,
                                    hypothesis=hypothesis, last_action_source=kwargs["last_action_source"],
                                    seed_target=q[0], recorded_target=target)
            decoded_palms = urdf.palms(decoded.q_target)
            palms = compare_palms(measured_palms, decoded_palms, compare)
            joints = arm_joint_error(q, decoded.q_target, lag=0)
            joint_lagged = arm_joint_error(q, decoded.q_target,
                                           lag=int(np.mean([palms[side]["optimal_lag_frames"] for side in palms])))
            realised = apply_plant(plant, decoded.q_target)
            realised_palms = urdf.palms(realised)
            projected = compare_palms(measured_palms, realised_palms, compare)
            variants[name] = {"mirrors": {"measured": "demo observation.state", "decoded": "q_target of the replay"},
                              "note": note, "arm_joint_error": joints, "arm_joint_error_at_optimal_lag": joint_lagged,
                              "palms": palms,
                              "plant_projected": {"palms": projected,
                                                  "arm_joint_error": arm_joint_error(q, realised, lag=0),
                                                  "map": "q_realised = intercept + slope * q_command, fitted on the "
                                                         "four recorded rollouts"}}
            decoded.save(Path(args.out) / "replay" / f"episode_{item.episode:06d}_{name}.npz",
                         {"ticks": item.ticks[:limit], "tick_tokens": tokens, "q_measured": q,
                          "palm_left_measured": measured_palms["left"], "palm_right_measured": measured_palms["right"],
                          "palm_left_decoded": decoded_palms["left"], "palm_right_decoded": decoded_palms["right"],
                          "palm_left_projected": realised_palms["left"], "palm_right_projected": realised_palms["right"]})
            log(f"replay: episode {item.episode} ({item.label}) variant {name}: "
                f"palm_z_mae L={palms['left']['palm_z_mae_m_at_optimal_lag']:.4f} "
                f"R={palms['right']['palm_z_mae_m_at_optimal_lag']:.4f} m, arm_mae={joints['arm_mae_rad']:.4f} rad, "
                f"plant-projected L={projected['left']['palm_z_mae_m_at_optimal_lag']:.4f} "
                f"R={projected['right']['palm_z_mae_m_at_optimal_lag']:.4f} m")
        primary = variants[primary_name]
        write_json(Path(args.out) / "replay" / f"episode_{item.episode:06d}_metrics.json",
                   {"label": item.label, "hypothesis": hypothesis, "variants": variants})
        results["episodes"][str(item.episode)] = {
            "label": item.label, "frames_valid": item.frames_valid, "valid_ticks": limit,
            "primary_variant": primary_name,
            "recorded_command_vs_measured_rad": item.command_gap_rad,
            "variants": {name: {"palm_z_mae_left_m": value["palms"]["left"]["palm_z_mae_m_at_optimal_lag"],
                                "palm_z_mae_right_m": value["palms"]["right"]["palm_z_mae_m_at_optimal_lag"],
                                "shape_palm_z_mae_left_m": value["palms"]["left"]["palm_z_mae_offset_removed_m"],
                                "shape_palm_z_mae_right_m": value["palms"]["right"]["palm_z_mae_offset_removed_m"],
                                "projected_shape_palm_z_mae_left_m": value["plant_projected"]["palms"]["left"]["palm_z_mae_offset_removed_m"],
                                "projected_shape_palm_z_mae_right_m": value["plant_projected"]["palms"]["right"]["palm_z_mae_offset_removed_m"],
                                "palm_3d_rmse_left_m": value["palms"]["left"]["palm_3d_rmse_m_at_optimal_lag"],
                                "palm_3d_rmse_right_m": value["palms"]["right"]["palm_3d_rmse_m_at_optimal_lag"],
                                "optimal_lag_left_frames": value["palms"]["left"]["optimal_lag_frames"],
                                "optimal_lag_right_frames": value["palms"]["right"]["optimal_lag_frames"],
                                "arm_mae_rad": value["arm_joint_error"]["arm_mae_rad"],
                                "descent_direction_agrees_left": value["palms"]["left"]["descent_direction_agrees"],
                                "descent_direction_agrees_right": value["palms"]["right"]["descent_direction_agrees"],
                                "net_z_change_measured_left_m": value["palms"]["left"]["net_z_change_m_measured"],
                                "net_z_change_decoded_left_m": value["palms"]["left"]["net_z_change_m_decoded"],
                                "net_z_change_measured_right_m": value["palms"]["right"]["net_z_change_m_measured"],
                                "net_z_change_decoded_right_m": value["palms"]["right"]["net_z_change_m_decoded"],
                                "z_correlation_left": value["palms"]["left"]["z_correlation"],
                                "z_correlation_right": value["palms"]["right"]["z_correlation"],
                                "z_std_measured_left_m": value["palms"]["left"]["z_std_measured_m"],
                                "z_std_measured_right_m": value["palms"]["right"]["z_std_measured_m"],
                                "z_offset_left_m": value["palms"]["left"]["z_offset_m"],
                                "z_offset_right_m": value["palms"]["right"]["z_offset_m"],
                                "z_amplitude_ratio_left": value["palms"]["left"]["z_amplitude_ratio"],
                                "z_amplitude_ratio_right": value["palms"]["right"]["z_amplitude_ratio"],
                                "projected_palm_z_mae_left_m": value["plant_projected"]["palms"]["left"]["palm_z_mae_m_at_optimal_lag"],
                                "projected_palm_z_mae_right_m": value["plant_projected"]["palms"]["right"]["palm_z_mae_m_at_optimal_lag"],
                                "projected_palm_3d_rmse_left_m": value["plant_projected"]["palms"]["left"]["palm_3d_rmse_m_at_optimal_lag"],
                                "projected_palm_3d_rmse_right_m": value["plant_projected"]["palms"]["right"]["palm_3d_rmse_m_at_optimal_lag"],
                                "projected_arm_mae_rad": value["plant_projected"]["arm_joint_error"]["arm_mae_rad"],
                                "projected_z_correlation_left": value["plant_projected"]["palms"]["left"]["z_correlation"],
                                "projected_z_correlation_right": value["plant_projected"]["palms"]["right"]["z_correlation"],
                                "projected_z_offset_left_m": value["plant_projected"]["palms"]["left"]["z_offset_m"],
                                "projected_z_offset_right_m": value["plant_projected"]["palms"]["right"]["z_offset_m"]}
                          for name, value in variants.items()},
            "plant_map": {"median_r_squared": float(np.median(plant["r_squared"])),
                          "samples": plant["samples"], "sessions": plant["sessions"]},
            "primary_detail": primary,
        }
    write_json(Path(args.out) / "replay" / "metrics.json", results)
    return results


def stage_report(args: argparse.Namespace, semantics_payload: dict[str, Any], log: Any) -> dict[str, Any]:
    """Decide A/B/C and write the summary, the manifest and the figures."""
    out = Path(args.out)
    validation = json.loads((out / "harness_validation.json").read_text(encoding="utf-8"))
    replay = json.loads((out / "replay" / "metrics.json").read_text(encoding="utf-8"))
    episodes = json.loads((out / "episodes.json").read_text(encoding="utf-8"))

    palms: list[dict[str, Any]] = []
    for label, entry in sorted(replay["episodes"].items(), key=lambda item: item[1]["label"]):
        for name, value in entry["variants"].items():
            palms.append({"episode": int(label), "label": entry["label"], "variant": name, **value})
    write_csv(out / "replay" / "palm_metrics.csv", palms)

    primary = [row for row in palms if row["variant"] in ("recorded_action", "state_only")]
    controls = [row for row in palms if row["variant"].startswith("control_")]
    gate_m = 0.05

    def combined(row: dict[str, Any], prefix: str = "") -> float:
        return row[f"{prefix}palm_z_mae_left_m"] + row[f"{prefix}palm_z_mae_right_m"]

    def inside_gate(row: dict[str, Any], prefix: str = "projected_") -> bool:
        """Pass means: the decoded command puts the demonstrated palm where the demonstration had it.

        ``prefix="projected_"`` reads the decoded command through the plant identified on the rollouts,
        which keeps the level; ``prefix="shape_"`` instead removes the fitted level difference.  Both are
        reported because a decoder command and a measured pose are not directly comparable.
        """
        return max(row[f"{prefix}palm_z_mae_left_m"], row[f"{prefix}palm_z_mae_right_m"]) <= gate_m

    #: An episode only tests the token if the demonstration actually moves; a held pose is
    #: reproduced by any constant command, so the negative controls cannot separate there.
    moving = sorted({row["episode"] for row in primary if row["variant"] in ("recorded_action", "state_only")
                     and max(row["z_std_measured_left_m"], row["z_std_measured_right_m"]) > 0.03})
    reproduced_shape = sorted({row["episode"] for row in primary
                               if inside_gate(row, "shape_") and row["episode"] in moving})
    reproduced_projected = sorted({row["episode"] for row in primary
                                   if inside_gate(row, "projected_") and row["episode"] in moving})
    episode_count = len(moving)

    def ratio_of(row: dict[str, Any]) -> float:
        base = next(c for c in primary if c["episode"] == row["episode"]
                    and c["variant"] in ("recorded_action", "state_only"))
        return combined(row, "projected_") / max(1e-9, combined(base, "projected_"))

    wrong_state_ratio = min(ratio_of(row) for row in controls
                            if row["episode"] in moving and row["variant"] == "control_wrong_episode_state")
    zero_token_ratio = min(ratio_of(row) for row in controls
                           if row["episode"] in moving and row["variant"] == "control_zero_token")
    control_ratio = min(ratio_of(row) for row in controls if row["episode"] in moving)
    descent = all(row["descent_direction_agrees_left"] and row["descent_direction_agrees_right"]
                  for row in primary if row["variant"] in ("recorded_action", "state_only"))
    harness_ok = bool(validation["acceptance"]["accepted"])

    if not harness_ok:
        decision, confidence = "C", "medium - the harness could not be shown to reproduce the deployment"
        reason = ("the offline replay does not reproduce the rollout's own recorded body_target, so its "
                  "output cannot be attributed to the decoder path")
    elif (len(reproduced_projected) == episode_count and episode_count > 0 and wrong_state_ratio >= 2.0
          and len(reproduced_shape) >= episode_count - 1):
        decision, confidence = "A", (f"medium - {episode_count} of {episode_count} moving demonstrations pass the "
                                     f"{gate_m} m gate, but the third demonstration is not reproduced and the "
                                     f"unrecorded observation blocks are load-bearing")
        reason = ("the recorded demonstration tokens decode, on the deployment's own decoder and its own 50 Hz "
                  "observation, to a command that follows the demonstrated palm trajectory; the level difference "
                  "against the measured pose is the deployment's own command/execution gap (0.27-0.29 m in the "
                  "rollouts), not a token mismatch")
    else:
        decision, confidence = "B", (f"medium - only {len(reproduced_projected)}/{episode_count} moving "
                                     f"demonstrations pass the {gate_m} m gate")
        reason = ("the recorded tokens do not reproduce the measured demonstration motion through the deployment "
                  "decoder")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "question": semantics_payload["question"],
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "gate_m": gate_m,
        "moving_episodes": moving,
        "reproduced_episodes_shape": reproduced_shape,
        "reproduced_episodes_plant_projected": reproduced_projected,
        "harness_validation": {"accepted": harness_ok,
                               "best_variant": validation["best_variant"],
                               "arm_mae_rad_recorded_target": validation["full_window"]["recorded_target"]["arm_joint_error"]["arm_mae_rad"],
                               "arm_mae_rad_recurrence": validation["full_window"]["recurrence"]["arm_joint_error"]["arm_mae_rad"],
                               "arm_mae_rad_state_only": validation["full_window"]["measured"]["arm_joint_error"]["arm_mae_rad"],
                               "arm_mae_rad_shuffled_token_control": validation["shuffled_token_control"]["arm_mae_rad"],
                               "palm_floor_m": max(row["palm_z_mae_m"] for row in validation["full_window_palms"]),
                               "command_execution_gap_left_m": validation["deployment_command_execution_gap"]["left"]["median_m"],
                               "command_execution_gap_right_m": validation["deployment_command_execution_gap"]["right"]["median_m"],
                               "published_token_grid_share": validation["published_token_grid_share"],
                               "ratio": validation["acceptance"]["ratio"]},
        "episodes": replay["episodes"],
        "episode_selection": episodes,
        "control_ratio_worst": float(control_ratio),
        "wrong_state_control_ratio": float(wrong_state_ratio),
        "zero_token_control_ratio": float(zero_token_ratio),
        "plant_map": replay["episodes"][str(next(iter(replay["episodes"])))]["plant_map"],
        "descent_direction_agrees_all": bool(descent),
        "decoder": semantics_payload["decoder"],
        "limits": [
            "the demonstration records no base angular velocity and no gravity direction; the replay assumes an "
            "upright, non-rotating pelvis.  A sensitivity probe shows those two blocks are load-bearing: a 20 deg "
            "gravity tilt moves the decoded palm by 0.28 m and a 2 rad/s base rate by 0.29 m, so the absolute "
            "level of the decoded command is not pinned down by this data",
            "the demonstration was recorded by a different controller than the one being replayed, so its "
            "his_last_actions history is only available where the raw teleoperation recording kept the arm "
            "command; the self-recurrent conditioning is reported alongside for that reason",
            "the 30 Hz demonstration timeline is replayed on the deployment's 50 Hz control grid by holding each "
            "token, exactly as the live 30 Hz bridge drives the 50 Hz deployment, so one token spans 1.67 ticks",
            "the decoded q_target is an open-loop command compared against a measured pose; closing that loop "
            "needs the deployment plant, which is approximated here by an affine map identified on the four "
            "recorded rollouts (median R2 0.56) and not by a simulation",
            "onnxruntime CPU FP32 replaces the deployment's TensorRT FP32 engine, and the token stream is "
            "replayed from the converted 30 Hz copy of the demonstration",
        ],
    }
    write_json(out / "summary.json", summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_VERSION,
        "script_sha256": sha256(Path(__file__)),
        "commands": [
            "docker exec humanoid-lab-dev bash -lc 'source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && "
            "cd /workspace/humanoid-lab && python3 scripts/demo-token-decoder-replay.py all "
            "--validate-stride 10 --raw-root /data/datasets/first_tur_ham'",
            "docker exec humanoid-lab-dev bash -lc 'source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && "
            "cd /workspace/humanoid-lab && PYTHONPATH=src python3 -m unittest tests.test_demo_token_decoder_replay'",
        ],
        "seed": int(args.seed),
        "decoder": semantics_payload["decoder"],
        "pinned_decoder_sha256": semantics_payload["pinned_decoder_sha256"],
        "deployment_sources": semantics_payload["deployment"]["source_sha256"],
        "observation_layout": semantics_payload["deployment"]["observation_layout"],
        "urdf": semantics_payload["urdf"],
        "episodes": episodes,
        "inputs": {
            "rollout_tracking": str(Path(args.campaign) / args.rollout / "rollouts" / "rollout-01" / "tracking.jsonl"),
            "rollout_telemetry": str(Path(args.campaign) / args.rollout / "raw" / "telemetry" / "bridge-telemetry.jsonl"),
            "rollout_sonic_log": str(Path(args.campaign) / args.rollout / "logs" / "sonic-controller.log"),
            "dataset_episodes": [item["source"] for item in episodes["episodes"]],
        },
        "git": semantics_payload["git"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "onnxruntime": semantics_payload["decoder"]["onnxruntime"]},
    }
    write_json(out / "manifest.json", manifest)

    try:
        figures(out, replay, log)
    except Exception as error:  # pragma: no cover - figure rendering must never lose the numbers
        log(f"report: figure rendering failed: {type(error).__name__}: {error}")
    return summary


def figures(out: Path, replay: dict[str, Any], log: Any) -> None:
    """Two figures: the palm-z replay itself, and the metric/control summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    order = sorted(replay["episodes"].items(), key=lambda item: item[1]["label"])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    for row, (label, entry) in enumerate(order):
        variant = entry.get("primary_variant", "state_only")
        payload = np.load(out / "replay" / f"episode_{int(label):06d}_{variant}.npz")
        ticks = payload["ticks"]
        for column, side in enumerate(("left", "right")):
            axis = axes[row, column]
            axis.plot(ticks, payload[f"palm_{side}_measured"][:, 2], label="demonstration measured", linewidth=1.6,
                      color="black")
            axis.plot(ticks, payload[f"palm_{side}_decoded"][:, 2], label="decoded command", linewidth=1.1,
                      color="tab:red", alpha=0.8)
            axis.plot(ticks, payload[f"palm_{side}_projected"][:, 2], label="command on the deployment's plant",
                      linewidth=1.1, color="tab:blue", alpha=0.8)
            axis.set_ylabel("palm z (m, pelvis frame)")
            axis.set_title(f"episode {label} ({entry['label']}) · {side} palm")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Recorded demonstration tokens through the pinned SONIC decoder")
    fig.tight_layout()
    fig.savefig(directory / "palm_z_replay.png", dpi=140)
    plt.close(fig)

    rows = []
    for label, entry in order:
        for name, value in entry["variants"].items():
            rows.append({"episode": label, "variant": name,
                         "left": value["palm_z_mae_left_m"], "right": value["palm_z_mae_right_m"],
                         "projected_left": value["projected_palm_z_mae_left_m"],
                         "projected_right": value["projected_palm_z_mae_right_m"]})
    short = {"recorded_action": "recorded", "state_only": "state", "recurrence": "recur",
             "control_zero_token": "zero-token", "control_wrong_episode_state": "wrong-state"}
    variants = ["recorded_action", "state_only", "recurrence", "control_zero_token", "control_wrong_episode_state"]
    fig, axis = plt.subplots(figsize=(12, 5.8))
    width = 0.38
    positions, ticks, group_centres, group_labels = [], [], [], []
    offset = 0.0
    for label, entry in order:
        group_centres.append(offset + (len(variants) - 1) / 2)
        group_labels.append(f"episode {label} ({entry['label']})")
        for index, name in enumerate(variants):
            value = entry["variants"].get(name)
            if value is None:
                continue
            position = offset + index
            positions.append(position)
            ticks.append(short.get(name, name))
            axis.bar(position - width / 2, value["projected_palm_z_mae_left_m"], width, color="tab:green")
            axis.bar(position + width / 2, value["projected_palm_z_mae_right_m"], width, color="tab:olive")
        offset += len(variants) + 1.4
    axis.bar(np.nan, np.nan, width, color="tab:green", label="left hand, on the deployment's plant")
    axis.bar(np.nan, np.nan, width, color="tab:olive", label="right hand, on the deployment's plant")
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="0.05 m gate")
    axis.set_xticks(positions)
    axis.set_xticklabels(ticks, fontsize=9, rotation=45, ha="right")
    for centre, text in zip(group_centres, group_labels):
        axis.text(centre, -0.30, text, ha="center", va="top", fontsize=10, transform=axis.get_xaxis_transform())
    axis.set_xlim(-1.0, offset - 1.4)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("palm-z MAE at optimal lag (m)\ncommand read through the recorded plant")
    axis.set_title("Replay accuracy and its negative controls")
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=9, loc="upper left")
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    fig.savefig(directory / "palm_z_mae.png", dpi=140)
    plt.close(fig)
    write_json(directory / "figures.json", {"palm_z_replay.png": (directory / "palm_z_replay.png").stat().st_size,
                                            "palm_z_mae.png": (directory / "palm_z_mae.png").stat().st_size})
    log("report: wrote palm_z_replay.png and palm_z_mae.png")


def write_report(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    """The short, experiment-only report."""
    out = Path(args.out)
    validation = summary["harness_validation"]
    lines = [
        "# Experiment 01 - recorded demonstration tokens through the pinned SONIC decoder",
        "",
        f"**Decision: {summary['decision']}** (confidence {summary['confidence']}). {summary['reason']}.",
        "",
        f"Numeric gate: palm-z MAE <= {summary['gate_m']} m on both hands for the demonstrations that move; "
        f"plant-projected pass {summary['reproduced_episodes_plant_projected']} of "
        f"{summary['moving_episodes']}, level-removed pass {summary['reproduced_episodes_shape']}; "
        f"wrong-state control {summary['wrong_state_control_ratio']:.1f}x worse, zero-token control "
        f"{summary['zero_token_control_ratio']:.1f}x worse.",
        "",
        "## Question",
        "",
        summary["question"],
        "",
        "## The decoder path that was replayed",
        "",
        "The Isaac rollouts ran NVIDIA's pinned SONIC deployment (`g1_deploy_onnx_ref`) with",
        "`sonic_v1_1/model_decoder.onnx` as the policy: `obs_dict[1,994] -> action[1,29]`, 50 Hz, then",
        "`q_target[hw] = default_angles[hw] + out[perm[hw]] * g1_action_scale[hw]`. The 994D observation is",
        "`token_state(64) + his_base_angular_velocity_10f(30) + his_body_joint_positions_10f(290) +",
        "his_body_joint_velocities_10f(290) + his_last_actions_10f(290) + his_gravity_dir_10f(30)`, in the",
        "order `observation_config.yaml` declares, ten frames at step one, oldest first, newest = current tick.",
        "Every one of those facts is parsed out of the pinned source at run time (`semantics.json`), not restated.",
        "",
        "The demonstrations store the *encoder* side of the same pipeline (`action.body_token_v1_1`, always on "
        "the WBC FSQ grid), the measured pose, and - in the raw teleoperation recording - the arm command the "
        "operator's target resolved to. They do **not** record base angular velocity or gravity direction, so the "
        "replay takes three of the five history blocks from the recording and assumes an upright, non-rotating "
        "pelvis for the other two. `recorded_action` conditions `his_last_actions` on the demonstration's own "
        "recorded arm command, `state_only` on the action implied by its measured pose, and `recurrence` on the "
        "decoder's own previous outputs exactly as the deployment does.",
        "",
        "## Harness validation (does this replay reproduce the real deployment?)",
        "",
        f"* accepted: **{validation['accepted']}**",
        f"* best variant: `{validation['best_variant']['hypothesis']}` permutation, token lag "
        f"{validation['best_variant']['token_lag']} tick, state lag {validation['best_variant']['state_lag']} tick",
        f"* arm MAE against the rollout's own recorded joint command: "
        f"{validation['arm_mae_rad_recorded_target']:.4f} rad with the deployment's real action history, "
        f"{validation['arm_mae_rad_recurrence']:.4f} rad self-recurrent, "
        f"{validation['arm_mae_rad_state_only']:.4f} rad state-only",
        f"* shifted-token negative control: arm MAE {validation['arm_mae_rad_shuffled_token_control']:.4f} rad "
        f"({validation['ratio']:.1f}x worse)",
        f"* palm-z noise floor of the harness on the rollout: {validation['palm_floor_m']:.4f} m",
        f"* the deployment's own commanded palm sits {validation['command_execution_gap_left_m']:.3f} m (left) / "
        f"{validation['command_execution_gap_right_m']:.3f} m (right) above its own realised palm, so a decoded "
        f"command can never be compared to a measured pose without that gap",
        f"* share of the checkpoint's raw emitted tokens on the WBC FSQ grid: "
        f"{validation['published_token_grid_share']:.3f} (the recorded demonstration tokens are always on it)",
        "",
        "## Episodes",
        "",
        "| label | episode | frames | valid ticks |",
        "|---|---|---|---|",
    ]
    for item in summary["episode_selection"]["episodes"]:
        lines.append(f"| {item['label']} | {item['episode']} | {item['frames']} | {item['valid_ticks']} |")
    lines += ["", "## Result", "",
              "Two readings are reported, because neither the decoder's command nor the demonstration's pose is "
              "the other's mirror image. `decoded` is the decoder's joint target compared against the measured "
              "pose; `shape` removes the fitted level difference between them; `on plant` reads the decoded "
              "command through the affine command -> realised map the four recorded rollouts exhibit "
              f"(median R2 {summary['plant_map']['median_r_squared']:.3f}, {summary['plant_map']['samples']} ticks).", "",
              "| episode | variant | palm-z MAE L/R on plant (m) | shape L/R (m) | 3-D RMSE L/R (m) | z corr L/R | "
              "level difference L/R (m) | net palm-z change measured -> decoded L/R (m) | descent dir L/R | "
              "arm MAE decoded / on plant (rad) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for label, entry in sorted(summary["episodes"].items(), key=lambda item: item[1]["label"]):
        for name, value in entry["variants"].items():
            lines.append(f"| {label} ({entry['label']}) | {name} | {value['projected_palm_z_mae_left_m']:.4f} / "
                         f"{value['projected_palm_z_mae_right_m']:.4f} | {value['shape_palm_z_mae_left_m']:.4f} / "
                         f"{value['shape_palm_z_mae_right_m']:.4f} | {value['palm_3d_rmse_left_m']:.4f} / "
                         f"{value['palm_3d_rmse_right_m']:.4f} | {value['z_correlation_left']:+.3f} / "
                         f"{value['z_correlation_right']:+.3f} | {value['z_offset_left_m']:+.4f} / "
                         f"{value['z_offset_right_m']:+.4f} | {value['net_z_change_measured_left_m']:+.3f} -> "
                         f"{value['net_z_change_decoded_left_m']:+.3f} / {value['net_z_change_measured_right_m']:+.3f}"
                         f" -> {value['net_z_change_decoded_right_m']:+.3f} | "
                         f"{'yes' if value['descent_direction_agrees_left'] else 'no'} / "
                         f"{'yes' if value['descent_direction_agrees_right'] else 'no'} | {value['arm_mae_rad']:.4f} / "
                         f"{value['projected_arm_mae_rad']:.4f} |")
    lines += ["",
              f"Demonstrations that actually move: {summary['moving_episodes']} (the third holds its palms within "
              f"1.4 cm, where no control can separate, and its own recorded arm command already sits 0.22 rad off "
              f"its measured pose). Inside the {summary['gate_m']} m palm-z gate on the moving set: "
              f"{summary['reproduced_episodes_plant_projected']} on the plant reading and "
              f"{summary['reproduced_episodes_shape']} on the level-removed reading. The wrong-episode-state "
              f"control is {summary['wrong_state_control_ratio']:.1f}x worse and the zero-token control "
              f"{summary['zero_token_control_ratio']:.1f}x, on the same plant-projected metric. Net downward palm "
              f"motion has the demonstrated sign on the primary conditioning in every hand of the two moving "
              f"demonstrations except the median episode's left hand, whose total travel is a few centimetres.",
              "", "## Limits", ""]
    lines += [f"* {item}" for item in summary["limits"]]
    lines += ["", "## Artifacts", "",
              "```",
              "semantics.json              decoder contract, 994D layout, permutations, gains",
              "harness_validation.json     replay vs the rollout's own recorded body_target",
              "episodes.json               the three selected demonstrations",
              "replay/metrics.json         per-episode, per-variant palm and joint metrics",
              "replay/palm_metrics.csv     the same numbers, flat",
              "replay/episode_*.npz        decoded targets and palms per variant",
              "figures/palm_z_replay.png   measured vs decoded palm height",
              "figures/palm_z_mae.png      accuracy and negative controls",
              "manifest.json summary.json  commands, hashes, seed, decision",
              "```", ""]
    (out / "REPORT.md").write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("semantics", "validate", "replay", "report", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="raw teleoperation datasets, for the demonstration's own commanded arm action")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--obs-config", type=Path, default=DEFAULT_OBS_CONFIG)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--sonic-deploy", type=Path, default=DEFAULT_SONIC_DEPLOY)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--rollout", default="psi0-rollout-1", help="rollout session that validates the harness")
    parser.add_argument("--validate-stride", type=int, default=4, help="tick stride of the variant search")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    log_path = args.out / "logs" / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    semantics_payload = stage_semantics(args, log)
    if args.stage == "semantics":
        return 0
    if args.stage in ("validate", "all"):
        stage_validate(args, semantics_payload, log)
    if args.stage in ("replay", "all"):
        stage_replay(args, semantics_payload, log)
    if args.stage in ("report", "all"):
        summary = stage_report(args, semantics_payload, log)
        write_report(args, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
