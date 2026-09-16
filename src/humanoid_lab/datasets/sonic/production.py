"""Single production path from a canonical whole-body episode to SONIC artifacts.

Dataset adapters stop at :class:`CanonicalEpisodeBuild`.  Everything after that
point is source-agnostic and lives here: the fixed offline orientation policy,
the 1751D observation, the pinned ONNX encoder, and the optional 78D action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .encoder_observation import (
    ENCODER_INPUT_DIM,
    OrientationPolicy,
    build_g1_encoder_observation,
    load_encoder_layout,
    require_conversion_policy,
)
from .encoder_runner import DEFAULT_PROVIDERS, G1EncoderRunner, compose_pilot_action
from .provenance import sha256_file
from .schema import CanonicalEpisode

PRODUCTION_ORIENTATION_POLICY = OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME
PINNED_ENCODER_SHA256 = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"


@dataclass(frozen=True)
class EncoderInputArtifacts:
    """The only encoder input representation accepted by production tooling."""

    observation: np.ndarray
    future_clamp_fraction: np.ndarray
    timestamps: np.ndarray


def build_encoder_input(episode: CanonicalEpisode) -> EncoderInputArtifacts:
    """Convert one canonical 50 Hz whole-body episode into the pinned 1751D input."""
    episode.validate()
    if len(episode.timestamps) > 1:
        frame_dt = np.diff(episode.timestamps)
        if not np.allclose(frame_dt, 1.0 / 50.0, rtol=0.0, atol=2e-6):
            raise ValueError(
                "production encoder requires a uniform 50 Hz CanonicalEpisode; "
                "resampling belongs in the dataset adapter"
            )
    policy = require_conversion_policy(PRODUCTION_ORIENTATION_POLICY)
    observation, clamp = build_g1_encoder_observation(episode, policy=policy)
    return EncoderInputArtifacts(
        observation=np.ascontiguousarray(observation, dtype=np.float32),
        future_clamp_fraction=np.ascontiguousarray(clamp, dtype=np.float32),
        timestamps=np.ascontiguousarray(episode.timestamps, dtype=np.float64),
    )


def write_encoder_input(output_dir: Path, episode: CanonicalEpisode) -> EncoderInputArtifacts:
    """Build and persist the common encoder input beside a canonical reference."""
    artifacts = build_encoder_input(episode)
    np.savez_compressed(
        Path(output_dir) / "encoder_observation.npz",
        observation=artifacts.observation,
        future_clamp_fraction=artifacts.future_clamp_fraction,
        timestamps=artifacts.timestamps,
    )
    return artifacts


def encode_prepared_episode(
    pilot_dir: Path,
    *,
    model_dir: Path,
    expected_sha256: str | None = None,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Encode a prepared canonical episode and write its sole token/action artifacts."""
    pilot = Path(pilot_dir).resolve()
    model_dir = Path(model_dir)
    observation_path = pilot / "encoder_observation.npz"
    reference_path = pilot / "reference.npz"
    manifest_path = pilot / "run_manifest.json"
    for path in (observation_path, reference_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"prepared SONIC artifact missing: {path}")

    layout = load_encoder_layout(model_dir / "observation_config.yaml")
    if layout.total_dim != ENCODER_INPUT_DIM:
        raise ValueError("pinned observation config does not describe the 1751D G1 layout")

    with np.load(observation_path) as payload:
        observation = np.asarray(payload["observation"], dtype=np.float32)
        clamp = np.asarray(payload["future_clamp_fraction"], dtype=np.float32)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("encoder_orientation_policy") != PRODUCTION_ORIENTATION_POLICY.value:
        raise ValueError(
            "prepared episode uses a non-production orientation policy: "
            f"{manifest.get('encoder_orientation_policy')!r}"
        )

    runner = G1EncoderRunner(
        model_dir / "model_encoder.onnx",
        providers=providers,
        expected_sha256=expected_sha256,
    )
    tokens = runner.encode_frames(observation)
    probes = [0, len(observation) // 2, len(observation) - 1]
    repeatable = all(np.array_equal(runner.encode(observation[index]), tokens[index]) for index in probes)
    if not repeatable:
        raise RuntimeError("encoder is not repeatable on this machine; refusing to write tokens")

    with np.load(reference_path) as reference:
        hands_available = manifest["final_action_78d"]["status"] == "available"
        left = np.asarray(reference["left_hand_joints"], dtype=np.float32) if hands_available else None
        right = np.asarray(reference["right_hand_joints"], dtype=np.float32) if hands_available else None
    action = compose_pilot_action(tokens, left, right)
    arrays: dict[str, np.ndarray] = {
        "motion_token": tokens,
        "frame_index": np.arange(len(tokens), dtype=np.int64),
        "timestamp": timestamps,
        # A G1 token describes a reference window starting at this row.  The
        # upstream runtime clamps out-of-range future samples to the last
        # motion frame, but those tail tokens are poor VLA supervision: their
        # intent gradually changes into an artificial hold.  Keep them for
        # exact SONIC replay, and explicitly mask them out for training.
        "training_valid_mask": clamp == 0.0,
    }
    if action is not None:
        arrays.update({"left_hand_joints": left, "right_hand_joints": right, "action": action})
    np.savez_compressed(pilot / "action_tokens.npz", **arrays)

    pinned: dict[str, object] = {}
    lock = Path(lock_path) if lock_path is not None else None
    if lock is not None and lock.is_file():
        import yaml

        document = yaml.safe_load(lock.read_text(encoding="utf-8"))
        pinned = {
            "sonic_source_commit": document.get("repositories", {}).get("sonic", {}).get("commit"),
            "encoder_model_revision": document.get("models", {}).get("sonic", {}).get("revision")
            or document.get("models", {}).get("sonic_deploy", {}).get("revision"),
            "lock_file": str(lock),
        }

    norms = np.linalg.norm(tokens, axis=1)
    content: dict[str, Any] = {
        "pipeline": "canonical_50hz_whole_body_to_sonic_v1_1",
        "encoder": runner.info.to_dict(),
        "pinned_versions": pinned,
        "observation_config_sha256": sha256_file(model_dir / "observation_config.yaml"),
        "observation_file": str(observation_path),
        "frames": int(len(tokens)),
        "motion_token_dim": int(tokens.shape[1]),
        "action_dim": 0 if action is None else int(action.shape[1]),
        "final_action_78d": {
            "status": "available" if action is not None else "blocked",
            "reason": manifest["final_action_78d"]["reason"],
        },
        "finite": bool(np.isfinite(tokens).all() and (action is None or np.isfinite(action).all())),
        "repeatability": {"probes": probes, "bitwise_identical": bool(repeatable)},
        "future_clamp_fraction": {"mean": float(clamp.mean()), "max": float(clamp.max())},
        "training_labels": {
            "semantics": "observation_t -> encode(reference window starting at t)",
            "valid_rule": "future_clamp_fraction == 0; clamped tail is replay-only",
            "valid_frames": int(np.count_nonzero(clamp == 0.0)),
            "excluded_tail_frames": int(np.count_nonzero(clamp != 0.0)),
        },
        "token_norm": {
            "p50": float(np.percentile(norms, 50)),
            "p99": float(np.percentile(norms, 99)),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "encoder_orientation_policy": PRODUCTION_ORIENTATION_POLICY.value,
        "parity_note": (
            "offline tokens cannot be bit-compared with deployment tokens: the live encoder consumes the "
            "measured robot base quaternion, which these datasets do not record"
        ),
    }
    (pilot / "encoder_manifest.json").write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return content
