#!/usr/bin/env python3
"""Psi0-SONIC bridge and its minimal web UI, as one process.

This is the session service: FastAPI control API + UI + policy-camera preview.
It owns the bridge's own ZMQ endpoints (state SUB, camera REQ, action PUB, Isaac
reset REQ) *and* the policy server process it serves: ``serve_psi0_sonic`` is
spawned as a child so an operator checkpoint switch (UI) is a controlled
restart -- session Stop, child terminate, new child for the selected run dir,
``/info`` identity verification, then ready.  Isaac and the SONIC Y controller
stay externally managed; dev.sh starts them next to this process.

Before serving, the launcher fails fast unless everything it depends on is
already correct:

* ``--checkpoint-dir`` is a real Psi0 run directory: ``run_config.json``,
  ``argv.txt`` and ``checkpoints/ckpt_<step>`` all exist;
* the run's action width is 78 or 80 and matches what the server reports, it
  declares exactly one image key and one resize equal to its ``center_crop``,
  and its ``num_past_frames`` (when the run config states it) is 0, i.e. the
  single frame this bridge sends;
* ``--checkpoint-step`` is an integer (``latest`` is rejected) and matches the
  step the policy server reports, and ``/info.run_dir`` must resolve to exactly
  this run directory (a stale server for another run fails the preflight);
* ``tcp://*:5556`` is owned by this service from startup until shutdown, so no
  other publisher can slip in between the check and the first action.

The UI offers the allowlisted checkpoints the launcher could build from real
artifacts: the fine-tuned run given on the command line, its training-start
(base) artifact materialized from the run's own ``run_config.json`` lineage into
a sibling ``base/<warm-start>`` run dir, and -- when
``--psi-dream-checkpoint-dir`` is given -- the released multi-task checkpoint,
served exactly like a fine-tune run.  Each entry's own ``run_config.json``
declares the camera key and transform its server must report, so the served
``/info`` is held to the selected run rather than to one global pin.  The API
takes an option id -- never a path.

    PYTHONPATH=src python3 scripts/psi0-isaac-eval.py \
        --checkpoint-dir /outputs/psi0-unitree-dex3-sonic-v1/finetune/<run> \
        --checkpoint-step 40000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from humanoid_lab.psi0_bridge import base_artifact  # noqa: E402
from humanoid_lab.psi0_bridge.base_artifact import BaseArtifactError  # noqa: E402
from humanoid_lab.psi0_bridge.checkpoints import (  # noqa: E402
    CheckpointController,
    CheckpointEntry,
)
from humanoid_lab.psi0_bridge.contracts import (  # noqa: E402
    ACTION_DIMS,
    MAX_TRANSFORM_SIDE,
    MIN_TRANSFORM_SIDE,
    ContractError,
)
from humanoid_lab.psi0_bridge.policy_server import PolicyServerError, PolicyServerProcess, probe_info  # noqa: E402
from humanoid_lab.psi0_bridge.prompt import CANONICAL_PROMPT  # noqa: E402
from humanoid_lab.psi0_bridge.policy_clock import PolicyClock  # noqa: E402
from humanoid_lab.psi0_bridge.psi0_client import Psi0ClientError, fetch_info  # noqa: E402
from humanoid_lab.psi0_bridge.session import Session, SessionConfig, SessionError  # noqa: E402

DEFAULT_PORT = 8015
DEFAULT_POLICY_PORT = 8014
DEFAULT_INFO_TIMEOUT_S = 900.0  # the 3B checkpoint takes minutes to load
INFO_POLL_S = 5.0
#: The released multi-task ψ-Dream checkpoint ships a single ckpt_40000.
DEFAULT_DREAM_STEP = 40000

FINETUNED_ID = "fine-tuned"
BASE_ID = "base"
DREAM_ID = "dream"


class LaunchError(RuntimeError):
    """The run cannot start; the message is meant for the operator."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Psi0-SONIC bridge + web UI")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Psi0 run directory inside this container "
                             "(run_config.json, argv.txt, checkpoints/ckpt_<step>)")
    parser.add_argument("--checkpoint-step", type=int, required=True,
                        help="integer checkpoint step to serve (for example 40000)")
    parser.add_argument("--base-run-dir", type=Path, default=None,
                        help="pre-materialized base (training-start) run dir; when omitted it is "
                             "derived from --checkpoint-dir's run_config.json and materialized "
                             "next to the fine-tune runs")
    parser.add_argument("--base-ckpt-step", type=int, default=base_artifact.BASE_STEP,
                        help="checkpoint directory name for the base entry (default 0: the "
                             "weights before the first fine-tune step)")
    parser.add_argument("--psi-dream-checkpoint-dir", type=Path, default=None,
                        help="released multi-task Psi0 SONIC run directory (for example the "
                             "downloaded psi0/sonic-checkpoints/multi-task.psi-dream.<stamp>); "
                             "shown as an extra ψ option when given")
    parser.add_argument("--psi-dream-checkpoint-step", type=int, default=DEFAULT_DREAM_STEP,
                        help=f"checkpoint step inside that run (default {DEFAULT_DREAM_STEP})")
    parser.add_argument("--groot-checkpoint-dir", type=Path, default=None,
                        help="GR00T checkpoint shown as the third model option")
    parser.add_argument("--policy-log", type=Path, default=None,
                        help="file the owned serve_psi0_sonic child logs to (default: inherit)")
    parser.add_argument("--policy-clock", choices=("wall", "simulation"), default="simulation",
                        help="policy action time basis for both models; simulation (default) reads Isaac's replay clock; wall reproduces legacy runs")
    parser.add_argument("--policy-clock-file", type=Path, default=None,
                        help="Isaac replay-clock path in simulation mode (default /outputs/psi0-isaac-eval-policy-clock.json); direct launches must arrange for Isaac to write this path")
    parser.add_argument("--policy-clock-timeout-s", type=float, default=5.0,
                        help="wall watchdog for an unavailable or stalled simulation clock")
    parser.add_argument("--groot-capture-dir", type=Path, default=None,
                        help="opt-in lossless GR00T request/response capture directory")
    parser.add_argument("--groot-capture-max-requests", type=int, default=32)
    parser.add_argument(
        "--groot-left-hand-contract",
        choices=("compatibility", "model-independent", "model-coupled"),
        default="model-independent",
        help="model-independent (default) matches training observations/actions; compatibility preserves legacy upstream behavior; model modes map live thumb/middle/index "
             "to trained thumb/index/middle observations and map actions back",
    )
    parser.add_argument("--psi0-neck-policy", choices=("error", "discard"), default="error",
                        help="error (default) rejects nonzero 80D neck padding; discard explicitly "
                             "drops unsupported neck channels and records their values in telemetry "
                             "(only use when the checkpoint's neck labels are masked padding)")
    parser.add_argument("--psi0-rtc-off", action="store_true",
                        help="opt in to unguided independent psi0 chunks; guided test-time RTC "
                             "remains the compatibility default")
    parser.add_argument("--psi0-action-exec-horizon", type=int, default=30,
                        help="rows executed from each independent chunk in --psi0-rtc-off mode; "
                             "also forwarded as the server action execution contract")
    parser.add_argument("--telemetry-dir", type=Path, default=None,
                        help="record this session's observations, target actions, applied `pose` "
                             "messages and head-camera frames as JSONL under this directory; "
                             "omitted means no recording")
    parser.add_argument("--telemetry-camera-every", type=int, default=5,
                        help="keep one head-camera JPEG per N PSI observations (default 5)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--psi0-url", default="ws://localhost:8014/ws",
                        help="Psi0 policy server WebSocket endpoint; /info is derived from it")
    parser.add_argument("--state-endpoint", default="tcp://localhost:5557",
                        help="SONIC g1_debug SUB endpoint")
    parser.add_argument("--state-topic", default="g1_debug")
    parser.add_argument("--camera-endpoint", default="tcp://localhost:5558",
                        help="camera get_frame REQ endpoint")
    parser.add_argument("--action-endpoint", default="tcp://*:5556",
                        help="Protocol v4 pose PUB bind endpoint")
    parser.add_argument("--isaac-control-endpoint", default="tcp://localhost:5559",
                        help="Isaac reset REP endpoint (owned by the Isaac service)")
    parser.add_argument("--reset-timeout-ms", type=int, default=2000)
    parser.add_argument("--warmstart-tokens", type=Path, default=None,
                        help="opt-in: a prepared demonstration token stream (JSON) published to "
                             "SONIC's action port over the settle tail of every Reset, so the "
                             "GR00T selection starts from a demonstration-supported upper limb "
                             "with the deployment's own history intact; off by default")
    parser.add_argument("--warmstart-delay-s", type=float, default=0.0,
                        help="seconds after a Reset before the warm-start stream starts "
                             "(wall seconds by default, simulation seconds with --warmstart-sim-clock)")
    parser.add_argument("--warmstart-sim-clock", type=Path, default=None,
                        help="opt-in: pace warm-start frames from Isaac's --replay-clock-output file")
    parser.add_argument("--warmstart-clock-timeout-s", type=float, default=5.0,
                        help="fail the warm start if its simulation clock is unavailable or stale")
    parser.add_argument("--initial-pose-handshake", action="store_true",
                        help="opt-in: on every Reset, publish the VLA client's own initial-pose "
                             "command (its LATENT_INITIAL_MOTION_TOKEN, packed with its own "
                             "protocol v4 packer) to SONIC's action port from the start of the "
                             "settle until Start, so the GR00T selection always starts from the "
                             "intended pre-policy command instead of the previous rollout's last "
                             "token; off by default and mutually exclusive with --warmstart-tokens")
    parser.add_argument("--command-ttl-s", type=float, default=0.3,
                        help="wait after Stop before Reset so the SONIC command TTL (0.25s) expires")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--recv-timeout", type=float, default=60.0,
                        help="seconds to wait for one action (the first absorbs model warmup)")
    parser.add_argument("--info-timeout", type=float, default=DEFAULT_INFO_TIMEOUT_S,
                        help="seconds to wait for the policy server's /info while it loads")
    parser.add_argument("--webrtc-host", default=None)
    parser.add_argument("--webrtc-port", default=None)
    parser.add_argument("--webrtc-client", default=None,
                        help="path to the Isaac Sim WebRTC native client (default: $ISAAC_WEBRTC_CLIENT)")
    parser.add_argument("--webrtc-url", default="",
                        help="optional browser URL for the WebRTC session")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the checkpoint, the action port and /info, then exit")
    args = parser.parse_args(argv)
    if args.policy_clock == "simulation" and args.policy_clock_file is None:
        # Direct launcher use (without dev.sh) still shares one deterministic
        # file name with the Isaac process the operator must start separately.
        args.policy_clock_file = Path("/outputs/psi0-isaac-eval-policy-clock.json")
    if args.policy_clock == "wall" and args.policy_clock_file is not None:
        parser.error("--policy-clock-file requires --policy-clock simulation")
    if args.policy_clock_timeout_s <= 0:
        parser.error("--policy-clock-timeout-s must be positive")
    if args.groot_capture_max_requests <= 0:
        parser.error("--groot-capture-max-requests must be positive")
    if args.psi0_neck_policy == "discard" and args.telemetry_dir is None:
        parser.error("--psi0-neck-policy discard requires --telemetry-dir to audit dropped values")
    if args.policy_clock_file is not None:
        args.policy_clock_file = args.policy_clock_file.resolve()
    return args


def validate_run_dir(run_dir: Path, step: int, *, minimum_step: int = 1) -> dict:
    """Fail fast unless ``run_dir`` is a servable Psi0 run at ``step``.

    ``minimum_step`` stays 1 for the CLI's fine-tuned ``--checkpoint-step`` (the
    server has no "latest" here); the base entry is the pre-fine-tune weights and
    is validated with 0.
    """
    if step < minimum_step:
        raise LaunchError(
            f"checkpoint step must be an integer >= {minimum_step}, got {step}"
        )
    if not run_dir.is_dir():
        raise LaunchError(f"checkpoint directory does not exist: {run_dir}")

    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise LaunchError(f"not a Psi0 run directory (no run_config.json): {run_dir}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise LaunchError(f"{config_path} is not a JSON object")

    argv_path = run_dir / "argv.txt"
    if not argv_path.is_file():
        raise LaunchError(f"run directory is missing argv.txt, which the server parses: {run_dir}")
    ckpt_path = run_dir / "checkpoints" / f"ckpt_{step}"
    if not ckpt_path.is_dir():
        raise LaunchError(f"checkpoint ckpt_{step} does not exist under {run_dir / 'checkpoints'}")

    model = config.get("model")
    action_dim = model.get("action_dim") if isinstance(model, dict) else None
    if action_dim not in ACTION_DIMS:
        raise LaunchError(
            f"run_config action_dim is {action_dim!r}; the bridge serves only {ACTION_DIMS}"
        )

    transform = (
        config.get("data", {}).get("transform", {}).get("model", {})
        if isinstance(config.get("data"), dict) else {}
    )
    resize = transform.get("resize", {}).get("size") if isinstance(transform, dict) else None
    if not isinstance(resize, (list, tuple)) or len(resize) != 2:
        raise LaunchError(f"run_config resize is {resize!r}; expected a 2-element [height, width]")
    resize_size = [int(resize[0]), int(resize[1])]
    for side in resize_size:
        if not MIN_TRANSFORM_SIDE <= side <= MAX_TRANSFORM_SIDE:
            raise LaunchError(
                f"run_config resize {resize_size} is outside "
                f"[{MIN_TRANSFORM_SIDE}, {MAX_TRANSFORM_SIDE}]"
            )
    if "center_crop" not in transform:
        raise LaunchError("run_config has no center_crop transform; refusing to guess the view")
    crop = transform.get("center_crop")
    crop_size = crop.get("size") if isinstance(crop, dict) else None
    if not isinstance(crop_size, (list, tuple)) or len(crop_size) != 2:
        raise LaunchError(
            f"run_config center_crop is {crop_size!r}; expected a 2-element [height, width]"
        )
    crop_size = [int(crop_size[0]), int(crop_size[1])]
    if crop_size != resize_size:
        raise LaunchError(
            f"run_config center_crop {crop_size} differs from its resize {resize_size}; the "
            "bridge serves the run's own resize and cannot reproduce a narrower view"
        )

    state_dim = (
        config.get("data", {}).get("transform", {}).get("field", {}).get("pad_state_dim")
        if isinstance(config.get("data"), dict) else None
    )
    if state_dim not in (43, 45):
        raise LaunchError(f"run_config pad_state_dim is {state_dim!r}; expected 43 or 45")

    # The bridge sends one state frame (the request's `history` is empty), so the
    # run must have been packed with num_past_frames=0.  run_config.json is the
    # materialized config the server validates against (argv.txt only supplies
    # structure), so its value is the evidence; a missing key is reported as
    # unknown instead of being guessed.
    repack = None
    if isinstance(config.get("data"), dict):
        repack = config["data"].get("transform", {}).get("repack")
    num_past_frames = repack.get("num_past_frames") if isinstance(repack, dict) else None
    if num_past_frames is not None and num_past_frames != 0:
        raise LaunchError(
            f"run_config repack.num_past_frames is {num_past_frames!r}; the bridge sends a "
            "single frame (history 1), so this run's state history would not match"
        )

    # The served run names its one camera.  The bridge sends the frame under that
    # key, so the name may differ between released checkpoints; the count may not,
    # because the bridge has exactly one camera to offer.
    image_keys = repack.get("image_keys") if isinstance(repack, dict) else None
    if (not isinstance(image_keys, (list, tuple)) or len(image_keys) != 1
            or not isinstance(image_keys[0], str) or not image_keys[0].strip()):
        raise LaunchError(
            f"run_config repack.image_keys is {image_keys!r}; the bridge sends exactly one "
            "camera frame, so this run must declare exactly one image key"
        )
    image_key = image_keys[0].strip()

    return {
        "run_dir": str(_canonical(run_dir)),
        "ckpt_step": step,
        "action_dim": action_dim,
        "chunk_size": model.get("action_chunk_size"),
        "resize": resize_size,
        "center_crop": crop_size,
        "state_dim": state_dim,
        "num_past_frames": num_past_frames,
        "image_key": image_key,
        "dataset_name": (
            config.get("data", {}).get("transform", {}).get("repack", {}).get("dataset_name")
        ),
    }


def _canonical(path: str | Path) -> Path:
    """Resolve to one canonical spelling (symlink-free) for identity comparisons."""
    try:
        return Path(path).resolve()
    except OSError:  # pragma: no cover - only on exotic filesystems
        return Path(path)


def wait_for_info(
    ws_url: str,
    *,
    timeout_s: float,
    expected_step: int,
    expected_run_dir: Path,
    expected_action_dim: int,
    expected_state_dim: int | None = None,
    expected_resize: list[int] | None = None,
    expected_center_crop: list[int] | None = None,
    expected_image_key: str | None = None,
    expected_dataset_name: str | None = None,
):
    """Poll ``/info`` until the policy server has loaded, then validate its identity.

    An unreachable server is retried (the checkpoint takes minutes to load); a
    contract mismatch fails immediately.  The served ``run_dir`` must be the
    canonical path of the directory the launcher selected — matching only the
    step or the action width would let a stale server for another run pass.  The
    request dimensions and transforms must agree with the run's config too, so a
    server that loaded a different artifact cannot be adopted quietly.
    """
    deadline = time.monotonic() + max(timeout_s, 0.0)
    last_error: Exception | None = None
    while True:
        try:
            info = fetch_info(ws_url)
        except ContractError:
            raise
        except Psi0ClientError as exc:
            last_error = exc
        else:
            if info.ckpt_step is not None and info.ckpt_step != expected_step:
                raise LaunchError(
                    f"the policy server reports ckpt_step {info.ckpt_step}, but this run was "
                    f"asked for step {expected_step}"
                )
            served = _canonical(info.run_dir)
            if served != expected_run_dir:
                raise LaunchError(
                    f"the policy server serves {served}, not the selected run "
                    f"{expected_run_dir}; is a stale server from another run on :8014?"
                )
            if info.action_dim != expected_action_dim:
                raise LaunchError(
                    f"run_config action_dim {expected_action_dim} does not match the served "
                    f"{info.action_dim}; is the server serving the same run?"
                )
            if expected_state_dim is not None and info.state_dim != expected_state_dim:
                raise LaunchError(
                    f"run_config pad_state_dim {expected_state_dim} does not match the served "
                    f"{info.state_dim}; the request layout would change"
                )
            if expected_resize is not None and list(info.resize_size) != list(expected_resize):
                raise LaunchError(
                    f"run_config resize {expected_resize} does not match the served "
                    f"{list(info.resize_size)}"
                )
            if expected_center_crop is not None and list(info.center_crop_size) != list(expected_center_crop):
                raise LaunchError(
                    f"run_config center_crop {expected_center_crop} does not match the served "
                    f"{list(info.center_crop_size)}"
                )
            if expected_image_key is not None and info.image_key != expected_image_key:
                raise LaunchError(
                    f"run_config image key {expected_image_key!r} does not match the served "
                    f"{info.image_key!r}; the request would carry the frame under a key the "
                    "served model does not read"
                )
            if expected_dataset_name is not None and info.dataset_name != expected_dataset_name:
                raise LaunchError(
                    f"run_config dataset_name {expected_dataset_name!r} does not match the served "
                    f"{info.dataset_name!r}; the request layout would change"
                )
            return info
        if time.monotonic() >= deadline:
            raise LaunchError(
                f"policy server /info is still unreachable after {timeout_s:.0f}s: {last_error}"
            )
        time.sleep(INFO_POLL_S)


def _identity_kwargs(summary: dict) -> dict:
    """The expected /info identity of a validated run directory."""
    return {
        "expected_step": summary["ckpt_step"],
        "expected_run_dir": Path(summary["run_dir"]),
        "expected_action_dim": summary["action_dim"],
        "expected_state_dim": summary["state_dim"],
        "expected_resize": summary["resize"],
        "expected_center_crop": summary["center_crop"],
        "expected_image_key": summary.get("image_key"),
        "expected_dataset_name": summary["dataset_name"],
    }


def preflight(args: argparse.Namespace) -> dict:
    """Validate the run directory and the served /info identity (no sockets taken)."""
    summary = validate_run_dir(args.checkpoint_dir, args.checkpoint_step)
    print(f"[psi0-isaac-eval] run dir ok: {summary}")
    print(f"[psi0-isaac-eval] canonical run dir: {summary['run_dir']}")

    info = wait_for_info(
        args.psi0_url,
        timeout_s=args.info_timeout,
        **_identity_kwargs(summary),
    )
    print(f"[psi0-isaac-eval] /info ok: run_dir {info.run_dir} (canonical match), action "
          f"{info.action_dim}D, chunk {info.action_chunk_size}, exec {info.action_exec_horizon}, "
          f"state {info.state_dim}D, history {info.history_length}, resize {info.resize_size}, "
          f"ckpt_step {info.ckpt_step}, dataset {info.dataset_name}")
    return summary


def finetuned_label(step: int) -> str:
    if step % 1000 == 0:
        return f"Fine-tuned ({step // 1000}k)"
    return f"Fine-tuned (step {step})"


def finetuned_entry(args: argparse.Namespace) -> CheckpointEntry:
    summary = validate_run_dir(args.checkpoint_dir, args.checkpoint_step)
    return CheckpointEntry(
        id=FINETUNED_ID,
        label=finetuned_label(args.checkpoint_step),
        run_dir=Path(summary["run_dir"]),
        step=args.checkpoint_step,
        detail=summary,
    )


def dream_entry(args: argparse.Namespace) -> Optional[CheckpointEntry]:
    """The released multi-task SONIC checkpoint, when the launcher was given one.

    It is a complete Psi0 run directory (``run_config.json``, ``argv.txt``,
    ``checkpoints/ckpt_<step>/model.safetensors`` in the deploy loader's key
    layout), so it is served exactly like a fine-tuned run: the only difference
    is that its own ``run_config`` names a different camera key and transform,
    which the identity check below holds it to.
    """
    if args.psi_dream_checkpoint_dir is None:
        return None
    step = args.psi_dream_checkpoint_step
    try:
        summary = validate_run_dir(args.psi_dream_checkpoint_dir, step)
    except Exception as exc:  # noqa: BLE001 - the option degrades, never the session
        print(f"[psi0-isaac-eval] psi-dream checkpoint unavailable: {exc}", file=sys.stderr)
        return CheckpointEntry(
            id=DREAM_ID, label=dream_label(step),
            run_dir=None, step=step,
            available=False, reason=f"{type(exc).__name__}: {exc}",
        )
    print(f"[psi0-isaac-eval] psi-dream checkpoint: {summary['run_dir']} "
          f"(step {step}, {summary['action_dim']}D, camera {summary['image_key']}, "
          f"resize {summary['resize']})", flush=True)
    return CheckpointEntry(
        id=DREAM_ID,
        label=dream_label(step),
        run_dir=Path(summary["run_dir"]),
        step=step,
        detail=summary,
    )


def dream_label(step: int) -> str:
    if step % 1000 == 0:
        return f"ψ-Dream ({step // 1000}k)"
    return f"ψ-Dream (step {step})"


def base_entry(args: argparse.Namespace) -> CheckpointEntry:
    """The training-start entry, materialized from the fine-tune run's own lineage.

    Any failure leaves the base entry explicitly unavailable -- the selector
    keeps two options and no other checkpoint is substituted.
    """
    step = args.base_ckpt_step
    base_dir = args.base_run_dir
    try:
        if base_dir is None:
            fine_dir = Path(_canonical(args.checkpoint_dir))
            source = base_artifact.read_base_source(fine_dir)
            base_dir = base_artifact.default_base_run_dir(fine_dir, source)
            base_artifact.materialize_base_run_dir(
                fine_dir, base_dir, source=source, step=step,
                log=lambda message: print(f"[psi0-isaac-eval] {message}", flush=True),
            )
        summary = validate_run_dir(base_dir, step, minimum_step=base_artifact.BASE_STEP)
    except Exception as exc:  # noqa: BLE001 - the base option degrades, never the session
        print(f"[psi0-isaac-eval] base checkpoint unavailable: {exc}", file=sys.stderr)
        return CheckpointEntry(
            id=BASE_ID, label="Base",
            run_dir=None, step=step,
            available=False, reason=f"{type(exc).__name__}: {exc}",
        )
    print(f"[psi0-isaac-eval] base checkpoint: {summary['run_dir']} "
          f"(step {step}, {summary['action_dim']}D)", flush=True)
    return CheckpointEntry(
        id=BASE_ID, label="Base",
        run_dir=Path(summary["run_dir"]), step=step, detail=summary,
    )


def _webrtc_info(args: argparse.Namespace) -> dict[str, str]:
    import os

    host = args.webrtc_host or os.environ.get("ISAAC_LIVESTREAM_ENDPOINT") or "localhost"
    port = args.webrtc_port or os.environ.get("ISAAC_LIVESTREAM_PORT") or "49100"
    client = args.webrtc_client or os.environ.get("ISAAC_WEBRTC_CLIENT") or ""
    return {"endpoint": f"{host}:{port}", "client": client, "url": args.webrtc_url}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --check-only is a diagnostic: it validates the run directory and the served
    # /info identity without taking the action socket.
    if args.check_only:
        try:
            preflight(args)
        except (LaunchError, ContractError) as exc:
            print(f"[psi0-isaac-eval] preflight failed: {exc}", file=sys.stderr)
            return 2
        print("[psi0-isaac-eval] check-only: OK")
        return 0

    if args.groot_checkpoint_dir is None:
        print("[psi0-isaac-eval] --groot-checkpoint-dir is required when serving the unified UI", file=sys.stderr)
        return 2

    # Serving: take the action socket first.  The service owns :5556 for its
    # whole lifetime, so a busy port fails startup here (not later at Start) and
    # no other publisher can slip in between a probe and the first action.
    from humanoid_lab.psi0_bridge.action_router import ActionRouter, RouterError
    from humanoid_lab.psi0_bridge.telemetry import SessionTelemetry, TelemetryError

    telemetry = None
    if args.telemetry_dir is not None:
        try:
            telemetry = SessionTelemetry(
                args.telemetry_dir,
                camera_every=args.telemetry_camera_every,
                session_tag=args.telemetry_dir.name,
            )
        except TelemetryError as exc:
            print(f"[psi0-isaac-eval] cannot record telemetry: {exc}", file=sys.stderr)
            return 2
        print(f"[psi0-isaac-eval] telemetry: {telemetry.events_path}")

    try:
        router = ActionRouter(public_endpoint=args.action_endpoint, telemetry=telemetry)
        session = Session(SessionConfig(
            ws_url=args.psi0_url,
            state_endpoint=args.state_endpoint,
            state_topic=args.state_topic,
            camera_endpoint=args.camera_endpoint,
            action_endpoint=args.action_endpoint,
            reset_endpoint=args.isaac_control_endpoint,
            instruction=CANONICAL_PROMPT,
            control_hz=args.control_hz,
            recv_timeout_s=args.recv_timeout,
            reset_timeout_ms=args.reset_timeout_ms,
            command_ttl_s=args.command_ttl_s,
            neck_policy=args.psi0_neck_policy,
        ), publisher=router.psi_sink, telemetry=telemetry, policy_clock=PolicyClock(
            args.policy_clock,
            args.policy_clock_file,
            stale_timeout_s=args.policy_clock_timeout_s,
        ))
    except (SessionError, RouterError) as exc:
        print(f"[psi0-isaac-eval] cannot own the action socket {args.action_endpoint}: {exc}",
              file=sys.stderr)
        return 2
    if telemetry is not None:
        # The GR00T backend is driven by NVIDIA's VLA client, so the only camera
        # reading the bridge can attach to its actions is its own monitor's.
        telemetry.set_frame_provider(session.monitor.frame)
    print(f"[psi0-isaac-eval] action socket owned: {args.action_endpoint}")

    # The fine-tuned entry is validated before anything is spawned; a bad run
    # directory must not leave a policy server behind.
    try:
        fine_entry = finetuned_entry(args)
    except (LaunchError, ContractError) as exc:
        print(f"[psi0-isaac-eval] preflight failed: {exc}", file=sys.stderr)
        session.close()
        router.close()
        return 2

    # This process owns the policy server: a UI checkpoint switch restarts this
    # child.  A foreign server on the same port is refused (the identity check
    # would fail anyway, but not before Isaac and the model spent minutes).
    policy_port = urlsplit(args.psi0_url).port or DEFAULT_POLICY_PORT
    server = PolicyServerProcess(
        port=policy_port,
        log_path=args.policy_log,
        rtc=not args.psi0_rtc_off,
        action_exec_horizon=args.psi0_action_exec_horizon,
        policy_clock=args.policy_clock,
        policy_clock_file=args.policy_clock_file,
        policy_clock_timeout_s=args.policy_clock_timeout_s,
    )
    foreign = probe_info("127.0.0.1", policy_port)
    if foreign is not None:
        print(f"[psi0-isaac-eval] :{policy_port} already serves a policy server "
              f"({foreign.get('run_dir')!r}); refusing to start next to it", file=sys.stderr)
        session.close()
        router.close()
        return 2
    try:
        server.start(fine_entry.run_dir, fine_entry.step)
    except PolicyServerError as exc:
        print(f"[psi0-isaac-eval] cannot start the policy server: {exc}", file=sys.stderr)
        session.close()
        router.close()
        return 2
    print(f"[psi0-isaac-eval] policy server pid {server.pid} for {fine_entry.run_dir} "
          f"(step {fine_entry.step})", flush=True)

    try:
        # Materialize/validate the base entry while the 3B policy server loads.
        entries = [fine_entry, base_entry(args)]
        dream = dream_entry(args)
        if dream is not None:
            entries.append(dream)
        try:
            info = wait_for_info(
                args.psi0_url,
                timeout_s=args.info_timeout,
                **_identity_kwargs(fine_entry.detail),
            )
        except (LaunchError, ContractError) as exc:
            print(f"[psi0-isaac-eval] preflight failed: {exc}", file=sys.stderr)
            return 2
        print(f"[psi0-isaac-eval] /info ok: run_dir {info.run_dir} (canonical match), action "
              f"{info.action_dim}D, chunk {info.action_chunk_size}, exec {info.action_exec_horizon}, "
              f"state {info.state_dim}D, history {info.history_length}, resize {info.resize_size}, "
              f"ckpt_step {info.ckpt_step}, dataset {info.dataset_name}")
        session.adopt_policy(info)

        def verify(entry: CheckpointEntry):
            return wait_for_info(
                args.psi0_url,
                timeout_s=args.info_timeout,
                **_identity_kwargs(entry.detail),
            )

        from humanoid_lab.psi0_bridge.groot_backend import GrootProcessGroup, validate_groot_checkpoint
        from humanoid_lab.psi0_bridge.model_controller import ModelController, ModelEntry
        from humanoid_lab.psi0_bridge.warmstart import (
            FileSimulationClock,
            WarmStartError,
            WarmStartStream,
            load_token_stream,
        )

        if args.initial_pose_handshake and args.warmstart_tokens is not None:
            print("[psi0-isaac-eval] --initial-pose-handshake and --warmstart-tokens both own the "
                  "settle; configure one of them", file=sys.stderr)
            return 2
        if args.warmstart_sim_clock is not None and args.warmstart_tokens is None:
            print("[psi0-isaac-eval] --warmstart-sim-clock requires --warmstart-tokens", file=sys.stderr)
            return 2

        initial_pose = None
        if args.initial_pose_handshake:
            from humanoid_lab.psi0_bridge.initial_pose import (
                InitialPoseError,
                initial_pose_handshake,
                load_initial_pose_command,
            )

            try:
                command = load_initial_pose_command()
            except InitialPoseError as exc:
                print(f"[psi0-isaac-eval] initial-pose command unusable: {exc}", file=sys.stderr)
                return 2
            initial_pose = initial_pose_handshake(
                router, command=command, telemetry=telemetry,
                log=lambda message: print(f"[psi0-isaac-eval] {message}", flush=True),
            )
            detail = command.summary()
            print(f"[psi0-isaac-eval] initial-pose handshake: {detail['module']}."
                  f"{detail['constant']} ({detail['token_dim']}D, tokens[0]="
                  f"{detail['token_head'][0]:+.4f}) held at {detail['control_hz']:g} Hz over every "
                  f"settle, token sha256 {detail['token_sha256_f32'][:12]}")
            if telemetry is not None:
                telemetry.event("initial_pose", state="configured", **detail)

        warmstart = None
        if args.warmstart_tokens is not None:
            try:
                stream = load_token_stream(args.warmstart_tokens)
            except WarmStartError as exc:
                print(f"[psi0-isaac-eval] warm-start tokens unusable: {exc}", file=sys.stderr)
                return 2
            warmstart = WarmStartStream(
                stream, router, telemetry=telemetry,
                simulation_clock=(
                    None if args.warmstart_sim_clock is None
                    else FileSimulationClock(args.warmstart_sim_clock)
                ),
                clock_timeout_s=args.warmstart_clock_timeout_s,
                log=lambda message: print(f"[psi0-isaac-eval] {message}", flush=True),
            )
            print(f"[psi0-isaac-eval] warm-start: {stream.ticks} demo ticks at "
                  f"{stream.control_hz:g} Hz ({stream.duration_s:.2f}s), episode "
                  f"{stream.source.get('episode_index')}, delay {args.warmstart_delay_s:g}s, "
                  f"sha256 {stream.sha256[:12]}")
            if telemetry is not None:
                telemetry.event("warmstart", state="configured", delay_s=args.warmstart_delay_s,
                                **stream.summary())

        try:
            validate_groot_checkpoint(args.groot_checkpoint_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[psi0-isaac-eval] GR00T checkpoint unavailable: {exc}", file=sys.stderr)
            return 2
        model_entries = [
            ModelEntry(
                id=entry.id, label=entry.label, kind="psi", run_dir=entry.run_dir,
                step=entry.step, available=entry.available, reason=entry.reason,
                detail=dict(entry.detail),
            )
            for entry in entries
        ]
        finetuned_entry_detail = dict(fine_entry.detail)
        groot = GrootProcessGroup(
            args.groot_checkpoint_dir,
            prompt=CANONICAL_PROMPT,
            log_dir=(args.policy_log.parent if args.policy_log else Path("/tmp/groot-eval")),
            policy_clock=args.policy_clock,
            policy_clock_file=args.policy_clock_file,
            policy_clock_timeout_s=args.policy_clock_timeout_s,
            capture_dir=args.groot_capture_dir,
            capture_max_requests=args.groot_capture_max_requests,
            left_hand_contract=args.groot_left_hand_contract,
        )
        controller = ModelController(
            psi_session=session,
            psi_server=server,
            psi_entries=model_entries,
            verify_psi=verify,
            groot=groot,
            router=router,
            initial_id=FINETUNED_ID,
            reset_endpoint=args.isaac_control_endpoint,
            warmstart=warmstart,
            warmstart_delay_s=args.warmstart_delay_s,
            initial_pose=initial_pose,
            log=lambda message: print(f"[psi0-isaac-eval] {message}", flush=True),
        )

        from humanoid_lab.psi0_bridge.app import create_app  # imported only when serving

        app = create_app(
            controller, webrtc=_webrtc_info(args), checkpoints=controller,
            subtitle="PSI / GR00T → NVIDIA SONIC → Isaac G1 + Dex3",
            policy_label="Policy backend",
        )
        labels = [entry.label for entry in entries] + ["GR00T"]
        print(f"[psi0-isaac-eval] UI on http://{args.host}:{args.port}/  "
              f"(prompt: {CANONICAL_PROMPT!r}); models: {', '.join(labels)}", flush=True)
        if telemetry is not None:
            telemetry.event(
                "policy",
                prompt=CANONICAL_PROMPT,
                fine_tuned=finetuned_entry_detail,
                groot_checkpoint=str(args.groot_checkpoint_dir),
                control_hz=args.control_hz,
                host=args.host,
                port=args.port,
            )
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if telemetry is not None:
            telemetry.event("shutdown")
            telemetry.close()
        server.stop()
        session.close()
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
