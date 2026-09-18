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
* the run's action width is 78 or 80 and matches what the server reports, its
  image transform is the pinned 240x320 resize, and its ``num_past_frames``
  (when the run config states it) is 0, i.e. the single frame this bridge sends;
* ``--checkpoint-step`` is an integer (``latest`` is rejected) and matches the
  step the policy server reports, and ``/info.run_dir`` must resolve to exactly
  this run directory (a stale server for another run fails the preflight);
* ``tcp://*:5556`` is owned by this service from startup until shutdown, so no
  other publisher can slip in between the check and the first action.

The UI offers exactly two allowlisted checkpoints: the fine-tuned run given on
the command line and its training-start (base) artifact, materialized from the
run's own ``run_config.json`` lineage into a sibling ``base/<warm-start>`` run
dir.  The API takes an option id -- never a path.

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
from humanoid_lab.psi0_bridge.contracts import ACTION_DIMS, RESIZE_SIZE, ContractError  # noqa: E402
from humanoid_lab.psi0_bridge.policy_server import PolicyServerError, PolicyServerProcess, probe_info  # noqa: E402
from humanoid_lab.psi0_bridge.prompt import CANONICAL_PROMPT  # noqa: E402
from humanoid_lab.psi0_bridge.psi0_client import Psi0ClientError, fetch_info  # noqa: E402
from humanoid_lab.psi0_bridge.session import Session, SessionConfig, SessionError  # noqa: E402

DEFAULT_PORT = 8015
DEFAULT_POLICY_PORT = 8014
DEFAULT_INFO_TIMEOUT_S = 900.0  # the 3B checkpoint takes minutes to load
INFO_POLL_S = 5.0

FINETUNED_ID = "fine-tuned"
BASE_ID = "base"


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
    parser.add_argument("--policy-log", type=Path, default=None,
                        help="file the owned serve_psi0_sonic child logs to (default: inherit)")
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
    return parser.parse_args(argv)


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
    if list(resize or []) != list(RESIZE_SIZE):
        raise LaunchError(
            f"run_config resize is {resize!r}; /info must report the pinned {RESIZE_SIZE} "
            "transform for the bridge contract to hold"
        )
    if "center_crop" not in transform:
        raise LaunchError("run_config has no center_crop transform; refusing to guess the view")
    crop = transform.get("center_crop")
    crop_size = crop.get("size") if isinstance(crop, dict) else None
    if list(crop_size or []) != list(RESIZE_SIZE):
        raise LaunchError(
            f"run_config center_crop is {crop_size!r}; the served transform is pinned to "
            f"{RESIZE_SIZE}"
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

    return {
        "run_dir": str(_canonical(run_dir)),
        "ckpt_step": step,
        "action_dim": action_dim,
        "chunk_size": model.get("action_chunk_size"),
        "resize": list(resize),
        "center_crop": list(crop_size),
        "state_dim": state_dim,
        "num_past_frames": num_past_frames,
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

    # Serving: take the action socket first.  The service owns :5556 for its
    # whole lifetime, so a busy port fails startup here (not later at Start) and
    # no other publisher can slip in between a probe and the first action.
    try:
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
        ))
    except SessionError as exc:
        print(f"[psi0-isaac-eval] cannot own the action socket {args.action_endpoint}: {exc}",
              file=sys.stderr)
        return 2
    print(f"[psi0-isaac-eval] action socket owned: {args.action_endpoint}")

    # The fine-tuned entry is validated before anything is spawned; a bad run
    # directory must not leave a policy server behind.
    try:
        fine_entry = finetuned_entry(args)
    except (LaunchError, ContractError) as exc:
        print(f"[psi0-isaac-eval] preflight failed: {exc}", file=sys.stderr)
        session.close()
        return 2

    # This process owns the policy server: a UI checkpoint switch restarts this
    # child.  A foreign server on the same port is refused (the identity check
    # would fail anyway, but not before Isaac and the model spent minutes).
    policy_port = urlsplit(args.psi0_url).port or DEFAULT_POLICY_PORT
    server = PolicyServerProcess(port=policy_port, log_path=args.policy_log)
    foreign = probe_info("127.0.0.1", policy_port)
    if foreign is not None:
        print(f"[psi0-isaac-eval] :{policy_port} already serves a policy server "
              f"({foreign.get('run_dir')!r}); refusing to start next to it", file=sys.stderr)
        session.close()
        return 2
    try:
        server.start(fine_entry.run_dir, fine_entry.step)
    except PolicyServerError as exc:
        print(f"[psi0-isaac-eval] cannot start the policy server: {exc}", file=sys.stderr)
        session.close()
        return 2
    print(f"[psi0-isaac-eval] policy server pid {server.pid} for {fine_entry.run_dir} "
          f"(step {fine_entry.step})", flush=True)

    try:
        # Materialize/validate the base entry while the 3B policy server loads.
        entries = [fine_entry, base_entry(args)]
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

        controller = CheckpointController(
            session, server, entries,
            verify=verify,
            initial_id=FINETUNED_ID,
            log=lambda message: print(f"[psi0-isaac-eval] {message}", flush=True),
        )

        from humanoid_lab.psi0_bridge.app import create_app  # imported only when serving

        app = create_app(session, webrtc=_webrtc_info(args), checkpoints=controller)
        print(f"[psi0-isaac-eval] UI on http://{args.host}:{args.port}/  "
              f"(prompt: {CANONICAL_PROMPT!r}); checkpoints: "
              f"{', '.join(entry.label for entry in entries)}", flush=True)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        server.stop()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
