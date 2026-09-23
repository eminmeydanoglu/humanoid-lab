"""Exact GR00T request capture hooks for the transformed SONIC runner."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from humanoid_lab.groot_inference_capture import GrootInferenceCapture

_lock = threading.Lock()
_capture: GrootInferenceCapture | None = None
_capture_dir: Path | None = None
_request_sequence = 0
_chunk_sequence = 0
_links: dict[int, dict[str, Any]] = {}


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _configure() -> GrootInferenceCapture | None:
    global _capture, _capture_dir
    if _capture is not None:
        return _capture
    raw = os.environ.get("HUMANOID_GROOT_CAPTURE_DIR")
    if not raw:
        return None
    _capture_dir = Path(raw)
    checkpoint = os.environ.get("HUMANOID_GROOT_CHECKPOINT")
    del checkpoint
    _capture = GrootInferenceCapture(
        _capture_dir,
        max_requests=int(os.environ.get("HUMANOID_GROOT_CAPTURE_MAX_REQUESTS", "32")),
    )
    return _capture


def capture_get_action(policy: Any, observation: Mapping[str, Any]):
    """Call PolicyClient.get_action once and losslessly capture both sides."""
    global _request_sequence
    capture = _configure()
    if capture is None:
        return policy.get_action(observation)
    with _lock:
        sequence = _request_sequence
        _request_sequence += 1
    send_id = f"vla-send-{sequence:06d}"
    receive_id = f"vla-receive-{sequence:06d}"
    language = observation.get("language", {})
    prompt = language.get("annotation.human.task_description", [[""]])
    try:
        prompt = str(prompt[0][0])
    except (IndexError, TypeError):
        prompt = ""
    checkpoint_path = os.environ.get("HUMANOID_GROOT_CHECKPOINT")
    checkpoint_dir = None if not checkpoint_path else Path(checkpoint_path)
    checkpoint = {
        "path": checkpoint_path,
        "config_sha256": None if checkpoint_dir is None else _sha256(checkpoint_dir / "config.json"),
        "processor_config_sha256": None if checkpoint_dir is None else _sha256(checkpoint_dir / "processor_config.json"),
        "statistics_sha256": None if checkpoint_dir is None else _sha256(checkpoint_dir / "statistics.json"),
    }
    pending = capture.begin(
        observation,
        prompt=prompt,
        embodiment_tag="unitree_g1_sonic",
        options={
            "left_hand_contract": os.environ.get(
                "HUMANOID_GROOT_LEFT_HAND_CONTRACT", "compatibility"
            ),
            "policy_clock": os.environ.get("HUMANOID_POLICY_CLOCK", "wall"),
        },
        source_stamps={
            "image_source": observation.get("timestamps"),
            "state_source": None,
            "projected_gravity_source": None,
            "note": "image timestamp is producer-provided; state and projected-gravity source stamps are unavailable",
        },
        checkpoint=checkpoint,
        send_id=send_id,
    )
    action, info = policy.get_action(observation)
    output = capture.finish(
        pending,
        action,
        info=info,
        receive_id=receive_id,
        execute_id=None,
        execute_note="execution mapping is recorded in execution-map.jsonl by request_id/chunk_version",
    )
    link = {
        "request_id": None if pending is None else pending.get("request_id"),
        "send_id": send_id,
        "receive_id": receive_id,
        "capture": None if output is None else output.name,
    }
    with _lock:
        _links[id(action)] = link
    return action, info


def bind_processed_action(raw_action: Mapping[str, Any], processed_action: Mapping[str, Any]) -> None:
    with _lock:
        link = _links.pop(id(raw_action), None)
        if link is not None:
            _links[id(processed_action)] = link


def chunk_installed(processed_action: Mapping[str, Any], start_row: int, inference_delay_s: float) -> None:
    global _chunk_sequence
    with _lock:
        link = _links.get(id(processed_action))
        if link is None:
            return
        _chunk_sequence += 1
        link["chunk_version"] = _chunk_sequence
        link["installed_start_row"] = int(start_row)
        link["inference_delay_s"] = float(inference_delay_s)
        _append({"kind": "chunk_installed", **link, "wall_time_ns": time.time_ns()})


def row_executed(processed_action: Mapping[str, Any], row: int, frame_index: int) -> None:
    with _lock:
        link = _links.get(id(processed_action))
        if link is None:
            return
        _append({
            "kind": "row_executed", **link, "row": int(row),
            "frame_index": int(frame_index), "wall_time_ns": time.time_ns(),
        })


def _append(record: Mapping[str, Any]) -> None:
    if _capture_dir is None:
        return
    path = _capture_dir / "execution-map.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
