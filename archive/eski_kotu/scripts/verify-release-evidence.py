#!/usr/bin/env python3
"""Fail-closed validator for CloudWalk release acceptance evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_COUNTS = {"adapter": 1, "groot_replay": 2, "sonic_native": 2, "isaac_closed_loop": 2, "render": 1, "rollout": 5}
PHASES = ("approach", "contact", "hand_close", "stable_grasp", "lift")
CLOSED_LOOP_EVENTS = {"live_rgb_state", "upstream_groot_policy", "upstream_v4_serializer", "native_sonic_29_body", "hands_24_joint_applied"}


class EvidenceError(ValueError):
    pass


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise EvidenceError(f"missing file: {path}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid JSON in {path}: {error}") from error


def _records(path: Path) -> list[dict]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError as error:
        raise EvidenceError(f"missing log: {path}") from error
    records = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise EvidenceError(f"{label} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise EvidenceError(f"{label} escapes repository root: {value}") from error
    if not path.is_file() or path.stat().st_size == 0:
        raise EvidenceError(f"{label} is missing or empty: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(root), *args), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def _validate_record(item: dict, kind: str, index: int, root: Path, commit: str, cutoff: float, hashes: dict[str, str]) -> tuple[Path, list[dict]]:
    label = f"{kind}[{index}]"
    if item.get("commit") != commit:
        raise EvidenceError(f"{label} is not bound to release commit {commit}")
    created = item.get("created_unix")
    if not isinstance(created, (int, float)) or isinstance(created, bool) or created < cutoff or created > time.time() + 300:
        raise EvidenceError(f"{label} has stale or invalid created_unix")
    command = item.get("command")
    if not isinstance(command, str) or not command.strip():
        raise EvidenceError(f"{label} is missing its executable command")
    artifact = _resolve(root, item.get("artifact"), f"{label}.artifact")
    if artifact.stat().st_mtime < cutoff:
        raise EvidenceError(f"{label} artifact is stale")
    digest = _sha256(artifact)
    if digest in hashes:
        raise EvidenceError(f"duplicate evidence content: {label} and {hashes[digest]}")
    hashes[digest] = label
    attachments = item.get("attachments", [])
    if not isinstance(attachments, list):
        raise EvidenceError(f"{label}.attachments must be a list")
    for attachment_index, value in enumerate(attachments):
        attached = _resolve(root, value, f"{label}.attachments[{attachment_index}]")
        attached_digest = _sha256(attached)
        if attached_digest in hashes:
            raise EvidenceError(f"duplicate evidence content: {label}.attachments[{attachment_index}] and {hashes[attached_digest]}")
        hashes[attached_digest] = f"{label}.attachments[{attachment_index}]"
    return artifact, _records(artifact) if artifact.suffix in {".log", ".jsonl"} else []


def _validate_adapter(path: Path, records: list[dict]) -> None:
    data = records[-1] if records else _load_json(path)
    if not isinstance(data, dict) or data.get("result") != "PASS" or data.get("tests_run", 0) < 1:
        raise EvidenceError("adapter evidence must report PASS and a positive tests_run")


def _validate_replay(records: list[dict]) -> None:
    passed = [record for record in records if record.get("result") == "PASS"]
    if len(passed) != 1 or passed[0].get("shape") != [1, 40, 78] or passed[0].get("finite") is not True:
        raise EvidenceError("each GR00T replay requires exactly one finite PASS [1,40,78] record")


def _validate_native(path: Path) -> None:
    text = path.read_text(errors="replace")
    if text.count("PASS isolated native harness dds=0 motor_transport=0") != 1 or "FAIL " in text:
        raise EvidenceError("each native SONIC run requires one isolated PASS and no FAIL marker")


def _validate_closed_loop(records: list[dict]) -> None:
    complete = [record for record in records if record.get("event") == "closed_loop"]
    results = [record for record in records if record.get("vla_connection") == "upstream_policy_native_sonic_connected"]
    if len(complete) != 1 or len(results) != 1:
        raise EvidenceError("each Isaac run requires one closed_loop event and one connected final record")
    event = complete[0]
    if event.get("inference_frames", 0) < 2 or event.get("native_body_frames", 0) < 2:
        raise EvidenceError("Isaac closed loop did not prove repeated GR00T inference and native body frames")
    if not CLOSED_LOOP_EVENTS.issubset(set(results[0].get("events", []))):
        raise EvidenceError("Isaac final record is missing required layer events")


def _ffprobe(path: Path) -> dict:
    command = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,duration", "-of", "json", str(path))
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise EvidenceError(f"cannot probe rollout video {path}: {error}") from error
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise EvidenceError(f"rollout video has no unique video stream: {path}")
    return streams[0]


def _validate_media(path: Path, minimum_bytes: int) -> None:
    if path.stat().st_size < minimum_bytes:
        raise EvidenceError(f"media artifact is implausibly small: {path}")


def _image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9}:
                continue
            if offset + 2 > len(data):
                break
            length = int.from_bytes(data[offset:offset + 2], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and offset + 7 <= len(data):
                return int.from_bytes(data[offset + 5:offset + 7], "big"), int.from_bytes(data[offset + 3:offset + 5], "big")
            offset += max(length, 2)
    raise EvidenceError(f"unsupported or corrupt rendered frame: {path}")


def _validate_render(item: dict, path: Path) -> None:
    _validate_media(path, 10_000)
    width, height = _image_size(path)
    if width < 640 or height < 480:
        raise EvidenceError("rendered frame must be at least 640x480")
    review = item.get("visual_review")
    if not isinstance(review, dict) or review.get("contents") != {"robot": True, "table": True, "bottle": True} or review.get("realistic") is not True or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise EvidenceError("render evidence requires a named realistic robot/table/bottle visual review")


def _validate_rollout(item: dict, video: Path, root: Path, commit: str, cutoff: float, hashes: dict[str, str], index: int) -> None:
    metrics = _resolve(root, item.get("metrics"), f"rollout[{index}].metrics")
    if metrics.stat().st_mtime < cutoff:
        raise EvidenceError(f"rollout[{index}] metrics are stale")
    digest = _sha256(metrics)
    if digest in hashes:
        raise EvidenceError(f"duplicate rollout metrics: rollout[{index}] and {hashes[digest]}")
    hashes[digest] = f"rollout[{index}].metrics"
    data = _load_json(metrics)
    if not isinstance(data, dict):
        raise EvidenceError(f"rollout[{index}] metrics must be an object")
    phases = item.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES) or not all(isinstance(phases[name], bool) for name in PHASES):
        raise EvidenceError(f"rollout[{index}] requires boolean annotations for every phase")
    expected = {"approach": data.get("approach"), "contact": data.get("contact_proxy"), "hand_close": data.get("hand_close"), "stable_grasp": data.get("stable_grasp"), "lift": data.get("lift")}
    if phases != expected:
        raise EvidenceError(f"rollout[{index}] phase annotations disagree with metrics")
    failure_layer = item.get("failure_layer")
    if all(phases.values()):
        if failure_layer is not None:
            raise EvidenceError(f"rollout[{index}] successful rollout cannot declare a failure layer")
    elif not isinstance(failure_layer, str) or not failure_layer.strip():
        raise EvidenceError(f"rollout[{index}] must classify the first failed layer")
    elif failure_layer != next(name for name in PHASES if not phases[name]):
        raise EvidenceError(f"rollout[{index}] failure_layer is not the first failed phase")
    if data.get("stable_grasp_frames", 0) < 10 and phases["stable_grasp"]:
        raise EvidenceError(f"rollout[{index}] stable grasp lacks ten frames")
    if data.get("max_lift_m", 0) < 0.05 and phases["lift"]:
        raise EvidenceError(f"rollout[{index}] lift is below 5 cm")
    if not all(phases.values()):
        raise EvidenceError(f"rollout[{index}] failed at layer {failure_layer}")
    stream = _ffprobe(video)
    if int(stream.get("width", 0)) < 320 or int(stream.get("height", 0)) < 240:
        raise EvidenceError(f"rollout[{index}] video resolution is too small")
    duration = float(stream.get("duration", 0) or 0)
    frames = int(stream.get("nb_frames", 0) or 0)
    if duration < 2 or frames < 20:
        raise EvidenceError(f"rollout[{index}] video is too short")


def validate(manifest_path: Path, max_age_hours: float) -> dict:
    root = Path(__file__).resolve().parents[1]
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise EvidenceError("release manifest schema must be 1")
    commit = manifest.get("commit")
    if not isinstance(commit, str) or len(commit) != 40 or commit != _git(root, "rev-parse", "HEAD"):
        raise EvidenceError("manifest commit must equal the checked-out HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise EvidenceError("tracked workspace changes make release evidence non-reproducible")
    cutoff = time.time() - max_age_hours * 3600
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_COUNTS):
        raise EvidenceError(f"manifest gates must be exactly {sorted(REQUIRED_COUNTS)}")
    hashes: dict[str, str] = {}
    summaries = []
    for kind, count in REQUIRED_COUNTS.items():
        items = gates[kind]
        if not isinstance(items, list) or len(items) != count:
            raise EvidenceError(f"gate {kind} requires exactly {count} evidence record(s)")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise EvidenceError(f"{kind}[{index}] must be an object")
            artifact, records = _validate_record(item, kind, index, root, commit, cutoff, hashes)
            if kind == "adapter": _validate_adapter(artifact, records)
            elif kind == "groot_replay": _validate_replay(records)
            elif kind == "sonic_native": _validate_native(artifact)
            elif kind == "isaac_closed_loop": _validate_closed_loop(records)
            elif kind == "render": _validate_render(item, artifact)
            elif kind == "rollout":
                _validate_media(artifact, 50_000)
                _validate_rollout(item, artifact, root, commit, cutoff, hashes, index)
            summaries.append({"gate": kind, "artifact": str(artifact.relative_to(root)), "sha256": _sha256(artifact)})
    return {"result": "PASS", "commit": commit, "max_age_hours": max_age_hours, "evidence": summaries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    args = parser.parse_args()
    if args.max_age_hours <= 0:
        raise EvidenceError("--max-age-hours must be positive")
    print(json.dumps(validate(args.manifest, args.max_age_hours), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
