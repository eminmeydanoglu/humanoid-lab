#!/usr/bin/env python3
"""Proof that the SONIC controller consumes the bridge's Protocol v4 tokens.

Observer only.  Two modes:

* **online** (default): SUBs to :5556 (the bridge's ``pose`` stream) and :5557
  (SONIC's ``g1_debug`` state, which republishes the external ``token_state`` the
  controller copied) and writes a retained NDJSON stream that carries *both*
  event kinds with an explicit ``kind`` field:

      {"kind": "published", "seq": 0, "t_wall": ..., "t_mono": ..., "frame_index": 0,
       "token": [64 floats], "left_hand": [7], "right_hand": [7]}
      {"kind": "consumed", "seq": 1, "t_wall": ..., "t_mono": ...,
       "token_state": [64 floats], "source": {"state_endpoint": ..., "state_topic": "g1_debug"},
       "matched_frame_index": 3, "latest_published_frame_index": 5, "lag_frames": 2,
       "max_abs_diff": 0.0, "matched": true, "reason": null}

* **offline** (``--offline-stream``): opens no socket at all — it reads only the
  retained NDJSON and recomputes every metric from the raw arrays (window match,
  tolerance, lag, unmatched reasons).  It also verifies the per-event match
  fields the online pass wrote, so online and offline results can be compared
  field by field.

    # online (live stack)
    python3 scripts/psi0-token-match.py --seconds 45 \
        --json <run>/token-match.json --stream <run>/token-stream.ndjson
    # offline (no ZMQ), same script
    python3 scripts/psi0-token-match.py --offline-stream <run>/token-stream.ndjson \
        --json <run>/offline-token-match.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

POSE_ENDPOINT = "tcp://127.0.0.1:5556"
STATE_ENDPOINT = "tcp://127.0.0.1:5557"
STATE_TOPIC = "g1_debug"
STATE_TOPIC_BYTES = STATE_TOPIC.encode()
HEADER_SIZE = 1280
TOLERANCE = 1e-6
WINDOW = 400  # recent published tokens a consumed sample may match against
TOKEN_DIM = 64
MATCH_RULE = (
    "a consumed sample matches when its 64D token_state equals a published token "
    "within tolerance; the matched frame_index is the closest such published token and "
    "lag_frames = latest published frame_index observed before the sample - matched frame_index"
)


class StreamError(ValueError):
    """The retained stream is malformed or has the wrong shape."""


def _decode_pose(message: bytes) -> dict:
    header = json.loads(message[4:4 + HEADER_SIZE].rstrip(b"\0"))
    payload = message[4 + HEADER_SIZE:]
    offset = 0
    fields: dict[str, np.ndarray] = {}
    for field in header["fields"]:
        size = int(np.prod(field["shape"]))
        dtype = {"f32": "<f4", "i64": "<i8"}[field["dtype"]]
        fields[field["name"]] = np.frombuffer(payload, dtype=dtype, count=size, offset=offset).copy()
        offset += size * np.dtype(dtype).itemsize
    return {
        "frame_index": int(fields["frame_index"][0]),
        "token": fields["token_state"].reshape(-1).astype(np.float32),
        "left_hand": fields.get("left_hand_joints", np.zeros(0, dtype=np.float32)).reshape(-1),
        "right_hand": fields.get("right_hand_joints", np.zeros(0, dtype=np.float32)).reshape(-1),
    }


def _match_one(window: deque, latest_index: int | None, token: np.ndarray) -> dict:
    """Match one consumed token against the published window (shared by both modes)."""
    if not window:
        return {
            "matched": False, "reason": "no published token observed yet",
            "matched_frame_index": None, "latest_published_frame_index": None,
            "lag_frames": None, "max_abs_diff": None,
        }
    diffs = [(float(np.abs(candidate - token).max()), index) for index, candidate in window]
    best_diff, best_index = min(diffs)
    return {
        "matched": best_diff <= TOLERANCE,
        "reason": None if best_diff <= TOLERANCE else "diff_above_tolerance",
        "matched_frame_index": best_index,
        "latest_published_frame_index": latest_index,
        "lag_frames": None if latest_index is None else latest_index - best_index,
        "max_abs_diff": best_diff,
    }


def _match_events(events) -> dict:
    """Shared matcher: consumed samples against the published window, in event order."""
    window: deque[tuple[int, np.ndarray]] = deque(maxlen=WINDOW)
    latest_index: int | None = None
    published = 0
    consumed: list[dict] = []
    for event in events:
        kind = event["kind"]
        if kind == "published":
            published += 1
            latest_index = int(event["frame_index"])
            window.append((latest_index, np.asarray(event["token"], dtype=np.float32)))
            continue
        if kind != "consumed":
            raise StreamError(f"unknown event kind {kind!r}")
        token = np.asarray(event["token_state"], dtype=np.float32)
        sample = {"t_mono": event["t_mono"], "seq": event["seq"], "token_state": token,
                  "source": event.get("source")}
        sample.update(_match_one(window, latest_index, token))
        consumed.append(sample)

    matched = [s for s in consumed if s["matched"]]
    unmatched = [s for s in consumed if not s["matched"]]
    reasons: dict[str, int] = {}
    for sample in unmatched:
        key = sample.get("reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    diffs = [s["max_abs_diff"] for s in consumed if s["max_abs_diff"] is not None]
    lags = [s["lag_frames"] for s in matched if s["lag_frames"] is not None]
    frame_indices = [int(e["frame_index"]) for e in events if e["kind"] == "published"]
    return {
        "tolerance": TOLERANCE,
        "published_count": published,
        "published_frame_index": {
            "first": frame_indices[0] if frame_indices else None,
            "last": frame_indices[-1] if frame_indices else None,
        },
        "consumed_samples": len(consumed),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "unmatched_reasons": reasons,
        "max_abs_diff": max(diffs, default=None),
        "matched_max_abs_diff": max((s["max_abs_diff"] for s in matched), default=None),
        "lag_frames": {"min": min(lags, default=None), "max": max(lags, default=None)},
        "match_rule": MATCH_RULE,
    }, consumed


def load_stream(path: Path) -> list[dict]:
    """Read a retained NDJSON stream, rejecting malformed lines and wrong widths."""
    events: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StreamError(f"{path}:{number}: not JSON: {exc}") from exc
        if not isinstance(event, dict) or "kind" not in event:
            raise StreamError(f"{path}:{number}: missing kind")
        if not isinstance(event.get("seq"), int) or not isinstance(event.get("t_mono"), (int, float)):
            raise StreamError(f"{path}:{number}: missing monotonic order/timestamp")
        kind = event["kind"]
        if kind == "published":
            token = event.get("token")
            if not isinstance(event.get("frame_index"), int):
                raise StreamError(f"{path}:{number}: published event without frame_index")
        elif kind == "consumed":
            token = event.get("token_state")
        else:
            raise StreamError(f"{path}:{number}: unknown kind {kind!r}")
        if not isinstance(token, list) or len(token) != TOKEN_DIM:
            raise StreamError(
                f"{path}:{number}: {kind} token width is "
                f"{len(token) if isinstance(token, list) else type(token).__name__}, expected {TOKEN_DIM}"
            )
        try:
            np.asarray(token, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise StreamError(f"{path}:{number}: token is not numeric: {exc}") from exc
        events.append(event)
    if not events:
        raise StreamError(f"{path}: stream is empty")
    return events


def run_offline(stream_path: Path) -> dict:
    """Recompute every metric from the retained stream; no sockets are opened."""
    events = load_stream(stream_path)
    metrics, consumed = _match_events(events)
    stored = [e for e in events if e["kind"] == "consumed"]
    mismatched = []
    for recomputed, original in zip(consumed, stored):
        for key in ("matched", "matched_frame_index", "latest_published_frame_index",
                    "lag_frames", "reason"):
            if original.get(key) != recomputed.get(key):
                mismatched.append({"seq": original["seq"], "field": key,
                                   "stored": original.get(key), "recomputed": recomputed.get(key)})
    metrics["mode"] = "offline"
    metrics["stream"] = stream_path.name
    metrics["stream_sha256"] = hashlib.sha256(stream_path.read_bytes()).hexdigest()
    metrics["verified_events"] = len(consumed)
    metrics["event_field_mismatches"] = mismatched
    return metrics


def run_online(args: argparse.Namespace) -> dict:
    """Observe the live stack and retain both published and consumed events."""
    import msgpack  # noqa: PLC0415 - only the online path needs the wire codec
    import zmq  # noqa: PLC0415 - offline mode must not touch ZMQ at all

    context = zmq.Context()
    pose = context.socket(zmq.SUB)
    pose.setsockopt(zmq.LINGER, 0)
    pose.setsockopt_string(zmq.SUBSCRIBE, "pose")
    pose.setsockopt(zmq.CONFLATE, 1)
    pose.connect(args.pose_endpoint)
    state = context.socket(zmq.SUB)
    state.setsockopt(zmq.LINGER, 0)
    state.setsockopt(zmq.SUBSCRIBE, STATE_TOPIC_BYTES)
    state.setsockopt(zmq.CONFLATE, 1)
    state.connect(args.state_endpoint)
    poller = zmq.Poller()
    poller.register(pose, zmq.POLLIN)
    poller.register(state, zmq.POLLIN)

    events: list[dict] = []
    window: deque[tuple[int, np.ndarray]] = deque(maxlen=WINDOW)
    latest_index: int | None = None
    started_mono = time.monotonic()
    stream = args.stream.open("w", encoding="utf-8")
    try:
        while time.monotonic() - started_mono < args.seconds:
            ready = dict(poller.poll(2))
            if pose in ready:
                decoded = _decode_pose(pose.recv())
                latest_index = decoded["frame_index"]
                window.append((latest_index, decoded["token"]))
                events.append({
                    "kind": "published",
                    "seq": len(events),
                    "t_wall": round(time.time(), 6),
                    "t_mono": round(time.monotonic() - started_mono, 6),
                    "frame_index": decoded["frame_index"],
                    "token": [round(float(v), 6) for v in decoded["token"]],
                    "left_hand": [round(float(v), 6) for v in decoded["left_hand"]],
                    "right_hand": [round(float(v), 6) for v in decoded["right_hand"]],
                })
                stream.write(json.dumps(events[-1]) + "\n")
                if len(events) % 50 == 0:
                    stream.flush()
            if state in ready:
                raw = state.recv()
                payload = msgpack.unpackb(raw[len(STATE_TOPIC_BYTES):], raw=False)
                token = np.asarray(payload.get("token_state", []), dtype=np.float32).reshape(-1)
                if token.size == TOKEN_DIM:
                    # Write the match result *at this moment*: the reviewer needs the
                    # matched/latest published frame_index next to the raw token.
                    match = _match_one(window, latest_index, token)
                    events.append({
                        "kind": "consumed",
                        "seq": len(events),
                        "t_wall": round(time.time(), 6),
                        "t_mono": round(time.monotonic() - started_mono, 6),
                        "token_state": [round(float(v), 6) for v in token],
                        "source": {"state_endpoint": args.state_endpoint, "state_topic": STATE_TOPIC},
                        **match,
                    })
                    stream.write(json.dumps(events[-1]) + "\n")
    finally:
        stream.close()
        pose.close(linger=0)
        state.close(linger=0)
        context.term()

    metrics, consumed = _match_events(events)
    metrics["mode"] = "online"
    metrics["duration_s"] = round(time.monotonic() - started_mono, 3)
    metrics["endpoints"] = {"pose": args.pose_endpoint, "state": args.state_endpoint}
    metrics["stream"] = args.stream.name
    metrics["stream_sha256"] = hashlib.sha256(args.stream.read_bytes()).hexdigest()
    metrics["match_samples"] = [
        {key: sample[key] for key in ("seq", "matched", "matched_frame_index",
                                      "latest_published_frame_index", "lag_frames",
                                      "max_abs_diff", "reason")}
        for sample in consumed[:3] + consumed[-3:]
    ]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="metrics output")
    parser.add_argument("--stream", type=Path, help="online: NDJSON output path")
    parser.add_argument("--offline-stream", type=Path,
                        help="offline: retained NDJSON to recompute from (no sockets)")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--pose-endpoint", default=POSE_ENDPOINT)
    parser.add_argument("--state-endpoint", default=STATE_ENDPOINT)
    args = parser.parse_args()

    try:
        if args.offline_stream is not None:
            metrics = run_offline(args.offline_stream)
        else:
            if args.stream is None:
                parser.error("--stream is required in online mode")
            metrics = run_online(args)
    except StreamError as exc:
        print(f"psi0-token-match: {exc}", file=__import__("sys").stderr)
        return 2

    args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in
                      ("mode", "published_count", "consumed_samples", "matched", "unmatched",
                       "max_abs_diff", "lag_frames")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
