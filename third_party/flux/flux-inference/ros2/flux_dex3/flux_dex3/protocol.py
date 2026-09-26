"""Versioned Dex3 observation/action frames shared by robot and GPU processes."""

import json
import math
import re

import numpy as np

VERSION = 1
IMAGE_SHAPES = ((480, 640, 3), (192, 256, 3))
ACTION_SHAPE = (32, 28)
MAX_METADATA = 4096
MAX_IMAGE = 480 * 640 * 3
MAX_FRAME = MAX_IMAGE
MAX_SEQ = (1 << 63) - 1
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class ProtocolError(ValueError):
    """Malformed or unsupported Dex3 wire message."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate metadata field: " + key)
        result[key] = value
    return result


def _metadata(value):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ProtocolError("invalid metadata") from exc
    if len(raw) > MAX_METADATA:
        raise ProtocolError("metadata too large")
    return raw


def _decode_metadata(frame):
    if not isinstance(frame, (bytes, bytearray, memoryview)) or len(frame) > MAX_METADATA:
        raise ProtocolError("invalid metadata frame")
    try:
        value = json.loads(bytes(frame).decode("utf-8"), object_pairs_hook=_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("nonfinite metadata")))
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError("invalid metadata JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("metadata must be an object")
    if type(value.get("version")) is not int or value["version"] != VERSION:
        raise ProtocolError("unsupported protocol version")
    return value


def _fields(meta, required):
    if set(meta) != set(required):
        raise ProtocolError("unexpected or missing metadata fields")


def _session_seq(meta):
    session = meta.get("session_id")
    seq = meta.get("seq")
    if not isinstance(session, str) or not _SESSION.fullmatch(session):
        raise ProtocolError("invalid session_id")
    if type(seq) is not int or not 0 <= seq <= MAX_SEQ:
        raise ProtocolError("invalid seq")
    return session, seq


def _nonnegative_finite(value, name):
    if type(value) not in (float, int):
        raise ProtocolError("invalid " + name)
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ProtocolError("invalid " + name)
    return value


def _timestamp(value):
    return _nonnegative_finite(value, "observation_timestamp")


def _text(value, name, max_bytes):
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError("invalid " + name)
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ProtocolError("invalid " + name) from exc
    if length > max_bytes or "\x00" in value:
        raise ProtocolError("invalid " + name)
    return value


def _optional_text(value, name, max_bytes):
    if not isinstance(value, str) or "\x00" in value:
        raise ProtocolError("invalid " + name)
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ProtocolError("invalid " + name) from exc
    if length > max_bytes:
        raise ProtocolError("invalid " + name)
    return value


def _checkpoint(value):
    if not isinstance(value, str) or (value and not re.fullmatch(r"sha256:[0-9a-f]{64}", value)):
        raise ProtocolError("invalid checkpoint identity")
    return value


def _array(value, shape, dtype, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != dtype:
        raise ProtocolError("invalid " + name + " shape or dtype")
    if name != "image" and not np.isfinite(array).all():
        raise ProtocolError("nonfinite " + name)
    return np.ascontiguousarray(array)


def encode_status_request():
    return [_metadata({"version": VERSION, "type": "STATUS"})]


def encode_predict_request(session_id, seq, observation_timestamp, task, image, state):
    _session_seq({"session_id": session_id, "seq": seq})
    _timestamp(observation_timestamp)
    _text(task, "task", 1024)
    image = np.asarray(image)
    if image.shape not in IMAGE_SHAPES:
        raise ProtocolError("invalid image shape")
    image = _array(image, image.shape, np.dtype("uint8"), "image")
    state = _array(state, (28,), np.dtype("float32"), "state")
    meta = {"version": VERSION, "type": "PREDICT", "session_id": session_id, "seq": seq,
            "observation_timestamp": observation_timestamp, "task": task,
            "image_shape": list(image.shape), "image_encoding": "rgb8",
            "state_shape": [28], "state_dtype": "<f4"}
    return [_metadata(meta), image.tobytes(order="C"), state.astype("<f4", copy=False).tobytes(order="C")]


def decode_request(frames):
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= 3:
        raise ProtocolError("invalid request frame count")
    if any(not isinstance(frame, (bytes, bytearray, memoryview)) or len(frame) > MAX_FRAME for frame in frames):
        raise ProtocolError("oversized or invalid frame")
    meta = _decode_metadata(frames[0])
    if meta.get("type") == "STATUS":
        if len(frames) != 1:
            raise ProtocolError("invalid STATUS frame count")
        _fields(meta, ("version", "type"))
        return meta
    if meta.get("type") != "PREDICT" or len(frames) != 3:
        raise ProtocolError("invalid request type or frame count")
    _fields(meta, ("version", "type", "session_id", "seq", "observation_timestamp", "task",
                   "image_shape", "image_encoding", "state_shape", "state_dtype"))
    _session_seq(meta)
    _timestamp(meta["observation_timestamp"])
    _text(meta["task"], "task", 1024)
    shape = meta["image_shape"]
    if type(shape) is not list or tuple(shape) not in IMAGE_SHAPES or any(type(n) is not int for n in shape):
        raise ProtocolError("invalid image shape")
    if meta["image_encoding"] != "rgb8" or meta["state_shape"] != [28] or meta["state_dtype"] != "<f4":
        raise ProtocolError("invalid image/state format")
    if len(frames[1]) != math.prod(shape) or len(frames[2]) != 28 * 4:
        raise ProtocolError("invalid image/state length")
    result = dict(meta)
    result["image"] = np.frombuffer(frames[1], dtype=np.uint8).reshape(shape).copy()
    result["state"] = np.frombuffer(frames[2], dtype="<f4").copy()
    if not np.isfinite(result["state"]).all():
        raise ProtocolError("nonfinite state")
    return result


def encode_status_reply(status, checkpoint="", error=""):
    if status not in ("LOADING", "READY", "ERROR"):
        raise ProtocolError("invalid status")
    _checkpoint(checkpoint)
    _optional_text(error, "error", 512)
    return [_metadata({"version": VERSION, "type": "STATUS", "status": status,
                       "checkpoint": checkpoint, "error": error})]


def encode_predict_reply(session_id, seq, actions, inference_ms):
    _session_seq({"session_id": session_id, "seq": seq})
    _nonnegative_finite(inference_ms, "inference_ms")
    actions = _array(actions, ACTION_SHAPE, np.dtype("float32"), "actions")
    meta = {"version": VERSION, "type": "PREDICT", "session_id": session_id, "seq": seq,
            "inference_ms": inference_ms, "action_shape": [32, 28], "action_dtype": "<f4"}
    return [_metadata(meta), actions.astype("<f4", copy=False).tobytes(order="C")]


def encode_error_reply(error, session_id=None, seq=None):
    _text(error, "error", 512)
    if session_id is not None or seq is not None:
        _session_seq({"session_id": session_id, "seq": seq})
    return [_metadata({"version": VERSION, "type": "ERROR", "error": error,
                       "session_id": session_id, "seq": seq})]


def decode_reply(frames):
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= 2:
        raise ProtocolError("invalid reply frame count")
    if any(not isinstance(frame, (bytes, bytearray, memoryview)) or len(frame) > MAX_FRAME for frame in frames):
        raise ProtocolError("oversized or invalid reply frame")
    meta = _decode_metadata(frames[0])
    kind = meta.get("type")
    if kind == "STATUS" and len(frames) == 1:
        _fields(meta, ("version", "type", "status", "checkpoint", "error"))
        if meta["status"] not in ("LOADING", "READY", "ERROR"):
            raise ProtocolError("invalid status")
        _checkpoint(meta["checkpoint"])
        _optional_text(meta["error"], "error", 512)
        return meta
    if kind == "ERROR" and len(frames) == 1:
        _fields(meta, ("version", "type", "error", "session_id", "seq"))
        _text(meta["error"], "error", 512)
        if meta["session_id"] is not None or meta["seq"] is not None:
            _session_seq(meta)
        return meta
    if kind != "PREDICT" or len(frames) != 2:
        raise ProtocolError("invalid reply type or frame count")
    _fields(meta, ("version", "type", "session_id", "seq", "inference_ms", "action_shape", "action_dtype"))
    _session_seq(meta)
    _nonnegative_finite(meta["inference_ms"], "inference_ms")
    if meta["action_shape"] != [32, 28] or meta["action_dtype"] != "<f4" or len(frames[1]) != 32 * 28 * 4:
        raise ProtocolError("invalid actions format or length")
    result = dict(meta)
    result["actions"] = np.frombuffer(frames[1], dtype="<f4").reshape(ACTION_SHAPE).copy()
    if not np.isfinite(result["actions"]).all():
        raise ProtocolError("nonfinite actions")
    return result
