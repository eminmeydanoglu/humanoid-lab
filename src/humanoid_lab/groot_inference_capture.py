"""Lossless, opt-in capture of one GR00T policy request and response."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1


def _arrays(value: Any, prefix: str, output: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        key = prefix or "root"
        array = np.ascontiguousarray(value)
        output[key] = array
        return {"array": key, "dtype": array.dtype.str, "shape": list(array.shape)}
    if isinstance(value, Mapping):
        return {str(k): _arrays(v, f"{prefix}.{k}" if prefix else str(k), output) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_arrays(v, f"{prefix}.{i}" if prefix else str(i), output) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported capture value at {prefix or '<root>'}: {type(value).__name__}")


def _restore(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, dict) and set(value) == {"array", "dtype", "shape"}:
        array = arrays[value["array"]]
        if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
            raise ValueError(f"array metadata mismatch for {value['array']}")
        return array
    if isinstance(value, dict):
        return {k: _restore(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v, arrays) for v in value]
    return value


class GrootInferenceCapture:
    """Write bounded request bundles; no directory means capture is disabled."""

    def __init__(self, directory: Path | None, *, max_requests: int = 32) -> None:
        self.directory = None if directory is None else Path(directory)
        self.max_requests = int(max_requests)
        self._count = 0
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None and self._count < self.max_requests

    def begin(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str,
        embodiment_tag: str,
        options: Mapping[str, Any] | None,
        source_stamps: Mapping[str, Any] | None,
        checkpoint: Mapping[str, Any],
        send_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        request_id = str(uuid.uuid4())
        arrays: dict[str, np.ndarray] = {}
        observation_tree = _arrays(observation, "observation", arrays)
        send_time_ns = time.time_ns()
        token = {
            "request_id": request_id,
            "send_id": send_id,
            "send_time_ns": send_time_ns,
            "observation_tree": observation_tree,
            "arrays": arrays,
            "prompt": prompt,
            "embodiment_tag": embodiment_tag,
            "options": dict(options or {}),
            "source_stamps": dict(source_stamps or {}),
            "checkpoint": dict(checkpoint),
        }
        self._count += 1
        return token

    def finish(
        self,
        token: dict[str, Any] | None,
        action: Mapping[str, Any],
        *,
        info: Mapping[str, Any] | None = None,
        receive_id: str | None = None,
        execute_id: str | None = None,
        execute_note: str | None = None,
    ) -> Path | None:
        if token is None or self.directory is None:
            return None
        arrays = dict(token.pop("arrays"))
        action_tree = _arrays(action, "action", arrays)
        receive_time_ns = time.time_ns()
        name = f"request-{self._count - 1:06d}-{token['request_id']}"
        final = self.directory / name
        temporary = self.directory / f".{name}.tmp-{os.getpid()}"
        temporary.mkdir()
        npz_path = temporary / "arrays.npz"
        np.savez(npz_path, **arrays)
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            **token,
            "receive_id": receive_id,
            "receive_time_ns": receive_time_ns,
            "execute_id": execute_id,
            "execute_note": execute_note,
            "action_tree": action_tree,
            "info": dict(info or {}),
            "arrays_file": "arrays.npz",
            "arrays_sha256": digest,
            "timing_note": "send/receive are capture wall-clock events; source stamps are producer-provided; execute is unknown unless the action owner records it later",
        }
        meta_tmp = temporary / "metadata.json.tmp"
        meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(meta_tmp, temporary / "metadata.json")
        os.replace(temporary, final)
        return final


def load_capture(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    path = directory / metadata["arrays_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["arrays_sha256"]:
        raise ValueError("capture arrays checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return {"metadata": metadata, "observation": _restore(metadata["observation_tree"], arrays), "action": _restore(metadata["action_tree"], arrays)}
