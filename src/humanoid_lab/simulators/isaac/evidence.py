"""Bounded-memory evidence writer for Isaac G1 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class StreamingVideoWriter:
    """Write each frame immediately and retain only the previous RGB frame."""

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = fps
        self._writer: Any = None
        self._previous: Any = None
        self.frames = 0
        self.changed_pixels = 0
        self.hashes: list[str] = []
        self.shape: list[int] | None = None
        self.error: str | None = None

    @staticmethod
    def normalize(frame: Any) -> Any:
        import numpy as np

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] not in (3, 4) and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"unexpected frame shape {array.shape}")
        return np.ascontiguousarray(array[..., :3].astype(np.uint8, copy=False)).copy()

    def append(self, frame: Any) -> dict[str, Any] | None:
        if self.error is not None:
            return None
        try:
            import numpy as np

            try:
                image = self.normalize(frame)
            except ValueError as exc:
                if "shape (0,)" in str(exc):
                    return None
                raise
            if self._writer is None:
                import imageio.v2 as imageio

                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = imageio.get_writer(
                    str(self.path), fps=self.fps, codec="libx264", macro_block_size=None
                )
                self.shape = [int(value) for value in image.shape]
            changed = 0 if self._previous is None else int(
                np.count_nonzero(np.any(image != self._previous, axis=-1))
            )
            digest = hashlib.sha256(memoryview(image)).hexdigest()
            self._writer.append_data(image)
            self._previous = image
            self.frames += 1
            self.changed_pixels += changed
            if len(self.hashes) < 4:
                self.hashes.append(digest)
            return {"frame_hash": digest, "changed_pixels": changed}
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return None

    def close(self) -> dict[str, Any]:
        if self._writer is not None:
            self._writer.close()
        return {
            "path": str(self.path),
            "frames": self.frames,
            "changed_pixels": self.changed_pixels,
            "sample_hashes": self.hashes,
            "shape": self.shape,
            "bytes": self.path.stat().st_size if self.path.is_file() else 0,
            "error": self.error,
        }


class EvidenceSink:
    def __init__(self, run_dir: Path, *, record: bool, fps: int) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status = JsonlWriter(run_dir / "component-status.jsonl")
        self.metrics = JsonlWriter(run_dir / "simulator-metrics.jsonl")
        self.policy_metrics = JsonlWriter(run_dir / "policy-metrics.jsonl")
        self.controller_metrics = JsonlWriter(run_dir / "controller-metrics.jsonl")
        self.head_video = StreamingVideoWriter(run_dir / "head-camera.mp4", fps) if record else None
        self.viewport_video = StreamingVideoWriter(run_dir / "debug-viewport.mp4", fps) if record else None

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def write_manifest(self, payload: dict[str, Any]) -> None:
        import yaml

        (self.run_dir / "manifest.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
        )

    def close(self) -> dict[str, Any]:
        self.status.close()
        self.metrics.close()
        self.policy_metrics.close()
        self.controller_metrics.close()
        return {
            "head_camera": self.head_video.close() if self.head_video else {"status": "disabled"},
            "debug_viewport": self.viewport_video.close() if self.viewport_video else {"status": "disabled"},
        }
