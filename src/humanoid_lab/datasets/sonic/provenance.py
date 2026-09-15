"""Checksums and immutable-source guards for evidence manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sonic_model_checksums(model_dir: Path) -> dict[str, str]:
    names = ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")
    missing = [name for name in names if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing SONIC v1.1 files: {missing}")
    return {name: sha256_file(model_dir / name) for name in names}


def assert_processed_destination(source_root: Path, destination: Path) -> None:
    source = source_root.resolve()
    target = destination.resolve()
    if target == source or source in target.parents:
        raise ValueError("processed output must not be written below first_tur_ham")
