#!/usr/bin/env python3
"""Create the immutable read-only adapter copy the GPU server will serve.

The ZMQ server refuses writable or symlinked checkpoints and hashes every
checkpoint-owned file before and after warm-up.  This script copies only the
adapter and processor artifacts the server requires -- never the training
optimizer, EMA or RNG state -- into a local directory, drops write permission,
and verifies the result with the server's own ``checkpoint_identity``, so the
digest printed here is the identity the run will report.

The base policy the adapter references is **read-only external state**.  If the
adapter config points at a path that no longer exists (for example the absolute
home path of the machine that trained it), pass ``--base-model-dir`` to point
the *copied* adapter config at the new location; the source checkout is never
modified, and the base policy is never copied.  The base directory is checked
for the required files, and its weight file can be pinned with
``--base-model-sha256`` (that verification reads the whole weight file, so it is
opt-in).  A small provenance sidecar is written next to the served directory
(never inside it, so ``checkpoint_identity`` stays exactly what the server
verifies).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

REQUIRED = ("adapter_config.json", "adapter_model.safetensors",
            "policy_preprocessor.json", "policy_postprocessor.json")
PROCESSORS = ("policy_preprocessor", "policy_postprocessor")
#: A base policy export needs its config and one weight file; the processor
#: files beside them belong to the adapter copy, not to the base.
BASE_CONFIG = "config.json"
BASE_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
LEROBOT_COMMIT = "e624f3f7f8411ec3a02635d06e79373341e5ef35"


class PrepareError(Exception):
    """The source checkpoint, the base policy or the destination is unusable."""


def checkpoint_files(source: Path) -> list[Path]:
    """The checkpoint-owned files the server hashes, in the server's own rule."""
    if not source.is_dir():
        raise PrepareError("source checkpoint directory missing: %s" % source)
    files = [source / name for name in REQUIRED]
    for processor in PROCESSORS:
        config = source / (processor + ".json")
        if config.is_symlink() or not config.is_file():
            raise PrepareError("missing processor config: %s" % config.name)
        manifest = json.loads(config.read_text(encoding="utf-8"))
        steps = manifest.get("steps") if isinstance(manifest, dict) else None
        if not isinstance(steps, list):
            raise PrepareError("invalid processor config: %s" % config.name)
        weights = []
        for step in steps:
            if not isinstance(step, dict):
                raise PrepareError("invalid processor step: %s" % config.name)
            filename = step.get("state_file")
            if filename is not None:
                if (not isinstance(filename, str) or Path(filename).name != filename
                        or not filename.startswith(processor + "_step_")
                        or not filename.endswith(".safetensors")):
                    raise PrepareError("invalid processor state_file in %s" % config.name)
                weights.append(source / filename)
        if not weights:
            raise PrepareError("missing checkpoint-owned processor weights: %s" % config.name)
        files.extend(weights)
    files = sorted(set(files))
    for file in files:
        if file.is_symlink() or not file.is_file():
            raise PrepareError("missing or symlinked checkpoint artifact: %s" % file)
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def adapter_base_path(adapter_config: Path, source: Path) -> str:
    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path") if isinstance(config, dict) else None
    if not isinstance(base, str) or not base:
        raise PrepareError("adapter config has no base_model_name_or_path: %s" % adapter_config.name)
    if not os.path.isabs(base):
        base = str((source / base).resolve())
    return base


def base_model_check(base_dir: str, expected_sha256: str | None = None) -> dict:
    """Required-file (and optional weight-hash) check for the external base policy."""
    # Absolute always: the served adapter config must not depend on the model
    # server's working directory.
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        raise PrepareError(
            "base policy directory missing: %s\n"
            "       pass --base-model-dir PATH to point the copied adapter config at the\n"
            "       exported base policy (it is referenced read-only, never copied)" % base)
    config = base / BASE_CONFIG
    if not config.is_file():
        raise PrepareError("base policy is missing %s in %s" % (BASE_CONFIG, base))
    weights = [base / name for name in BASE_WEIGHTS if (base / name).is_file()]
    if not weights:
        raise PrepareError("base policy has no weight file (%s) in %s"
                           % " or ".join(BASE_WEIGHTS), base)
    report = {
        "base_model_dir": str(base),
        "base_config": {"name": config.name, "size": config.stat().st_size},
        "base_weights": [{"name": weight.name, "size": weight.stat().st_size} for weight in weights],
        "base_model_weight_sha256": None,
    }
    weight = weights[0]
    if expected_sha256:
        actual = sha256_file(weight)
        if actual != expected_sha256:
            raise PrepareError("base policy weight hash mismatch for %s:\n"
                               "       expected %s\n       actual   %s" % (weight, expected_sha256, actual))
        report["base_model_weight_sha256"] = actual
        report["base_model_hash_verified"] = True
    else:
        report["base_model_hash_verified"] = False
        report["base_model_weight"] = str(weight)
    return report


def rewrite_base_path(destination: Path, base_dir: str) -> None:
    """Point the *copied* adapter config at the given base policy path."""
    config_path = destination / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_dir
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def load_identity(repo_root: Path):
    """Import the server's own checkpoint identity check (requires numpy/pyzmq)."""
    inference = repo_root / "third_party/flux/flux-inference"
    for path in (repo_root / "third_party/flux/flux-training/src",
                 inference / "ros2/flux_dex3", inference):
        sys.path.insert(0, str(path))
    try:
        from examples.dex3.zmq_server import checkpoint_identity
    except ImportError as exc:
        raise PrepareError(
            "cannot import the server's checkpoint check (%s); run this script with "
            "the model environment interpreter, for example FLUX_MODEL_PYTHON" % exc
        ) from exc
    return checkpoint_identity


def write_provenance(destination: Path, source: Path, files: list[Path], base: dict,
                     identity: str) -> Path:
    """Record what this copy was made from; written beside, never inside, the copy."""
    sidecar = destination.with_name(destination.name + ".provenance.json")
    record = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "scripts/flux-prepare-checkpoint.py",
        "source_checkpoint": str(source),
        "source_files": [{"name": file.name, "size": file.stat().st_size} for file in files],
        "base_model": base,
        "lerobot_commit_pin": LEROBOT_COMMIT,
        "checkpoint_identity": identity,
    }
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def remove_tree(path: Path) -> None:
    """Best-effort removal of a tree this script created (restores write mode first)."""
    if not path.exists():
        return
    os.chmod(path, 0o755)
    for item in path.iterdir():
        if item.is_file() and not item.is_symlink():
            os.chmod(item, 0o644)
    shutil.rmtree(path, ignore_errors=True)


def prepare(source: Path, destination: Path, identity, base_dir: str | None,
            expected_base_sha256: str | None) -> tuple[str, dict, Path]:
    files = checkpoint_files(source)
    source_base = adapter_base_path(source / "adapter_config.json", source)
    base = base_model_check(base_dir or source_base, expected_base_sha256)
    rewritten = base_dir is not None and os.path.abspath(base_dir) != os.path.abspath(source_base)
    total = sum(file.stat().st_size for file in files)
    print("copying %d checkpoint-owned files (%.1f MiB) -> %s"
          % (len(files), total / (1 << 20), destination))
    if base_dir is not None:
        print("base policy: %s%s" % (base["base_model_dir"],
                                     " (adapter config rewritten in the copy)"
                                     if rewritten else " (already the source's path)"))
    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o755)
    try:
        for file in files:
            shutil.copyfile(file, destination / file.name)
        # Rewrite before the read-only pass: the copy is the only place the
        # base path may change, and it must happen before it is sealed.
        if base_dir is not None and rewritten:
            rewrite_base_path(destination, base["base_model_dir"])
        for file in destination.iterdir():
            if file.is_file():
                os.chmod(file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(destination, 0o555)
        digest = identity(str(destination))
    except BaseException:
        # A failed preparation must not leave a half-sealed copy behind that a
        # later run would mistake for a prepared checkpoint.
        remove_tree(destination)
        raise
    sidecar = write_provenance(destination, source, files, base, digest)
    return digest, base, sidecar


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=os.environ.get("FLUX_CHECKPOINT_SOURCE"),
                        help="writable training checkpoint (env FLUX_CHECKPOINT_SOURCE)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="immutable copy location (default: DATA_ROOT/models/flux-dex3/<basename>)")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-model-dir", default=os.environ.get("FLUX_BASE_MODEL_DIR"),
                        help="external base policy export the copied adapter config points at "
                             "(env FLUX_BASE_MODEL_DIR); never copied")
    parser.add_argument("--base-model-sha256", default=os.environ.get("FLUX_BASE_MODEL_SHA256"),
                        help="expected sha256 of the base policy weight file (env "
                             "FLUX_BASE_MODEL_SHA256); verifies it, reading the whole file")
    parser.add_argument("--force", action="store_true", help="replace an existing copy")
    parser.add_argument("--check-only", action="store_true", help="verify an existing copy and exit")
    args = parser.parse_args(argv)

    if args.source is None and not args.check_only:
        print("error: --source (or FLUX_CHECKPOINT_SOURCE) is required", file=sys.stderr)
        return 2
    if args.check_only and args.dest is None and args.source is None:
        print("error: --check-only needs --dest or --source", file=sys.stderr)
        return 2
    data_root = Path(os.environ.get("HUMANOID_DATA_ROOT", args.repo_root / "data"))
    destination = args.dest or data_root / "models" / "flux-dex3" / args.source.name

    try:
        identity = load_identity(args.repo_root)
        if args.check_only:
            digest = identity(str(destination))
            # The served copy's own adapter config decides where the base policy
            # lives; verify it is reachable (and optionally hash-pinned).
            base = base_model_check(adapter_base_path(destination / "adapter_config.json", destination),
                                    args.base_model_sha256)
            print("checkpoint_identity=%s" % digest)
            print("base_model_dir=%s" % base["base_model_dir"])
            if base["base_model_hash_verified"]:
                print("base_model_weight_sha256=%s" % base["base_model_weight_sha256"])
            else:
                print("base_model_weight_sha256=(not computed; pass --base-model-sha256 to verify)")
            return 0
        if destination.exists():
            if not args.force:
                digest = identity(str(destination))
                # The copy already exists: verify its own base policy reference
                # so a moved export is caught here instead of at model load.
                base = base_model_check(
                    adapter_base_path(destination / "adapter_config.json", destination),
                    args.base_model_sha256)
                print("already prepared: %s" % digest)
                print("checkpoint_identity=%s" % digest)
                print("base_model_dir=%s" % base["base_model_dir"])
                return 0
            # Remove the old immutable tree: the directory itself is read-only
            # only by mode, so restore owner write permission first.
            remove_tree(destination)
        digest, base, sidecar = prepare(args.source.resolve(), destination, identity,
                                        args.base_model_dir, args.base_model_sha256)
    except PrepareError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except ValueError as exc:
        print("error: checkpoint identity rejected the copy: %s" % exc, file=sys.stderr)
        return 1
    print("checkpoint_identity=%s" % digest)
    print("destination=%s (read-only, no symlinks)" % destination)
    print("base_model_dir=%s" % base["base_model_dir"])
    if base["base_model_hash_verified"]:
        print("base_model_weight_sha256=%s" % base["base_model_weight_sha256"])
    else:
        print("base_model_weight_sha256=(not computed; pass --base-model-sha256 to verify)")
    print("provenance=%s" % sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
