"""Materialize the fine-tune's training-start (base) artifact as a servable run dir.

The evidence chain this module follows instead of guessing a path:

* every run in the fine-tune lineage records the warm start it trained from in
  ``run_config.json`` (``model.model_name_or_path``, mirrored by
  ``model.pretrained_action_header_path``); that directory is the pinned Psi0
  SONIC warm-start download (it carries ``MODEL_PROVENANCE.json`` with the
  upstream repo/revision);
* the deploy loader (``Psi0Model.from_pretrained``) reads only
  ``run_dir/checkpoints/ckpt_<step>/model.safetensors`` and expects
  ``vlm_model.*`` + ``action_header.*`` keys, while the warm start ships a
  HuggingFace-style ``model.safetensors`` (``model.*``) plus a separate
  ``action_header.safetensors``.

The materializer therefore writes a base run directory that reuses the fine-tune
run's request contract (``argv.txt``/``run_config.json``/pooled text cache) and
merges the warm start's two files into the loader's key layout under
``checkpoints/ckpt_<step>``.  The one config delta is ``model.state_null_token``:
the warm start predates that parameter, so the base entry serves it disabled.
The parameter is only substituted into the state token during training-time
state dropout (the ``eval()`` path never reads it), so the served behavior is
exactly the warm-start weights; keeping the flag on would require fabricating a
tensor the released artifact does not contain.

Nothing here downloads or invents weights: a missing or unreadable warm start
raises :class:`BaseArtifactError`, which callers surface as "base unavailable"
rather than substituting another checkpoint.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

VLM_PREFIX = "vlm_model."
ACTION_HEADER_PREFIX = "action_header."
ACTION_HEADER_FILE = "action_header.safetensors"
MODEL_FILE = "model.safetensors"
PROVENANCE_FILE = "BASE_ARTIFACT.json"
POOLED_CACHE_FILE = "clip_pooled_cache.pt"
#: The base is the weights before any fine-tune step; the checkpoint directory is
#: named for that step so ``/info.ckpt_step`` stays an honest integer.
BASE_STEP = 0

_NULL_TOKEN_FLAGS = ("--model.state-null-token", "--model.state_null_token")


class BaseArtifactError(RuntimeError):
    """The base artifact cannot be derived or materialized; base stays unavailable."""


@dataclass(frozen=True)
class BaseSource:
    """The training-start checkpoint named by the fine-tune run's own config."""

    checkpoint_dir: Path
    model_file: Path
    action_header_file: Path
    provenance_file: Optional[Path]
    model_name_or_path: str
    provenance: Optional[dict[str, Any]] = None


def read_base_source(fine_run_dir: Path) -> BaseSource:
    """Read the warm start the fine-tune trained from; never guess a location."""
    config_path = fine_run_dir / "run_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaseArtifactError(f"cannot read the fine-tune run_config {config_path}: {exc}") from exc
    model = config.get("model") if isinstance(config, dict) else None
    value = model.get("model_name_or_path") if isinstance(model, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise BaseArtifactError(
            f"{config_path} records no model.model_name_or_path; the training-start "
            "checkpoint cannot be identified"
        )
    expected = value.strip()
    header_value = model.get("pretrained_action_header_path")
    if isinstance(header_value, str) and header_value.strip() and header_value.strip() != expected:
        raise BaseArtifactError(
            f"run_config names model_name_or_path {expected!r} but "
            f"pretrained_action_header_path {header_value!r}; the warm start is ambiguous"
        )
    checkpoint_dir = Path(expected)
    if not checkpoint_dir.is_dir():
        raise BaseArtifactError(
            f"the training-start checkpoint named by run_config does not exist: {checkpoint_dir}"
        )
    model_file = checkpoint_dir / MODEL_FILE
    header_file = checkpoint_dir / ACTION_HEADER_FILE
    for path in (model_file, header_file):
        if not path.is_file():
            raise BaseArtifactError(f"the training-start checkpoint is missing {path.name}: {path}")
    provenance_file = checkpoint_dir / "MODEL_PROVENANCE.json"
    provenance = None
    if provenance_file.is_file():
        try:
            provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = None
    return BaseSource(
        checkpoint_dir=checkpoint_dir,
        model_file=model_file,
        action_header_file=header_file,
        provenance_file=provenance_file if provenance_file.is_file() else None,
        model_name_or_path=expected,
        provenance=provenance if isinstance(provenance, dict) else None,
    )


def default_base_run_dir(fine_run_dir: Path, source: BaseSource) -> Path:
    """``<experiment root>/base/<warm-start name>`` next to the fine-tune runs."""
    return fine_run_dir.parent.parent / "base" / source.checkpoint_dir.name


def _canonical(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:  # pragma: no cover - only on exotic filesystems
        return Path(path)


def _patch_argv(text: str) -> str:
    lines = [
        line for line in text.splitlines()
        if not any(line.strip().startswith(flag) for flag in _NULL_TOKEN_FLAGS)
    ]
    return "\n".join(lines) + "\n"


def _patched_run_config(config: dict[str, Any]) -> dict[str, Any]:
    # base predates the learned null token: its action header has no ``state_null``
    # parameter, so the served config must not ask for one (strict load).
    model = config.get("model")
    if isinstance(model, dict) and "state_null_token" in model:
        model["state_null_token"] = False
    return config


def _merge_state_dict(source: BaseSource, out_file: Path) -> tuple[int, int]:
    """Write ``vlm_model.*`` + ``action_header.*`` keys into one safetensors file."""
    from safetensors.torch import load_file, save_file

    merged: dict[str, Any] = {}
    for key, tensor in load_file(str(source.model_file)).items():
        if key.startswith(ACTION_HEADER_PREFIX):
            raise BaseArtifactError(f"{source.model_file} unexpectedly carries {key!r}")
        merged[VLM_PREFIX + key] = tensor
    vlm_keys = len(merged)
    for key, tensor in load_file(str(source.action_header_file)).items():
        if key.startswith(VLM_PREFIX):
            raise BaseArtifactError(f"{source.action_header_file} unexpectedly carries {key!r}")
        merged[ACTION_HEADER_PREFIX + key] = tensor
    header_keys = len(merged) - vlm_keys

    partial = out_file.with_name(out_file.name + ".partial")
    try:
        save_file(merged, str(partial))
        os.replace(partial, out_file)
    finally:
        if partial.exists():
            partial.unlink(missing_ok=True)
    return vlm_keys, header_keys


def _source_files_record(source: BaseSource) -> dict[str, Any]:
    record: dict[str, Any] = {}
    provenance_files = {}
    if source.provenance is not None:
        for entry in source.provenance.get("files", []) or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                provenance_files[entry["path"]] = entry
    for path in (source.model_file, source.action_header_file):
        entry = provenance_files.get(path.name, {})
        record[path.name] = {
            "size": path.stat().st_size,
            "sha256": entry.get("sha256"),
        }
    return record


def materialize_base_run_dir(
    fine_run_dir: Path,
    out_dir: Path,
    *,
    source: Optional[BaseSource] = None,
    step: int = BASE_STEP,
    log=lambda message: None,
) -> dict[str, Any]:
    """Create/refresh the servable base run dir and return its provenance record."""
    source = source or read_base_source(fine_run_dir)
    fine_run_dir = _canonical(fine_run_dir)
    out_dir = Path(out_dir)
    if _canonical(out_dir) in (_canonical(fine_run_dir), _canonical(source.checkpoint_dir)):
        raise BaseArtifactError(
            f"refusing to materialize over an input directory: {out_dir}"
        )

    checkpoint_dir = out_dir / "checkpoints" / f"ckpt_{step}"
    merged_file = checkpoint_dir / MODEL_FILE
    provenance_path = out_dir / PROVENANCE_FILE

    expected_source = {
        "model_name_or_path": source.model_name_or_path,
        "model_file": str(source.model_file),
        "action_header_file": str(source.action_header_file),
    }
    if merged_file.is_file() and provenance_path.is_file():
        try:
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if all(existing.get(key) == value for key, value in expected_source.items()):
            log(f"[base] reusing materialized base run dir: {out_dir}")
            return existing

    log(f"[base] materializing base run dir {out_dir} <- {source.checkpoint_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((fine_run_dir / "run_config.json").read_text(encoding="utf-8"))
    patched = _patched_run_config(copy.deepcopy(config))
    (out_dir / "run_config.json").write_text(json.dumps(patched, indent=2) + "\n", encoding="utf-8")
    argv_text = (fine_run_dir / "argv.txt").read_text(encoding="utf-8")
    (out_dir / "argv.txt").write_text(_patch_argv(argv_text), encoding="utf-8")
    pooled_cache = fine_run_dir / POOLED_CACHE_FILE
    if pooled_cache.is_file():
        shutil.copy2(pooled_cache, out_dir / POOLED_CACHE_FILE)

    vlm_keys, header_keys = _merge_state_dict(source, merged_file)

    record = {
        "kind": "psi0-base-materialization",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fine_run_dir": str(fine_run_dir),
        "fine_run_config": str(fine_run_dir / "run_config.json"),
        "arg_txt": str(fine_run_dir / "argv.txt"),
        "model_name_or_path": expected_source["model_name_or_path"],
        "model_file": expected_source["model_file"],
        "action_header_file": expected_source["action_header_file"],
        "source_checkpoint_dir": str(source.checkpoint_dir),
        "source_files": _source_files_record(source),
        "source_provenance": {
            key: source.provenance.get(key)
            for key in ("repo", "revision", "variant")
        } if source.provenance is not None else None,
        "merged": {
            "path": str(merged_file),
            "step": step,
            "vlm_model_keys": vlm_keys,
            "action_header_keys": header_keys,
            "size": merged_file.stat().st_size,
        },
        "config_delta": [
            "model.state_null_token: false (the warm start has no action_header.state_null "
            "parameter; the token only participates in training-time state dropout)"
        ],
        "argv_patch": f"removed {_NULL_TOKEN_FLAGS[0]} (dash spelling; underscore accepted too)",
    }
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"[base] wrote {merged_file} ({record['merged']['size']} bytes, "
        f"{vlm_keys} vlm_model + {header_keys} action_header keys)")
    return record
