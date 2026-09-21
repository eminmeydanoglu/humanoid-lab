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

An existing materialization is never trusted on its recorded paths alone.  Reuse
re-verifies the warm start's size+sha256, the merged file's size+sha256,
safetensors key counts/prefixes and truncation, the served ``run_config``/
``argv.txt`` patches and the required files; a mismatch, corruption or
half-written artifact is rebuilt into a fresh temp directory and swapped into
place, so the served directory never holds a partial state.  A record that
predates a fingerprint (or whose warm start's manifest never described one) is
hashed and the fingerprint is recorded, instead of rewriting a sound artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
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
#: Verification reads each artifact once per launch; the model load dominates the
#: startup either way, so one big read per file keeps the syscall overhead out.
_HASH_CHUNK = 8 * 1024 * 1024
#: Guards the header reader against a garbage length field (real headers for this
#: model are a few hundred KB).
_MAX_HEADER_BYTES = 64 * 1024 * 1024


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_index(path: Path) -> tuple[dict[str, int], int]:
    """Key -> payload bytes from the header alone (no tensor data), plus the file
    length that header declares (``8 + header + data``).

    Reading the header is how a truncated or half-written file is caught before
    anything loads its tensors: a missing tail shows up as a declared length that
    is larger than the file on disk.
    """
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise BaseArtifactError(f"{path} is not a safetensors file: it has no header")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= _MAX_HEADER_BYTES:
                raise BaseArtifactError(
                    f"{path} declares a nonsense safetensors header length {header_length}"
                )
            raw_header = handle.read(header_length)
    except OSError as exc:
        raise BaseArtifactError(f"cannot read {path}: {exc}") from exc
    if len(raw_header) != header_length:
        raise BaseArtifactError(
            f"{path} is truncated: the header stops at {len(raw_header)} of "
            f"{header_length} bytes"
        )
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise BaseArtifactError(f"{path} has an unreadable safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise BaseArtifactError(f"{path} has a safetensors header that is not an object")

    lengths: dict[str, int] = {}
    data_end = 0
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)):
            raise BaseArtifactError(f"{path} header entry {key!r} has no usable data_offsets")
        start, end = offsets
        if end < start:
            raise BaseArtifactError(f"{path} header entry {key!r} has reversed data_offsets")
        lengths[key] = end - start
        data_end = max(data_end, end)
    return lengths, 8 + header_length + data_end


def _source_files_record(source: BaseSource) -> dict[str, Any]:
    """Fingerprint the warm start as it is on disk now (size + sha256)."""
    return {
        path.name: {"size": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in (source.model_file, source.action_header_file)
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Small-file atomic write: a torn provenance must not force a full rebuild."""
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink(missing_ok=True)


def _temp_dir(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.tmp-{os.getpid()}"


def _replaced_dir(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.replaced"


def _cleanup_stale_temps(out_dir: Path) -> None:
    """Drop the temp dirs of a materialization that died mid-build."""
    parent = out_dir.parent
    if not parent.is_dir():
        return
    prefix = f".{out_dir.name}.tmp-"
    for entry in parent.iterdir():
        if entry.name.startswith(prefix):
            shutil.rmtree(entry, ignore_errors=True)


def _replace_dir(tmp_dir: Path, out_dir: Path) -> None:
    """Swap a fully built directory into place; the target never holds a partial dir.

    A crash between the two renames leaves the previous copy as ``.replaced``
    (the next successful build reclaims it) and no ``out_dir``; nothing ever
    observes a half-written directory under the served path.
    """
    replaced = _replaced_dir(out_dir)
    if replaced.exists():
        shutil.rmtree(replaced, ignore_errors=True)
    if out_dir.exists():
        os.replace(out_dir, replaced)
    try:
        os.replace(tmp_dir, out_dir)
    except OSError:
        if replaced.exists() and not out_dir.exists():
            os.replace(replaced, out_dir)
        raise
    if replaced.exists():
        shutil.rmtree(replaced, ignore_errors=True)


def _verify_reuse(
    existing: dict[str, Any],
    *,
    fine_run_dir: Path,
    source: BaseSource,
    out_dir: Path,
    step: int,
) -> tuple[list[str], dict[str, Any]]:
    """Re-verify a materialized base run dir against the fine run it came from.

    Returns the problems that forbid reuse (empty means verified) and the
    fingerprints that a record predating them was missing, to be recorded
    instead of rewriting an artifact whose only gap is the missing fingerprint.
    """
    problems: list[str] = []
    refreshed: dict[str, Any] = {}

    def same_path(recorded: Any, expected: Path) -> bool:
        return isinstance(recorded, str) and _canonical(recorded) == _canonical(expected)

    if existing.get("kind") != "psi0-base-materialization":
        return ["the provenance is not a base materialization record"], {}
    if not same_path(existing.get("fine_run_dir"), fine_run_dir):
        problems.append("the record belongs to another fine-tune run")
    for key, expected in (("model_name_or_path", source.model_name_or_path),
                          ("model_file", str(source.model_file)),
                          ("action_header_file", str(source.action_header_file))):
        if existing.get(key) != expected:
            problems.append(f"the recorded {key} differs from run_config's warm start")

    merged_record = existing.get("merged") if isinstance(existing.get("merged"), dict) else {}
    checkpoint_dir = out_dir / "checkpoints" / f"ckpt_{step}"
    merged_file = checkpoint_dir / MODEL_FILE
    if not same_path(merged_record.get("path"), merged_file):
        problems.append("the recorded merged path differs")
    if merged_record.get("step") != step:
        problems.append("the recorded merged step differs")

    for path in (out_dir / "run_config.json", out_dir / "argv.txt", merged_file):
        if not path.is_file():
            problems.append(f"the required file {path} is missing")
    source_record = existing.get("source_files")
    for path in (source.model_file, source.action_header_file):
        entry = source_record.get(path.name) if isinstance(source_record, dict) else None
        if not isinstance(entry, dict):
            problems.append(f"the record has no fingerprint for {path.name}")
        elif entry.get("size") != path.stat().st_size:
            problems.append(f"the warm-start file {path.name} changed size")
    if problems:
        return problems, {}

    try:
        fine_config = json.loads((fine_run_dir / "run_config.json").read_text(encoding="utf-8"))
        fine_argv = (fine_run_dir / "argv.txt").read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot re-read the fine-tune run's config/argv: {exc}"], {}
    expected_config = _patched_run_config(copy.deepcopy(fine_config))
    try:
        served_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"the served run_config is unreadable: {exc}")
        served_config = None
    else:
        if served_config != expected_config:
            problems.append("the served run_config differs from the fine-tune run's (patched) config")
    fine_model = fine_config.get("model") if isinstance(fine_config, dict) else None
    if isinstance(fine_model, dict) and "state_null_token" in fine_model:
        served_model = served_config.get("model") if isinstance(served_config, dict) else None
        if not (isinstance(served_model, dict) and served_model.get("state_null_token") is False):
            problems.append("the served run_config does not disable model.state_null_token")

    try:
        served_argv = (out_dir / "argv.txt").read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(f"the served argv.txt is unreadable: {exc}")
    else:
        if any(line.strip().startswith(flag) for line in served_argv.splitlines()
               for flag in _NULL_TOKEN_FLAGS):
            problems.append("the served argv.txt still carries a null-token flag")
        if served_argv != _patch_argv(fine_argv):
            problems.append("the served argv.txt differs from the fine-tune run's argv")

    fine_cache = fine_run_dir / POOLED_CACHE_FILE
    served_cache = out_dir / POOLED_CACHE_FILE
    if fine_cache.is_file():
        if not served_cache.is_file():
            problems.append(f"the copy of {POOLED_CACHE_FILE} is missing")
        elif (fine_cache.stat().st_size != served_cache.stat().st_size
              or _sha256_file(fine_cache) != _sha256_file(served_cache)):
            problems.append(f"the copy of {POOLED_CACHE_FILE} differs from the fine-tune run's")
    elif served_cache.is_file():
        problems.append(f"a stale {POOLED_CACHE_FILE} is present although the fine-tune run has none")

    try:
        merged_lengths, declared_size = _safetensors_index(merged_file)
        vlm_lengths, _ = _safetensors_index(source.model_file)
        header_lengths, _ = _safetensors_index(source.action_header_file)
    except BaseArtifactError as exc:
        problems.append(str(exc))
    else:
        expected_lengths = {VLM_PREFIX + key: length for key, length in vlm_lengths.items()}
        expected_lengths.update(
            {ACTION_HEADER_PREFIX + key: length for key, length in header_lengths.items()}
        )
        if merged_lengths != expected_lengths:
            problems.append(
                "the merged safetensors keys/payload sizes do not match the warm start"
            )
        vlm_keys = sum(1 for key in merged_lengths if key.startswith(VLM_PREFIX))
        header_keys = sum(1 for key in merged_lengths if key.startswith(ACTION_HEADER_PREFIX))
        if len(merged_lengths) != vlm_keys + header_keys:
            problems.append("the merged safetensors carries keys outside the two namespaces")
        if merged_record.get("vlm_model_keys") != vlm_keys:
            problems.append("the merged vlm_model key count differs from the record")
        if merged_record.get("action_header_keys") != header_keys:
            problems.append("the merged action_header key count differs from the record")
        actual_size = merged_file.stat().st_size
        if declared_size != actual_size:
            problems.append(
                f"the merged safetensors is truncated: the header declares {declared_size} "
                f"bytes, the file holds {actual_size}"
            )
        if merged_record.get("size") != actual_size:
            problems.append("the merged file size differs from the record")

    refreshed_sources: dict[str, Any] = {}
    for path in (source.model_file, source.action_header_file):
        entry = source_record.get(path.name) if isinstance(source_record, dict) else {}
        try:
            computed = _sha256_file(path)
        except OSError as exc:
            problems.append(f"cannot hash {path}: {exc}")
            continue
        recorded = entry.get("sha256") if isinstance(entry, dict) else None
        if isinstance(recorded, str):
            if recorded != computed:
                problems.append(f"the warm-start file {path.name} changed since materialization")
        else:
            refreshed_sources[path.name] = computed
    try:
        merged_sha = _sha256_file(merged_file)
    except OSError as exc:
        problems.append(f"cannot hash {merged_file}: {exc}")
    else:
        recorded_sha = merged_record.get("sha256")
        if isinstance(recorded_sha, str):
            if recorded_sha != merged_sha:
                problems.append("the merged file changed since materialization")
        else:
            refreshed["merged_sha256"] = merged_sha
    if refreshed_sources:
        refreshed["source_sha256"] = refreshed_sources
    return problems, refreshed


def _record_fingerprints(record: dict[str, Any],
                         refreshed: dict[str, Any]) -> dict[str, Any]:
    """Merge computed fingerprints into a record whose sources lack them."""
    record = copy.deepcopy(record)
    for name, sha in (refreshed.get("source_sha256") or {}).items():
        entry = record.get("source_files", {}).get(name)
        if isinstance(entry, dict):
            entry["sha256"] = sha
    merged_sha = refreshed.get("merged_sha256")
    if isinstance(record.get("merged"), dict) and isinstance(merged_sha, str):
        record["merged"]["sha256"] = merged_sha
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
        if isinstance(existing, dict):
            try:
                problems, refreshed = _verify_reuse(
                    existing, fine_run_dir=fine_run_dir, source=source, out_dir=out_dir, step=step,
                )
            except OSError as exc:
                problems, refreshed = [f"cannot verify the existing materialization: {exc}"], {}
            if not problems:
                if not refreshed:
                    log(f"[base] reusing the verified base run dir: {out_dir}")
                    return existing
                record = _record_fingerprints(existing, refreshed)
                _write_json_atomic(provenance_path, record)
                log(f"[base] the base run dir is sound but its record predates the sha256 "
                    f"fingerprints; recorded them: {out_dir}")
                return record
            log("[base] the base run dir failed verification, rebuilding it atomically: "
                + "; ".join(problems))

    log(f"[base] materializing base run dir {out_dir} <- {source.checkpoint_dir}")

    _cleanup_stale_temps(out_dir)
    tmp_dir = _temp_dir(out_dir)
    try:
        # Build the whole run dir elsewhere and swap it in afterwards, so the
        # served path never holds a half-written artifact and a failed rebuild
        # leaves the previous one intact.
        tmp_checkpoint = tmp_dir / "checkpoints" / f"ckpt_{step}"
        tmp_checkpoint.mkdir(parents=True, exist_ok=True)
        tmp_merged = tmp_checkpoint / MODEL_FILE
        config = json.loads((fine_run_dir / "run_config.json").read_text(encoding="utf-8"))
        patched = _patched_run_config(copy.deepcopy(config))
        (tmp_dir / "run_config.json").write_text(json.dumps(patched, indent=2) + "\n",
                                                 encoding="utf-8")
        argv_text = (fine_run_dir / "argv.txt").read_text(encoding="utf-8")
        (tmp_dir / "argv.txt").write_text(_patch_argv(argv_text), encoding="utf-8")
        pooled_cache = fine_run_dir / POOLED_CACHE_FILE
        if pooled_cache.is_file():
            shutil.copy2(pooled_cache, tmp_dir / POOLED_CACHE_FILE)

        vlm_keys, header_keys = _merge_state_dict(source, tmp_merged)

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
                "size": tmp_merged.stat().st_size,
                "sha256": _sha256_file(tmp_merged),
            },
            "config_delta": [
                "model.state_null_token: false (the warm start has no action_header.state_null "
                "parameter; the token only participates in training-time state dropout)"
            ],
            "argv_patch": f"removed {_NULL_TOKEN_FLAGS[0]} (dash spelling; underscore accepted too)",
        }
        _write_json_atomic(tmp_dir / PROVENANCE_FILE, record)
        _replace_dir(tmp_dir, out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    log(f"[base] wrote {merged_file} ({record['merged']['size']} bytes, "
        f"{vlm_keys} vlm_model + {header_keys} action_header keys)")
    return record
