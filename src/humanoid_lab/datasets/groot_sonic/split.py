"""Episode selection: the existing Psi0 split manifest, read as-is.

The GR00T pack does not re-decide the split. It converts exactly the episodes
the Psi0 pack was cut into -- same collections, same two excluded episodes, same
train/val identity -- so a model trained on either pack sees the same
demonstrations and the same held-out episodes. The manifest is only read; the
pipeline never writes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .contract import ROOT, ConversionConfig

SCHEMA_VERSION = 1
PSI0_CONFIG_PATH = ROOT / "configs/datasets/psi0/unitree_dex3_sonic_v1.yaml"
SPLITS = ("train", "val")


class SplitError(ValueError):
    """The manifest is not the frozen Psi0 split of this source corpus."""


def load_split_manifest(path: Path | str) -> dict[str, Any]:
    """Read the Psi0 ``split_manifest.json`` (the pack's frozen episode split)."""
    path = Path(path)
    if not path.is_file():
        raise SplitError(f"the Psi0 split manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SplitError(f"unsupported split manifest schema in {path}")
    return manifest


def assert_manifest_matches_config(manifest: dict[str, Any], config: ConversionConfig) -> None:
    """Fail closed unless the manifest is the Psi0 split of this very corpus.

    The manifest's own config hash belongs to the Psi0 contract, so it cannot be
    compared with this one; identity is checked through the fields that do
    describe the data: the dataset name, the collection set, the per-collection
    exclusions, the instructions and the coverage of every usable episode.
    """
    name = str(manifest.get("name", ""))
    if name != config.split_name:
        raise SplitError(f"the split manifest belongs to {name!r}, expected {config.split_name!r}")
    collections = manifest.get("collections") or {}
    if sorted(collections) != sorted(config.collection_names):
        raise SplitError("the split manifest collections disagree with the contract")
    for collection in config.collections:
        entry = collections[collection.name]
        excluded = tuple(sorted(int(index) for index in entry.get("excluded", ())))
        if excluded != tuple(sorted(collection.excluded)):
            raise SplitError(
                f"{collection.name}: the manifest excludes {excluded} but the contract declares "
                f"{tuple(sorted(collection.excluded))}"
            )
        if str(entry.get("instruction", "")) != collection.instruction:
            raise SplitError(f"{collection.name}: the manifest instruction disagrees with the contract")
        usable = sorted(int(index) for index in list(entry["train"]) + list(entry["val"]))
        if len(set(usable)) != len(usable):
            raise SplitError(f"{collection.name}: the split lists an episode twice")
        if set(usable) & set(excluded):
            raise SplitError(f"{collection.name}: an excluded episode appears in the split")
        if not entry["val"]:
            raise SplitError(f"{collection.name}: the validation split is empty")
    if str(manifest.get("strategy", "")) != "episode_level_stratified_by_collection":
        raise SplitError(f"unexpected split strategy {manifest.get('strategy')!r}")


def assert_instructions_match_psi0(config: ConversionConfig) -> None:
    """The instruction text is shared with the Psi0 pack, so it cannot drift."""
    if not PSI0_CONFIG_PATH.is_file():
        raise SplitError(f"the Psi0 contract is missing: {PSI0_CONFIG_PATH}")
    psi0 = yaml.safe_load(PSI0_CONFIG_PATH.read_text(encoding="utf-8"))
    tasks = {str(key): str(value) for key, value in (psi0.get("tasks") or {}).items()}
    if tasks != config.tasks:
        drifted = sorted(
            key for key in set(tasks) | set(config.tasks) if tasks.get(key) != config.tasks.get(key)
        )
        raise SplitError(f"the task instructions drifted from the Psi0 contract: {drifted}")


def split_episodes(manifest: dict[str, Any], config: ConversionConfig) -> dict[str, dict[str, list[int]]]:
    """``{split: {collection: [source episode, ...]}}``, in manifest order."""
    result: dict[str, dict[str, list[int]]] = {}
    for split in SPLITS:
        result[split] = {
            collection.name: [int(index) for index in manifest["collections"][collection.name][split]]
            for collection in config.collections
        }
    return result


def selection(
    manifest: dict[str, Any],
    config: ConversionConfig,
    *,
    train_per_task: int | None = None,
    val_per_task: int | None = None,
    all_episodes: bool = False,
) -> dict[str, dict[str, list[int]]]:
    """Mini mode (first N episodes per task per split) or the whole manifest."""
    if all_episodes and (train_per_task is not None or val_per_task is not None):
        raise SplitError("--all-episodes cannot be combined with a per-task limit")
    episodes = split_episodes(manifest, config)
    if all_episodes:
        return episodes
    limits = {split: train_per_task if split == "train" else val_per_task for split in SPLITS}
    chosen: dict[str, dict[str, list[int]]] = {}
    for split in SPLITS:
        limit = limits[split]
        chosen[split] = {
            name: ([] if limit is None else list(values[:limit]))
            for name, values in episodes[split].items()
        }
    return chosen


def selection_summary(chosen: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, int]]:
    return {split: {name: len(values) for name, values in by_task.items()} for split, by_task in chosen.items()}


__all__ = [
    "PSI0_CONFIG_PATH",
    "SPLITS",
    "SplitError",
    "assert_instructions_match_psi0",
    "assert_manifest_matches_config",
    "load_split_manifest",
    "selection",
    "selection_summary",
    "split_episodes",
]
