"""Deterministic episode-level stratified split (plan Gate 0).

The split is decided before any episode is converted, is derived only from the
frozen contract, and is fully reproducible: the seed and the per-collection
episode lists are written into ``split_manifest.json``. Episodes are never
shared between the train and validation splits, every collection contributes at
least one validation episode, and the two frozen exclusions appear in no split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .contract import ConversionConfig

SCHEMA_VERSION = 1


def _val_count(usable: int, val_fraction: float, minimum: int) -> int:
    """Per-collection validation size, at least ``minimum`` and below ``usable``."""
    if usable <= minimum:
        return max(1, usable - 1)
    count = round(val_fraction * usable)
    return min(max(count, minimum), usable - 1)


def build_split(config: ConversionConfig) -> dict[str, Any]:
    """Return the split manifest as a JSON-serializable mapping."""
    config.assert_contract()
    collections: dict[str, Any] = {}
    train_total = val_total = 0
    for collection in config.collections:
        usable = config.usable_episodes(collection.name)
        count = _val_count(len(usable), config.val_fraction, config.min_val_per_collection)
        # A per-collection derived seed keeps one collection's draw independent of
        # another collection's size, so adding a collection cannot reshuffle the rest.
        rng = random.Random(f"{config.split_seed}:{collection.name}")
        val = sorted(rng.sample(list(usable), count))
        train = [index for index in usable if index not in set(val)]
        if len(val) < config.min_val_per_collection:
            raise ValueError(f"{collection.name}: no validation episode could be selected")
        if set(train) & set(val):
            raise ValueError(f"{collection.name}: train and validation overlap")
        if sorted(train + val) != list(usable):
            raise ValueError(f"{collection.name}: the split does not cover the usable episodes")
        train_total += len(train)
        val_total += len(val)
        collections[collection.name] = {
            "total": config.source_total_episodes(collection.name),
            "usable": len(usable),
            "excluded": list(collection.excluded),
            "exclusion_reason": collection.exclusion_reason,
            "instruction": collection.instruction,
            "train": train,
            "val": val,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "config": {"path": str(config.path), "sha256": config.sha256},
        "sources": {"raw_root": str(config.raw_root), "sonic_root": str(config.sonic_root)},
        "strategy": "episode_level_stratified_by_collection",
        "seed": config.split_seed,
        "val_fraction": config.val_fraction,
        "min_val_per_collection": config.min_val_per_collection,
        "collections": collections,
        "totals": {
            "usable": train_total + val_total,
            "train": train_total,
            "val": val_total,
            "val_fraction_realized": val_total / (train_total + val_total),
        },
        # Validation normalizes with the train split's statistics (plan section 9.2).
        "normalization_stats": f"{config.train_repo}/meta/stats_psi0.json",
        "task_instructions": dict(config.tasks),
    }


def write_split_manifest(config: ConversionConfig, path: Path | None = None) -> Path:
    """Write ``split_manifest.json`` and return its path."""
    manifest = build_split(config)
    target = Path(path) if path is not None else config.output_root / "split_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_split_manifest(path: Path | str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported split manifest schema in {path}")
    return manifest


def assert_manifest_matches_config(manifest: dict[str, Any], config: ConversionConfig) -> None:
    """Refuse to convert against a split that another contract produced."""
    if manifest["config"]["sha256"] != config.sha256:
        raise ValueError(
            f"split manifest was built from config {manifest['config']['sha256']}, "
            f"current contract is {config.sha256}; rebuild the split"
        )
    if manifest["seed"] != config.split_seed:
        raise ValueError("split manifest seed disagrees with the contract")
    if sorted(manifest["collections"]) != sorted(config.collection_names):
        raise ValueError("split manifest collections disagree with the contract")
