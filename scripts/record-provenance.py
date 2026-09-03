#!/usr/bin/env python3
"""Record MODEL_PROVENANCE.json for existing pinned model downloads.

Hashes every file (SHA-256) under <data-root>/models/<name> and writes
MODEL_PROVENANCE.json next to it, with repo/revision from versions.lock.yaml.
Idempotent; existing large weights are NOT re-downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lock = yaml.safe_load(open(os.path.join(ROOT, "versions.lock.yaml")))
data_root = os.environ.get("HUMANOID_DATA_ROOT", os.path.expanduser("~/humanoid-lab-data"))
models_root = os.path.join(data_root, "models")


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


failed = 0
for name, m in lock["models"].items():
    dest = os.path.join(models_root, name)
    if not os.path.isdir(dest):
        print(f"[skip] {name}: {dest} yok", flush=True)
        continue
    print(f"== {name} :: {m['repo']} @ {m['revision']}", flush=True)
    prov = {
        "repo": m["repo"],
        "revision": m["revision"],
        "variant": m.get("variant", ""),
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "hashed locally from pre-existing pinned download; regenerate by re-running scripts/fetch-models.sh",
        "files": [],
    }
    try:
        for dirpath, dirs, files in os.walk(dest):
            dirs[:] = [d for d in dirs if d != ".cache"]
            for fn in sorted(files):
                fp = os.path.join(dirpath, fn)
                if fn == "MODEL_PROVENANCE.json":
                    continue
                prov["files"].append(
                    {"path": os.path.relpath(fp, dest), "sha256": sha256_of(fp)}
                )
        prov["files"].sort(key=lambda x: x["path"])
        with open(os.path.join(dest, "MODEL_PROVENANCE.json"), "w") as f:
            json.dump(prov, f, indent=2)
            f.write("\n")
        print(f"   [ok] {len(prov['files'])} files hashed", flush=True)
    except OSError as exc:
        print(f"   [fail] {name}: {exc}", flush=True)
        failed += 1
print("PROVENANCE DONE" if not failed else f"PROVENANCE FAILED={failed}")
sys.exit(1 if failed else 0)
