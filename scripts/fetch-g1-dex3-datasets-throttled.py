#!/usr/bin/env python3
"""Resume the G1 Dex3 mirror with a bandwidth cap.

`hf download` has no rate limit, so this fetches only the files still missing
from the local mirror with `curl --limit-rate` (one transfer at a time, so the
cap applies to the whole job) and records the huggingface_hub cache sidecar for
every fetched file so a later `hf download` treats it as cached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_URL = "https://huggingface.co/api/datasets/{repo}"
TREE_URL = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
FILE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def curl_fetch(url: str, dest: Path, rate: str, expected: int, retries: int = 4) -> None:
    for _ in range(retries):
        if dest.exists() and dest.stat().st_size == expected:
            return
        cmd = [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "3", "--retry-delay", "5", "--limit-rate", rate,
            "--output", str(dest),
        ]
        if dest.exists() and 0 < dest.stat().st_size < expected:
            cmd += ["--continue-at", "-"]
        if subprocess.run(cmd + [url]).returncode == 0 and dest.exists() and dest.stat().st_size == expected:
            return
        time.sleep(5)
    raise RuntimeError(f"transfer failed after {retries} attempts: {url}")


def write_provenance(root: Path, sources: dict) -> None:
    provenance = {"collection": "mindchain/unitree-robotics-g1-dex3-datasets", "datasets": {}}
    for name, repo in sorted(sources.items()):
        target = root / name
        if not target.is_dir():
            continue
        try:
            revision = get_json(API_URL.format(repo=repo)).get("sha")
        except Exception:  # noqa: BLE001 - provenance is best effort
            revision = None
        total = files = 0
        for directory, _children, names in os.walk(target):
            if ".cache" in Path(directory).parts:
                continue
            for entry in names:
                total += (Path(directory) / entry).stat().st_size
                files += 1
        provenance["datasets"][name] = {"repo": repo, "revision": revision, "files": files, "bytes": total}
    provenance["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("wrote provenance.json", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="data/datasets/first_tur_ham/unitree-g1-dex3")
    parser.add_argument("--rate", default="6500k", help="curl --limit-rate value; default 6500k ~= 6.5 MB/s")
    parser.add_argument("datasets", nargs="*", help="only these dataset directories (default: all)")
    args = parser.parse_args()

    root = Path(args.datasets_root)
    sources = json.loads((root / "sources.json").read_text())
    selected = {name: repo for name, repo in sorted(sources.items()) if not args.datasets or name in args.datasets}
    if not selected:
        print("no datasets selected", file=sys.stderr)
        return 2

    plans = []
    total_missing = 0
    for name, repo in selected.items():
        try:
            revision = get_json(API_URL.format(repo=repo)).get("sha")
            files = [
                (entry["path"], entry.get("size", 0))
                for entry in get_json(TREE_URL.format(repo=repo))
                if entry["type"] == "file"
            ]
        except Exception as exc:  # noqa: BLE001 - report and abort listing
            print(f"FAIL {name}: cannot list files: {exc}", flush=True)
            return 2
        missing = []
        for rel_path, size in sorted(files):
            local = root / name / rel_path
            if not local.exists() or local.stat().st_size != size:
                missing.append((rel_path, size))
        total_missing += sum(size for _, size in missing)
        print(f"[{name}] {len(files) - len(missing)}/{len(files)} present, {len(missing)} missing ({sum(size for _, size in missing) / 1e9:.2f} GB)", flush=True)
        plans.append((name, repo, revision, missing))

    if total_missing == 0:
        print("nothing missing; all selected datasets are complete", flush=True)
        write_provenance(root, sources)
        return 0
    print(f"fetching {total_missing / 1e9:.2f} GB at {args.rate}", flush=True)

    fetched = failed = 0
    for name, repo, revision, missing in plans:
        for rel_path, size in missing:
            dest = root / name / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            try:
                curl_fetch(FILE_URL.format(repo=repo, path=rel_path), dest, args.rate, size)
            except RuntimeError as exc:
                print(f"FAIL {name}/{rel_path}: {exc}", flush=True)
                failed += 1
                continue
            got = dest.stat().st_size
            if got != size:
                print(f"FAIL {name}/{rel_path}: size {got} != expected {size}", flush=True)
                failed += 1
                continue
            sidecar = root / name / ".cache/huggingface/download" / f"{rel_path}.metadata"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(f"{revision}\n{sha256_file(dest)}\n{time.time()}\n")
            fetched += 1
            elapsed = time.time() - started
            print(f"ok   {name}/{rel_path} {size / 1e6:.0f} MB in {elapsed:.0f}s ({size / elapsed / 1e6 if elapsed else 0:.1f} MB/s)", flush=True)

    print(f"done: {fetched} files fetched, {failed} failed", flush=True)
    if failed:
        return 1
    write_provenance(root, sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
