#!/usr/bin/env python3
"""doctor.sh icin timestamped JSON/YAML rapor."""
import argparse
import datetime
import json
import sys

try:
    import yaml
except ImportError:
    yaml = None


def _load(path: str):
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-yaml", required=True)
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--host-data", default="")      # doctor.sh: /tmp/hl-host.json
    ap.add_argument("--docker-data", default="")     # doctor.sh: /tmp/hl-docker.json
    ap.add_argument("--container-data", default="")  # doctor.sh: /tmp/hl-container.json
    ap.add_argument("--git-data", default="")
    ap.add_argument("--models-data", default="")
    ap.add_argument("--dds-data", default="")
    ap.add_argument("--disk-data", default="")
    args = ap.parse_args()

    host = _load(args.host_data)
    docker = _load(args.docker_data)
    container = _load(args.container_data)
    git = _load(args.git_data)
    models = _load(args.models_data)
    dds = _load(args.dds_data)
    disk = _load(args.disk_data)

    data = {
        "doctor_version": 1,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {
            "os": host.get("os", "unknown"),
            "kernel": host.get("kernel", ""),
            "hostname": host.get("hostname", ""),
            "gpu": host.get("gpu", ""),
            "driver": host.get("driver", ""),
            "driver_baseline_ok": host.get("driver_baseline_ok"),
        },
        "docker": docker,
        "container": container,
        "git": git,
        "models": models,
        "dds": dds,
        "disk": disk,
        "exit_rc": args.rc,
    }
    with open(args.out_json, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    if yaml is not None:
        with open(args.out_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
