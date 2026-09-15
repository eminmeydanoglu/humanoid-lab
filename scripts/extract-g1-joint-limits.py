#!/usr/bin/env python3
"""Extract G1 joint position limits from the pinned SONIC MuJoCo model.

Range checks in QC need a physical reference.  The pinned deployment already
ships one (``gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml``),
so the limits are read from that file instead of being typed in by hand, and the
source checksum travels with them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DEFAULT_MODEL = Path(
    "/opt/src/sonic/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
)
JOINT_PATTERN = re.compile(r"<joint\s+([^>]*?)/?>", re.DOTALL)
NAME_PATTERN = re.compile(r'name="([^"]+)"')
RANGE_PATTERN = re.compile(r'range="([^"]+)"')


def extract(model: Path) -> dict[str, object]:
    text = model.read_text(encoding="utf-8")
    joints: dict[str, dict[str, float | str]] = {}
    for attributes in JOINT_PATTERN.findall(text):
        name = NAME_PATTERN.search(attributes)
        span = RANGE_PATTERN.search(attributes)
        if name is None or span is None:
            continue
        lower, upper = (float(value) for value in span.group(1).split())
        joints[name.group(1)] = {"lower": lower, "upper": upper, "units": "rad"}
    if len(joints) < 43:
        raise SystemExit(f"only {len(joints)} limited joints parsed from {model}")
    return {
        "schema_version": 1,
        "source": str(model),
        "source_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "extraction": "regex over <joint name=... range=...> tags",
        "units": "rad",
        "joints": joints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    document = extract(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(document['joints'])} joints from {document['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
