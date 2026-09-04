#!/usr/bin/env python3
"""ASCII STL meshleri binary STL'e cevirip G1 asset kopyasi uretir (kaynak agac degismez)."""
from __future__ import annotations

import argparse
import re
import shutil
import struct
from pathlib import Path

VERTEX = re.compile(r"^\s*vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
NORMAL = re.compile(r"^\s*facet\s+normal\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def is_ascii_stl(path: Path) -> bool:
    # binary STL de "solid " baslayabilir -> tam UTF-8 decode decisive
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return text.lstrip().lower().startswith("solid")


def convert_ascii_stl(path: Path) -> int:
    facets: list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]] = []
    normal = (0.0, 0.0, 0.0)
    vertices: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        match = NORMAL.match(line)
        if match:
            normal = tuple(float(x) for x in match.groups())
            vertices = []
            continue
        match = VERTEX.match(line)
        if match:
            vertices.append(tuple(float(x) for x in match.groups()))
            if len(vertices) == 3:
                facets.append((normal, vertices))
                vertices = []
    if not facets:
        raise ValueError(f"no STL facets parsed: {path}")
    with path.open("wb") as f:
        f.write(b"SONIC G1 binary STL conversion".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(facets)))
        for face_normal, face_vertices in facets:
            f.write(struct.pack("<3f", *face_normal))
            for vertex in face_vertices:
                f.write(struct.pack("<3f", *vertex))
            f.write(struct.pack("<H", 0))
    return len(facets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "meshes").is_dir():
        raise SystemExit(f"G1 asset tree with meshes/ expected: {source}")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    files = facets = 0
    for mesh in output.rglob("*.STL"):
        if is_ascii_stl(mesh):
            facets += convert_ascii_stl(mesh)
            files += 1
    print(f"prepared={output} ascii_stl_converted={files} facets={facets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
