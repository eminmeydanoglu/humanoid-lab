#!/usr/bin/env python3
"""Experiment 09 -- GR00T third-cube visual appearance A/B.

One question: the BlockStacking demonstrations stack red, yellow and *green*
blocks while the caption that trained the policy says "red, yellow, blue", and
the simulated scene's third block is *blue*.  If the simulated third cube is
rendered in the colour the demonstrations actually show, does GR00T's high
target palm and its approach to the cubes improve?

Cell ``A`` is the canonical scene.  Cell ``G`` runs the same command with the
opt-in scene profile that changes only the third cube's rendered material; the
cube's identity (``blue``), pose, size, mass, physics material, the other cubes,
the tape, the table, the camera and the lighting are the same, the instruction is
the exact training-metadata string in both cells, and the initial-pose handshake
of experiment 08 runs in both cells so it cannot be the treatment.

Stages:

    scripts/groot-cube-color-ab.py colour     # was the treatment applied, visually?
    scripts/groot-cube-color-ab.py analyse    # validity + paired metrics + decision
    scripts/groot-cube-color-ab.py figures    # at most three figures
    scripts/groot-cube-color-ab.py manifest   # sha256 manifest
    scripts/groot-cube-color-ab.py all

The per-rollout reading (tracking series, telemetry, support calibration, palm
kinematics) is experiment 07's own ``Rollout`` object and the settle/delivery
evidence is experiment 08's own helpers, imported rather than re-implemented.
The colour statistic is the one the dataset comparison and the variant profile's
provenance were both built on: the mean RGB of a colour-class blob inside one
frame, taken as a median across frames, in sRGB.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

SCHEMA_VERSION = 1
SCRIPT_VERSION = "groot-cube-color-ab.py/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
EXPERIMENTS = CAMPAIGN / "experiments"
DEFAULT_OUT = EXPERIMENTS / "09-groot-cube-color-ab"

#: ``A`` renders the shipped blue third cube; ``G`` the demonstration-matched green.
CELLS = ("A", "G")
CELL_TREATMENT = {
    "A": "canonical scene: the third cube renders the shipped blue",
    "G": "variant scene: the third cube renders the demonstration-matched green",
}
#: Which session supplies each cell's repeats: (session directory, rollout index).
#: Cell G's first repeat was rejected -- the robot lay on the floor for the whole
#: policy window after a fall in the model-switch settle -- so the replacement
#: session supplies repeat 1 and the original session supplies repeats 2 and 3.
#: The pairing is by slot: slot 1 in both cells is the first policy rollout after
#: the model switch, so the two cells are compared like for like.
CELL_REPEATS = {
    "A": (("A", 1), ("A", 2), ("A", 3)),
    "G": (("G-replacement", 1), ("G", 2), ("G", 3)),
}
#: The exact training-metadata instruction, identical in both cells by design.
PROMPT = "Stack the three cubic blocks on the black tape in the order red, yellow, blue."
#: Policy-output windows, in seconds after Start.
WINDOWS_S = {"first_1s": (0.0, 1.0), "first_5s": (0.0, 5.0), "active": (0.0, None)}
#: Pre-declared acceptance, fixed before the G campaign was launched.
ACCEPTANCE = {
    "pairs": 3,
    "paired_repeats_required": 2,
    "min_palm_z_drop_m": 0.05,
    "min_relative_cube_distance_drop": 0.30,
    "min_colour_channels": 3,
}
#: The cube whose rendered material the treatment changes.  Its identity -- the
#: name the prompt, the prim and every telemetry key use -- stays "blue".
THIRD_CUBE = "blue"
CUBES = ("red", "yellow", "blue")
SHIPPED_THIRD_ALBEDO = (0.05, 0.20, 0.80)
VARIANT_THIRD_ALBEDO = (0.044, 0.147, 0.113)
SHIPPED_PROFILE = "configs/profiles/isaac-g1-sonic-blockstacking-dex3.json"
VARIANT_PROFILE = "configs/profiles/isaac-g1-sonic-blockstacking-dex3-green-third.json"
#: The demonstration measurement the variant was derived to match, with the
#: provenance string shipped next to the cube's material in the variant profile.
DEMO_THIRD_SRGB = (70.1, 123.5, 87.2)
DEMO_THIRD_HUE = 69.2
#: The demonstration's own per-frame spread of that statistic (p05..p95), which is
#: what "the same colour" can mean across two cameras, poses and exposures.
DEMO_THIRD_P05 = (40.1, 81.5, 52.2)
DEMO_THIRD_P95 = (89.8, 143.7, 107.6)
DEMO_THIRD_FRAMES = 136
DEMO_THIRD_PROVENANCE = (
    "median over 136 third-cube detections in 24 sampled G1_Dex3_BlockStacking "
    "training episodes (6 time points each), same blob statistic as the sim side"
)
#: The blob classes used for every colour measurement in both domains.
COLOUR_CLASSES = {
    "red": lambda h, s, v: ((h < 10) | (h > 170)) & (s > 120) & (v > 70),
    "yellow": lambda h, s, v: (h >= 15) & (h < 35) & (s > 110) & (v > 90),
    "green": lambda h, s, v: (h >= 45) & (h < 95) & (s > 80) & (v > 50),
    "blue": lambda h, s, v: (h >= 95) & (h < 140) & (s > 80) & (v > 50),
}
#: Every third head-camera frame is measured, exactly as the transfer calibration
#: that produced the variant albedo did.
COLOUR_STRIDE = 3
#: Blob area band: below it the blob is a highlight, above it the storage crates
#: under the worktop (which share the blue and yellow classes with the cubes).
BLOB_MIN_AREA_PX = 60
BLOB_MAX_AREA_PX = 4000
#: The cubes rest on the 0.994 m worktop, which the head camera images in the
#: upper part of the frame; the storage crates below it start further down.  A
#: blob whose centroid is below this row is not a cube on the worktop.
WORKTOP_CENTROID_Y_MAX = 250
#: A measured palm within this distance of a cube centre counts as contact; the
#: half-side of a 0.05 m cube is 0.025 m and its half diagonal 0.043 m.
CONTACT_M = 0.075
#: Red and yellow are unchanged material in both cells, so any difference beyond
#: this is a lighting/measurement drift worth reporting as such (sRGB per channel).
CONTROL_TOLERANCE_SRGB = 15.0
CONTROL_TOLERANCE_HUE = 6.0
#: The third cube must change by at least this much for the treatment to be real.
MIN_THIRD_CUBE_DISTANCE_SRGB = 25.0


class AbError(RuntimeError):
    """The experiment cannot be read; the message is meant for the operator."""


# ---------------------------------------------------------------- plumbing


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def warm_ab():
    module = globals().get("_warm_ab")
    if module is None:
        module = _load_script("warm_ab", REPO_ROOT / "scripts" / "groot-token-warmstart-ab.py")
        globals()["_warm_ab"] = module
    return module


def handshake_ab():
    module = globals().get("_handshake_ab")
    if module is None:
        module = _load_script("handshake_ab",
                              REPO_ROOT / "scripts" / "groot-initial-pose-handshake-ab.py")
        globals()["_handshake_ab"] = module
    return module


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n",
                    encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                value = f"{value:.6g}"
            elif isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            values.append("" if value is None else str(value).replace(",", ";"))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dir(cell: str, out: Path) -> Path:
    return out / "runs" / cell


def cell_session_names(cell: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for name, _ in CELL_REPEATS[cell]))


def policy_windows(session: Path) -> list[tuple[float, float]]:
    """The policy windows of the rollouts this session still holds.

    A rejected rollout is moved out of ``rollouts/``, so its window drops out
    here as well: the appearance and the metrics are read from the runs the
    experiment actually analyses.
    """
    session_json = read_json(session / "session.json")
    kept = {int(path.name.split("-")[-1]) for path in (session / "rollouts").glob("rollout-*")}
    windows = []
    for index, entry in enumerate(session_json.get("rollouts") or [], start=1):
        if index in kept:
            windows.append((float(entry["start_wall_ns"]) / 1e9,
                            float(entry["stop_wall_ns"]) / 1e9))
    return windows


# ------------------------------------------------------ colour measurement


def srgb_to_linear(values: Iterable[float]) -> np.ndarray:
    """The sRGB EOTF, applied per channel (the same transfer both domains use)."""
    c = np.asarray(values, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def cube_blob(image: np.ndarray, name: str,
              *, centroid_y_max: float | None = WORKTOP_CENTROID_Y_MAX) -> dict[str, Any] | None:
    """One frame's largest blob of a colour class, with the blob's mean colour.

    The same statistic in both domains: the training episodes were measured on
    the same definition, which is why the variant's albedo is a transferred value
    rather than a guess.  The vertical guard keeps the measurement on the worktop:
    the storage crates below it share the blue and yellow classes.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = COLOUR_CLASSES[name](hue, saturation, value).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best: tuple[int, int, int, int, int, int, float] | None = None
    for index in range(1, count):
        area = int(stats[index][4])
        x, y, width, height = (int(channel) for channel in stats[index][:4])
        if not (BLOB_MIN_AREA_PX <= area <= BLOB_MAX_AREA_PX):
            continue
        ys, xs = np.where(labels == index)
        centroid_y = float(ys.mean())
        if centroid_y_max is not None and centroid_y >= centroid_y_max:
            continue
        if best is None or area > best[0]:
            best = (area, index, x, y, width, height, centroid_y)
    if best is None:
        return None
    area, index, x, y, width, height, centroid_y = best
    ys, xs = np.where(labels == index)
    pixels = image[ys, xs]
    return {
        "area_px": area,
        "bbox_xywh": [x, y, width, height],
        "centroid_xy": [float(xs.mean()), centroid_y],
        "rgb_mean": [float(channel) for channel in pixels.mean(axis=0)[::-1]],
        "rgb_median": [float(channel) for channel in np.median(pixels, axis=0)[::-1]],
        "hsv_mean": [float(channel) for channel in hsv[ys, xs].mean(axis=0)],
        "touches_border": bool(x <= 0 or y <= 0 or x + width >= image.shape[1]
                               or y + height >= image.shape[0]),
        "patch": pixels,
    }


def frame_rows(session: Path) -> list[dict[str, Any]]:
    """Every head-camera frame the bridge recorded, in capture order."""
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(session / "raw" / "telemetry" / "bridge-telemetry.jsonl"):
        if row.get("kind") != "frame":
            continue
        jpeg = row.get("camera_jpeg")
        if not jpeg:
            continue
        rows.append({
            "index": int(row.get("index", len(rows))),
            "wall_time": row["wall_time_ns"] / 1e9,
            "path": session / "raw" / "telemetry" / "head_camera" / Path(jpeg).name,
        })
    return rows


def summarise_blobs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"frames": 0}
    rgb = np.array([row["rgb_mean"] for row in rows], dtype=float)
    hsv = np.array([row["hsv_mean"] for row in rows], dtype=float)
    area = np.array([row["area_px"] for row in rows], dtype=float)
    fully = [row for row in rows if not row["touches_border"]]
    rgb_full = np.array([row["rgb_mean"] for row in fully], dtype=float) if fully else rgb
    return {
        "frames": len(rows),
        "frames_fully_visible": len(fully),
        "median_rgb": [round(float(value), 2) for value in np.median(rgb, axis=0)],
        "median_rgb_fully_visible": [round(float(value), 2) for value in np.median(rgb_full, axis=0)],
        "median_linear_fully_visible": [
            round(float(value), 4) for value in srgb_to_linear(np.median(rgb_full, axis=0))],
        "median_hue": round(float(np.median(hsv[:, 0])), 2),
        "median_saturation": round(float(np.median(hsv[:, 1])), 2),
        "median_value": round(float(np.median(hsv[:, 2])), 2),
        "median_area_px": float(np.median(area)),
        "median_bbox_xywh": [float(value) for value in np.median(
            np.array([row["bbox_xywh"] for row in rows], dtype=float), axis=0)],
    }


def whole_cube_colour(image: np.ndarray, name: str,
                      *, pad: int = 12) -> dict[str, Any] | None:
    """The cube's colour over its whole box, top face included.

    The class mask is threshold-based, and the same thresholds keep the canonical
    blue cube's pale top face but drop the variant green one's, so the blob
    statistic is not face-for-face comparable between the cells.  This reading
    takes every coloured pixel of the cube's box instead, top face included, and
    is reported next to the blob statistic: it is the whole-cube appearance.
    """
    blob = cube_blob(image, name)
    if blob is None:
        return None
    x, y, width, height = blob["bbox_xywh"]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + width + pad), min(image.shape[0], y + height + pad)
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    pixels = crop.reshape(-1, 3).astype(float)[:, ::-1]
    keep = (hsv[:, 1] > 20) & (hsv[:, 2] > 40)
    if int(keep.sum()) < 50:
        return None
    kept = pixels[keep]
    luminance = kept.mean(axis=1)
    lit = kept[luminance >= np.percentile(luminance, 80)]
    return {
        "rgb_median": [float(value) for value in np.median(kept, axis=0)],
        "rgb_lit_median": [float(value) for value in np.median(lit, axis=0)],
        "pixels": int(keep.sum()),
        "box_px": [int(x1 - x0), int(y1 - y0)],
    }


def demo_envelope(value: Sequence[float]) -> dict[str, Any]:
    """Where one measured colour sits inside the demonstration's own distribution."""
    inside = [bool(DEMO_THIRD_P05[channel] <= value[channel] <= DEMO_THIRD_P95[channel])
              for channel in range(3)]
    return {
        "value": [float(channel) for channel in value],
        "inside_p05_p95": inside,
        "all_channels_inside": all(inside),
        "delta_to_median": [float(value[channel] - DEMO_THIRD_SRGB[channel]) for channel in range(3)],
    }


def measure_cell(cell: str, out: Path, *, stride: int = COLOUR_STRIDE) -> dict[str, Any]:
    """The third cube and the two control cubes, measured from the policy's own view.

    Frames are pooled over the cell's sessions and restricted to the policy
    windows of the rollouts the experiment kept, so a rejected window cannot
    enter the appearance evidence.  Every class is measured twice: once with the
    worktop guard (the primary reading, the object on the worktop) and once
    without it, because the transfer calibration that produced the variant's
    albedo used the unguarded selection.
    """
    third_class = "green" if cell == "G" else "blue"
    names = ("red", "yellow", third_class)
    guarded: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    unguarded: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    whole: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    sessions: dict[str, Any] = {}
    for session_name in cell_session_names(cell):
        session = out / "runs" / session_name
        frames = frame_rows(session)
        windows = policy_windows(session)
        kept = [row for row in frames
                if any(low <= row["wall_time"] <= high for low, high in windows)]
        sessions[session_name] = {
            "frames_in_session": len(frames),
            "frames_in_valid_policy_windows": len(kept),
            "policy_windows": windows,
        }
        for row in kept[::stride]:
            image = cv2.imread(str(row["path"]))
            if image is None:
                continue
            for name in names:
                for store, guard in ((guarded, WORKTOP_CENTROID_Y_MAX), (unguarded, None)):
                    blob = cube_blob(image, name, centroid_y_max=guard)
                    if blob is None:
                        continue
                    blob = {key: value for key, value in blob.items() if key != "patch"}
                    blob.update({"frame_index": row["index"], "wall_time": row["wall_time"]})
                    store[name].append(blob)
                cube = whole_cube_colour(image, name)
                if cube is not None:
                    cube.update({"frame_index": row["index"], "wall_time": row["wall_time"]})
                    whole[name].append(cube)
    classes = {name: summarise_blobs(rows) for name, rows in guarded.items()}
    classes_unguarded = {name: summarise_blobs(rows) for name, rows in unguarded.items()}
    whole_cube = {
        name: {
            "frames": len(rows),
            "median_rgb": (None if not rows else [float(value) for value in np.median(
                np.array([row["rgb_median"] for row in rows]), axis=0)]),
            "median_rgb_lit": (None if not rows else [float(value) for value in np.median(
                np.array([row["rgb_lit_median"] for row in rows]), axis=0)]),
        }
        for name, rows in whole.items()
    }
    third = classes[third_class]
    third_whole = whole_cube[third_class]
    envelope = {
        "blob": (None if not third.get("median_rgb") else demo_envelope(third["median_rgb"])),
        "whole_cube": (None if not third_whole.get("median_rgb")
                       else demo_envelope(third_whole["median_rgb"])),
    }
    return {
        "cell": cell,
        "treatment": CELL_TREATMENT[cell],
        "third_cube_class": third_class,
        "sessions": sessions,
        "classes": classes,
        "classes_unguarded": classes_unguarded,
        "whole_cube": whole_cube,
        "third_cube": third,
        "third_cube_median_rgb": third.get("median_rgb"),
        "third_cube_median_rgb_unguarded": classes_unguarded[third_class].get("median_rgb"),
        "third_cube_whole_cube_rgb": third_whole.get("median_rgb"),
        "third_cube_median_hue": third.get("median_hue"),
        "demonstration_envelope": envelope,
    }


def colour_validation(out: Path) -> dict[str, Any]:
    """Was the treatment applied, and is it the demonstration's colour?"""
    cells = {cell: measure_cell(cell, out) for cell in CELLS}
    a, g = cells["A"], cells["G"]
    problems: list[str] = []
    for cell, entry in cells.items():
        sampled = sum(session["frames_in_valid_policy_windows"] for session in entry["sessions"].values())
        if not sampled:
            problems.append(f"cell {cell} has no head-camera frames inside its valid policy windows")
        if entry["third_cube"].get("frames", 0) < 10:
            problems.append(f"cell {cell}: too few third-cube detections "
                            f"({entry['third_cube'].get('frames', 0)})")
    third_delta = None
    if a.get("third_cube_median_rgb") and g.get("third_cube_median_rgb"):
        third_delta = [
            float(g["third_cube_median_rgb"][channel] - a["third_cube_median_rgb"][channel])
            for channel in range(3)
        ]
        distance = float(np.linalg.norm(np.array(g["third_cube_median_rgb"], dtype=float)
                                        - np.array(a["third_cube_median_rgb"], dtype=float)))
        if distance < MIN_THIRD_CUBE_DISTANCE_SRGB:
            problems.append(f"the third cube barely changed between the cells "
                            f"({distance:.1f} sRGB < {MIN_THIRD_CUBE_DISTANCE_SRGB})")
    demo_delta = None
    if g.get("third_cube_median_rgb"):
        demo_delta = [float(g["third_cube_median_rgb"][channel] - DEMO_THIRD_SRGB[channel])
                      for channel in range(3)]
    # The two unchanged cubes are the controls: same material, same lighting, so
    # a difference between the cells there is drift in the measurement, not the
    # treatment.
    controls: dict[str, Any] = {}
    for name in ("red", "yellow"):
        a_entry, g_entry = a["classes"].get(name, {}), g["classes"].get(name, {})
        if not (a_entry.get("frames") and g_entry.get("frames")):
            continue
        rgb_delta = [float(g_entry["median_rgb"][channel] - a_entry["median_rgb"][channel])
                     for channel in range(3)]
        hue_delta = float(g_entry["median_hue"] - a_entry["median_hue"])
        # The red class spans OpenCV's hue wrap (h<10 or h>170), so its mean hue is
        # not comparable between two pixel mixes; red is controlled on RGB alone.
        hue_checked = name != "red"
        unchanged = (max(abs(value) for value in rgb_delta) <= CONTROL_TOLERANCE_SRGB
                     and (not hue_checked or abs(hue_delta) <= CONTROL_TOLERANCE_HUE))
        controls[name] = {
            "median_rgb_A": a_entry["median_rgb"],
            "median_rgb_G": g_entry["median_rgb"],
            "rgb_delta": rgb_delta,
            "hue_A": a_entry["median_hue"],
            "hue_G": g_entry["median_hue"],
            "hue_delta": hue_delta,
            "hue_checked": hue_checked,
            "area_A_px": a_entry["median_area_px"],
            "area_G_px": g_entry["median_area_px"],
            "unchanged_within_tolerance": unchanged,
        }
        if not unchanged:
            problems.append(f"the {name} control cube changed between the cells "
                            f"({rgb_delta} sRGB, hue {hue_delta:+.1f})")
    scale = {
        "third_cube_area_A_px": a["third_cube"].get("median_area_px"),
        "third_cube_area_G_px": g["third_cube"].get("median_area_px"),
        "third_cube_bbox_A": a["third_cube"].get("median_bbox_xywh"),
        "third_cube_bbox_G": g["third_cube"].get("median_bbox_xywh"),
    }
    applied = not problems
    return {
        "schema_version": SCHEMA_VERSION,
        "statistic": ("mean RGB of the largest colour-class blob in one head-camera frame, "
                      "median across every third frame; hue/saturation/value from the same blob"),
        "cells": cells,
        "third_cube_rgb_delta_G_minus_A": third_delta,
        "third_cube_rgb_delta_G_minus_demo": demo_delta,
        "demo_envelope": {
            "median_srgb": list(DEMO_THIRD_SRGB),
            "p05_srgb": list(DEMO_THIRD_P05),
            "p95_srgb": list(DEMO_THIRD_P95),
            "hue": DEMO_THIRD_HUE,
            "frames": DEMO_THIRD_FRAMES,
            "provenance": DEMO_THIRD_PROVENANCE,
        },
        "demo_third_cube_srgb": list(DEMO_THIRD_SRGB),
        "demo_third_cube_hue": DEMO_THIRD_HUE,
        "demo_provenance": DEMO_THIRD_PROVENANCE,
        "control_cubes": controls,
        "scale_controls": scale,
        "treatment_applied": applied,
        "problems": problems,
    }


# --------------------------------------------------------------- validity


def validity_record(rollout: Any, *, baseline_jump: float | None = None) -> dict[str, Any]:
    """Experiment 08's validity gates, with the handshake required in *both* cells.

    The handshake is not the treatment here: every cell runs it, so a cell that
    did not deliver it is invalid rather than a different condition.
    """
    module = handshake_ab()
    limits = module.GATE_LIMITS
    actions = module.applied_actions(rollout)
    settle = rollout.settle
    relative = settle["relative"]
    calm = (relative >= -module.CALM_TAIL_S) & (relative < 0.0)
    joint_speed = (np.abs(settle["measured_velocity"][calm][:, warm_ab().ARM_HAND]).max(axis=1)
                   if calm.any() else np.zeros(0))
    palm_speed = np.concatenate([
        module.finite_difference_speed(settle["measured_palm"][side], settle["wall"])[calm]
        for side in ("left", "right")]) if calm.any() else np.zeros(0)
    calm_indices = np.flatnonzero(calm)
    reference_index = max(0, int(calm_indices[0]) - 1) if calm_indices.size else 0
    pelvis = settle["root"][calm]
    pelvis_reference = settle["root"][reference_index]
    pelvis_drift = (float(np.abs(pelvis[:, 2] - pelvis_reference[2]).max())
                    if calm.any() else float("nan"))
    pelvis_xy = (float(np.linalg.norm(pelvis[:, :2] - pelvis_reference[:2], axis=1).max())
                 if calm.any() else float("nan"))

    def cubes_over(low: float, high: float) -> dict[str, float]:
        window = [row for row in rollout.cube_samples if low <= row["wall_time_ns"] / 1e9 < high]
        first = window[0]["cubes"] if window else {}
        last = window[-1]["cubes"] if window else {}
        return {name: float(np.linalg.norm(np.asarray(last[name])[:2] - np.asarray(first[name])[:2]))
                for name in first if name in last}

    table = warm_ab().scene_constants()["table_surface_m"]
    cube_displacement = cubes_over(rollout.reset_wall + 1.0, rollout.start_wall)
    settle_cubes = [row for row in rollout.cube_samples
                    if rollout.reset_wall + 1.0 <= row["wall_time_ns"] / 1e9 < rollout.start_wall]
    minimum_cube_z = min((min(np.asarray(entry)[2] for entry in row["cubes"].values())
                          for row in rollout.cube_samples), default=float("nan"))
    pre = rollout.start_settle_index
    held_target = settle["target"][pre]
    transition = (relative > 0.0) & (relative < 0.5)
    target_jump = (float(np.abs(settle["target"][transition] - held_target[None, :]).max())
                   if transition.any() else 0.0)
    finite = bool(np.isfinite(rollout.target).all() and np.isfinite(rollout.measured).all())
    fall = bool(rollout.meta.get("fall", {}).get("detected"))
    if not rollout.intended:
        raise AbError(f"{rollout.label}: no client initial-pose message to compare delivery against")
    delivery = module.delivery_record(rollout, actions, rollout.intended)
    first_policy = delivery["first_policy_token_relative_s"]

    gates: dict[str, Any] = {
        "G1_delivery": {
            "sources_in_settle": sorted(delivery["applied_by_source_during_settle"]),
            "initial_pose_messages": delivery["initial_pose_messages"],
            "matches_intended_token": delivery["initial_pose_matches_intended"],
            "distinct_tokens": delivery["initial_pose_distinct_tokens"],
            "first_relative_to_reset_s": delivery["initial_pose_first_relative_to_reset_s"],
            "rate_hz": delivery["initial_pose_rate_hz"],
            "limits": {"start_slack_s": limits["delivery_start_slack_s"],
                       "min_rate_hz": limits["delivery_min_hz"]},
        },
        "G2_calm_at_settle_end": {
            "frames": int(calm.sum()),
            "joint_speed_p95_rad_s": float(np.percentile(joint_speed, 95)) if joint_speed.size else float("nan"),
            "joint_speed_max_rad_s": float(joint_speed.max()) if joint_speed.size else float("nan"),
            "palm_speed_p95_m_s": float(np.percentile(palm_speed, 95)) if palm_speed.size else float("nan"),
            "limits": {"joint_speed_p95_rad_s": limits["calm_joint_speed_p95_rad_s"],
                       "joint_speed_max_rad_s": limits["calm_joint_speed_max_rad_s"],
                       "palm_speed_p95_m_s": limits["calm_palm_speed_p95_m_s"]},
        },
        "G3_pelvis_stable": {
            "height_drift_m": pelvis_drift,
            "height_min_m": float(pelvis[:, 2].min()) if calm.any() else float("nan"),
            "xy_drift_m": pelvis_xy,
            "limits": {"height_drift_m": limits["pelvis_height_drift_max_m"],
                       "height_min_m": limits["pelvis_height_min_m"],
                       "xy_drift_m": limits["pelvis_xy_drift_max_m"]},
        },
        "G4_scene_intact": {
            "cube_displacement_m": cube_displacement,
            "max_cube_displacement_m": (float(max(cube_displacement.values()))
                                        if cube_displacement else float("nan")),
            "minimum_cube_centre_z_m": minimum_cube_z,
            "table_surface_m": table,
            "probe_samples_in_settle": len(settle_cubes),
            "limit": limits["cube_displacement_max_m"],
        },
        "G5_handoff_bounded": {
            "target_jump_rad": target_jump,
            "pre_declared_ceiling_rad": limits["transition_target_jump_ceiling_rad"],
            "within_pre_declared_ceiling": bool(
                target_jump <= limits["transition_target_jump_ceiling_rad"]),
            "baseline_max_jump_rad": baseline_jump,
            "sim_finite": finite,
            "fall": fall,
        },
        "G6_token_hygiene": {
            "policy_tokens_after_start": delivery["policy_tokens_after_start"],
            "non_groot_after_first_policy_token": int(len([
                row for row in actions
                if row["source"] != "groot" and rollout.start_wall <= row["wall_time"] < rollout.stop_wall
                and row["relative"] >= (first_policy if first_policy is not None else 0.0)])),
        },
    }
    gates["G1_delivery"]["passed"] = bool(
        delivery["initial_pose_messages"] > 0
        and delivery["initial_pose_matches_intended"]
        and delivery["initial_pose_distinct_tokens"] == 1
        and delivery["applied_by_source_during_settle"] == {
            "initial_pose": delivery["initial_pose_messages"]}
        and delivery["initial_pose_first_relative_to_reset_s"] is not None
        and delivery["initial_pose_first_relative_to_reset_s"] <= limits["delivery_start_slack_s"]
        and (delivery["initial_pose_rate_hz"] or 0.0) >= limits["delivery_min_hz"])
    gates["G2_calm_at_settle_end"]["passed"] = bool(
        joint_speed.size
        and np.percentile(joint_speed, 95) <= limits["calm_joint_speed_p95_rad_s"]
        and joint_speed.max() <= limits["calm_joint_speed_max_rad_s"]
        and np.percentile(palm_speed, 95) <= limits["calm_palm_speed_p95_m_s"])
    gates["G3_pelvis_stable"]["passed"] = bool(
        calm.any() and pelvis_drift <= limits["pelvis_height_drift_max_m"]
        and float(pelvis[:, 2].min()) >= limits["pelvis_height_min_m"]
        and pelvis_xy <= limits["pelvis_xy_drift_max_m"])
    gates["G4_scene_intact"]["passed"] = bool(
        cube_displacement and max(cube_displacement.values()) <= limits["cube_displacement_max_m"]
        and minimum_cube_z >= table - 0.01)
    gates["G5_handoff_bounded"]["passed"] = bool(
        finite and not fall
        and (baseline_jump is None
             or target_jump <= max(limits["transition_target_jump_ceiling_rad"],
                                   baseline_jump + limits["handoff_margin_over_baseline_rad"])))
    gates["G6_token_hygiene"]["passed"] = bool(
        delivery["policy_tokens_after_start"] > 0
        and gates["G6_token_hygiene"]["non_groot_after_first_policy_token"] == 0)
    gates["integrity_passed"] = all(
        gates[name]["passed"] for name in ("G2_calm_at_settle_end", "G3_pelvis_stable",
                                           "G4_scene_intact", "G5_handoff_bounded",
                                           "G6_token_hygiene"))
    gates["passed"] = bool(gates["integrity_passed"] and gates["G1_delivery"]["passed"])
    start_support = {group: rollout.start_support()[group]
                     for group in ("arms", "left_arm", "right_arm", "hands", "upper_body")}
    return {
        "label": rollout.label,
        "cell": rollout.cell,
        "rollout": rollout.index,
        "directory": rel(rollout.directory),
        "reset_wall_time": float(rollout.reset_wall),
        "start_wall_time": float(rollout.start_wall),
        "stop_wall_time": float(rollout.stop_wall),
        "settle_seconds": float(rollout.start_wall - rollout.reset_wall),
        "onset_seconds_after_start": float(rollout.meta["onset"]["wall_time"] - rollout.start_wall),
        "rollout_seconds": float(rollout.wall[-1] - rollout.start_wall),
        "applied_actions_total": int(rollout.meta.get("applied_actions") or 0),
        "fall": fall,
        "delivery": delivery,
        "start_state": {
            "measured_palm_z_m": {side: float(rollout.settle["measured_palm"][side]
                                              [rollout.start_settle_index][2])
                                  for side in ("left", "right")},
            "support": start_support,
        },
        "gates": gates,
    }


# ---------------------------------------------------------------- metrics


def _rotate_into_base(rollout: Any, mask: np.ndarray) -> np.ndarray:
    """The rotation that takes a world point into the robot's base frame, per frame."""
    module = warm_ab()
    return np.array([module.quaternion_wxyz_to_matrix(quaternion).T
                     for quaternion in rollout.root_quat[mask]])


def per_cube_distances(rollout: Any, side: str, mask: np.ndarray, *, measured: bool = False) -> dict[str, np.ndarray]:
    palm = (rollout.measured_palm if measured else rollout.target_palm)[side][mask]
    origins = rollout.root[mask]
    rotations = _rotate_into_base(rollout, mask)
    out: dict[str, np.ndarray] = {}
    for name, centre in rollout.cubes.items():
        local = np.einsum("nij,nj->ni", rotations, np.asarray(centre, dtype=float) - origins)
        out[name] = np.linalg.norm(palm - local, axis=1)
    return out


def nearest_cube_series(rollout: Any, side: str, mask: np.ndarray) -> list[str]:
    distances = per_cube_distances(rollout, side, mask)
    if not distances:
        return []
    names = list(distances)
    stacked = np.stack([distances[name] for name in names], axis=1)
    return [names[int(index)] for index in np.argmin(stacked, axis=1)]


def cube_outcomes(rollout: Any, *, window: tuple[float, float]) -> dict[str, Any]:
    """What the cubes did over the policy window, from the live scene probe."""
    low, high = rollout.start_wall + window[0], rollout.stop_wall
    samples = [row for row in rollout.cube_samples if low <= row["wall_time_ns"] / 1e9 <= high]
    out: dict[str, Any] = {"probe_samples": len(samples)}
    if not samples:
        return out
    first, last = samples[0]["cubes"], samples[-1]["cubes"]
    for name in CUBES:
        if name not in first or name not in last:
            continue
        start = np.asarray(first[name], dtype=float)
        end = np.asarray(last[name], dtype=float)
        out[f"displacement_xy_{name}_m"] = float(np.linalg.norm(end[:2] - start[:2]))
        out[f"lift_z_{name}_m"] = float(end[2] - start[2])
    return out


def window_metrics(rollout: Any, name: str) -> dict[str, Any]:
    """Policy output and its realisation, per window, kept apart."""
    module = handshake_ab()
    mask = rollout.window(name)
    low_s, high_s = WINDOWS_S[name]
    entry: dict[str, Any] = {
        "frames": int(mask.sum()),
        "seconds": float(high_s if high_s is not None else max(0.0, rollout.relative[-1])),
    }
    for side in ("left", "right"):
        target_z = rollout.target_palm[side][mask][:, 2]
        measured_z = rollout.measured_palm[side][mask][:, 2]
        entry[f"target_palm_z_{side}_median_m"] = float(np.median(target_z))
        entry[f"target_palm_z_{side}_mean_m"] = float(target_z.mean())
        entry[f"target_palm_z_{side}_min_m"] = float(target_z.min())
        entry[f"measured_palm_z_{side}_median_m"] = float(np.median(measured_z))
        entry[f"measured_palm_z_{side}_min_m"] = float(measured_z.min())
        for cube, distance in per_cube_distances(rollout, side, mask).items():
            entry[f"target_palm_distance_{side}_{cube}_min_m"] = float(distance.min())
            entry[f"target_palm_distance_{side}_{cube}_median_m"] = float(np.median(distance))
        for cube, distance in per_cube_distances(rollout, side, mask, measured=True).items():
            entry[f"measured_palm_distance_{side}_{cube}_min_m"] = float(distance.min())
    entry["target_palm_z_mean_m"] = 0.5 * (entry["target_palm_z_left_median_m"]
                                           + entry["target_palm_z_right_median_m"])
    entry["measured_palm_z_mean_m"] = 0.5 * (entry["measured_palm_z_left_median_m"]
                                             + entry["measured_palm_z_right_median_m"])
    for cube in CUBES:
        entry[f"target_palm_distance_any_{cube}_min_m"] = min(
            entry[f"target_palm_distance_{side}_{cube}_min_m"] for side in ("left", "right"))
        entry[f"measured_palm_distance_any_{cube}_min_m"] = min(
            entry[f"measured_palm_distance_{side}_{cube}_min_m"] for side in ("left", "right"))
    entry["target_palm_distance_any_cube_min_m"] = min(
        entry[f"target_palm_distance_any_{cube}_min_m"] for cube in CUBES)
    entry["measured_palm_distance_any_cube_min_m"] = min(
        entry[f"measured_palm_distance_any_{cube}_min_m"] for cube in CUBES)
    # The acting hand: whichever palm the policy brings closest to the cubes in
    # this window.  The premise is about one arm reaching too high, so the acting
    # side is reported on its own next to the two-sided mean.
    entry["acting_side"] = (
        "left" if min(entry[f"target_palm_distance_left_{cube}_min_m"] for cube in CUBES)
        <= min(entry[f"target_palm_distance_right_{cube}_min_m"] for cube in CUBES) else "right")
    acting = entry["acting_side"]
    entry["acting_target_palm_z_median_m"] = entry[f"target_palm_z_{acting}_median_m"]
    entry["acting_target_palm_distance_third_min_m"] = entry[
        f"target_palm_distance_{acting}_{THIRD_CUBE}_min_m"]
    entry["acting_measured_palm_distance_any_cube_min_m"] = min(
        entry[f"measured_palm_distance_{acting}_{cube}_min_m"] for cube in CUBES)
    entry["measured_contact_any_cube"] = bool(
        entry["acting_measured_palm_distance_any_cube_min_m"] <= CONTACT_M)
    # The task-phase proxy: which cube the acting palm is nearest to, per frame.
    nearest = nearest_cube_series(rollout, acting, mask)
    entry["phase_nearest_fraction"] = {
        cube: (float(sum(1 for value in nearest if value == cube) / len(nearest)) if nearest else 0.0)
        for cube in CUBES}
    entry["phase_first_nearest"] = nearest[0] if nearest else None
    entry["phase_last_nearest"] = nearest[-1] if nearest else None
    entry["phase_order"] = [nearest[0]] + [value for index, value in enumerate(nearest[1:], start=1)
                                           if value != nearest[index - 1]] if nearest else []
    # The policy's own applied tokens and their support against the demonstration
    # cloud (experiment 05's statistic, the same calibration the other campaigns use).
    # The active window ends at this rollout's Stop: the session's telemetry
    # continues into the next rollout, so an open upper bound would count it.
    actions = module.applied_actions(rollout)
    high_wall = rollout.start_wall + high_s if high_s is not None else rollout.stop_wall
    tokens = [row for row in actions
              if row["source"] == "groot"
              and rollout.start_wall + low_s <= row["wall_time"] < high_wall]
    entry["groot_tokens"] = len(tokens)
    entry["groot_token_support"] = module.token_support([row["token"] for row in tokens])
    if len(tokens) > 1:
        # Token dynamics: how far the applied token moves per step and per second.
        steps = [float(np.abs(np.asarray(tokens[index + 1]["token"]) - np.asarray(tokens[index]["token"])).sum())
                 for index in range(len(tokens) - 1)]
        span = float(tokens[-1]["wall_time"] - tokens[0]["wall_time"])
        entry["groot_token_step_l1_mean"] = float(np.mean(steps))
        entry["groot_token_step_l1_per_s"] = (float(np.sum(steps) / span) if span > 0 else None)
        entry["groot_token_step_span_s"] = span
    if tokens:
        first = tokens[0]
        entry["first_policy_token_relative_s"] = float(first["relative"])
        entry["first_policy_token_frame_index"] = int(first["frame_index"])
    entry["cube_outcomes"] = cube_outcomes(rollout, window=(low_s, high_s))
    entry["stack"] = rollout.meta.get("stack")
    return entry


def rollout_metrics(rollout: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": rollout.label,
        "cell": rollout.cell,
        "rollout": rollout.index,
        "directory": rel(rollout.directory),
        "session": rel(rollout.session),
        "onset_seconds_after_start": float(rollout.meta["onset"]["wall_time"] - rollout.start_wall),
        "rollout_seconds": float(rollout.wall[-1] - rollout.start_wall),
        "cubes": {name: [float(value) for value in centre] for name, centre in rollout.cubes.items()},
        "windows": {name: window_metrics(rollout, name) for name in WINDOWS_S},
    }
    return record


# ----------------------------------------------------------- paired effect


def paired_effect(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    """One pair's deltas, treatment minus baseline, direction kept."""
    out: dict[str, Any] = {"pair": f"{baseline['label']}/{treatment['label']}"}
    for window in WINDOWS_S:
        a, g = baseline["windows"][window], treatment["windows"][window]
        for metric, key, better in (
            ("target_palm_z_mean", "target_palm_z_mean_m", "lower"),
            ("acting_target_palm_z", "acting_target_palm_z_median_m", "lower"),
            ("target_palm_z_left", "target_palm_z_left_median_m", "lower"),
            ("target_palm_z_right", "target_palm_z_right_median_m", "lower"),
            ("measured_palm_z_mean", "measured_palm_z_mean_m", "lower"),
            ("target_palm_distance_any_cube", "target_palm_distance_any_cube_min_m", "lower"),
            ("target_palm_distance_third", "acting_target_palm_distance_third_min_m", "lower"),
            ("measured_palm_distance_any_cube", "acting_measured_palm_distance_any_cube_min_m", "lower"),
        ):
            value_a, value_g = a.get(key), g.get(key)
            out[f"{window}_{metric}_baseline_m"] = value_a
            out[f"{window}_{metric}_treatment_m"] = value_g
            out[f"{window}_{metric}_delta_m"] = (None if value_a is None or value_g is None
                                                 else value_g - value_a)
            if value_a:
                out[f"{window}_{metric}_relative_delta"] = (value_g - value_a) / value_a
            out[f"{window}_{metric}_better"] = better
            out[f"{window}_{metric}_direction"] = (
                None if value_a is None or value_g is None else
                ("lower" if value_g < value_a else "higher" if value_g > value_a else "equal"))
        for cube in CUBES:
            key = f"target_palm_distance_any_{cube}_min_m"
            value_a, value_g = a.get(key), g.get(key)
            out[f"{window}_target_palm_distance_{cube}_baseline_m"] = value_a
            out[f"{window}_target_palm_distance_{cube}_treatment_m"] = value_g
            out[f"{window}_target_palm_distance_{cube}_delta_m"] = (
                None if value_a is None or value_g is None else value_g - value_a)
            out[f"{window}_target_palm_distance_{cube}_relative_delta"] = (
                None if value_a is None or value_g is None or not value_a else (value_g - value_a) / value_a)
        support_a = a["groot_token_support"]
        support_g = g["groot_token_support"]
        if support_a.get("frames") and support_g.get("frames"):
            out[f"{window}_token_support_median_baseline"] = support_a["median"]
            out[f"{window}_token_support_median_treatment"] = support_g["median"]
            out[f"{window}_token_support_median_delta"] = support_g["median"] - support_a["median"]
            out[f"{window}_token_support_median_relative_delta"] = (
                (support_g["median"] - support_a["median"]) / support_a["median"]
                if support_a["median"] else None)
            out[f"{window}_token_support_p99_baseline"] = support_a["loeo_p99"]
            out[f"{window}_token_support_p99_treatment"] = support_g["loeo_p99"]
        out[f"{window}_tokens_baseline"] = a["groot_tokens"]
        out[f"{window}_tokens_treatment"] = g["groot_tokens"]
        for key in ("groot_token_step_l1_mean", "groot_token_step_l1_per_s"):
            value_a, value_g = a.get(key), g.get(key)
            out[f"{window}_{key}_baseline"] = value_a
            out[f"{window}_{key}_treatment"] = value_g
            out[f"{window}_{key}_delta"] = (None if value_a is None or value_g is None
                                            else value_g - value_a)
        for cube in CUBES:
            for field in (f"displacement_xy_{cube}_m", f"lift_z_{cube}_m"):
                value_a = a["cube_outcomes"].get(field)
                value_g = g["cube_outcomes"].get(field)
                out[f"{window}_{field}_baseline"] = value_a
                out[f"{window}_{field}_treatment"] = value_g
                out[f"{window}_{field}_delta"] = (None if value_a is None or value_g is None
                                                  else value_g - value_a)
        out[f"{window}_phase_order_baseline"] = a["phase_order"]
        out[f"{window}_phase_order_treatment"] = g["phase_order"]
        out[f"{window}_phase_last_nearest_baseline"] = a["phase_last_nearest"]
        out[f"{window}_phase_last_nearest_treatment"] = g["phase_last_nearest"]
    return out


def _t_critical(df: int, confidence: float = 0.95) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 1.96)


def summarise_metric(values: Sequence[float], baseline_values: Sequence[float]) -> dict[str, Any]:
    """Mean paired delta with a 95% CI, against the baseline cell's repeat spread."""
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {"n": 0}
    mean = float(np.mean(clean))
    sd = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    half = _t_critical(len(clean) - 1) * sd / math.sqrt(len(clean)) if len(clean) > 1 else 0.0
    baseline_clean = [float(value) for value in baseline_values
                      if value is not None and math.isfinite(value)]
    baseline_sd = float(np.std(baseline_clean, ddof=1)) if len(baseline_clean) > 1 else float("nan")
    baseline_mean = float(np.mean(baseline_clean)) if baseline_clean else float("nan")
    return {
        "n": len(clean),
        "mean_delta": mean,
        "sd_delta": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "baseline_mean": baseline_mean,
        "baseline_repeat_sd": baseline_sd,
        "standardised_effect_vs_baseline_sd": (mean / baseline_sd
                                               if baseline_sd and math.isfinite(baseline_sd) else None),
        "directions": [("lower" if value < 0 else "higher" if value > 0 else "equal") for value in clean],
    }


def decide(effects: Sequence[dict[str, Any]], colour: dict[str, Any],
           validity: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The single A/B/C/D decision, from the pre-declared acceptance rule."""
    if not colour.get("treatment_applied"):
        return {"decision": "D", "reason": "the treatment could not be validly applied visually: "
                                           + "; ".join(colour.get("problems") or ["unknown"])}
    invalid = [row["label"] for row in validity if not row["gates"]["passed"]]
    if invalid:
        return {"decision": "D", "reason": f"invalid runs: {', '.join(invalid)}"}
    required = ACCEPTANCE["paired_repeats_required"]
    window = "active"
    z_deltas = [effect.get(f"{window}_target_palm_z_mean_delta_m") for effect in effects]
    z_agree = sum(1 for value in z_deltas if value is not None and value < 0)
    mean_z_drop = float(np.mean([value for value in z_deltas if value is not None])) if z_deltas else 0.0
    distance_drops = {
        cube: [effect.get(f"{window}_target_palm_distance_{cube}_relative_delta") for effect in effects]
        for cube in CUBES}
    distance_agree = {cube: sum(1 for value in values if value is not None and value <= -ACCEPTANCE[
        "min_relative_cube_distance_drop"]) for cube, values in distance_drops.items()}
    cube_hit = {cube: count >= required for cube, count in distance_agree.items()
                if all(value is not None for value in distance_drops[cube])}
    z_hit = z_agree >= required and mean_z_drop <= -ACCEPTANCE["min_palm_z_drop_m"]
    evidence = {
        "window": window,
        "target_palm_z_deltas_m": z_deltas,
        "target_palm_z_same_direction_pairs": z_agree,
        "mean_target_palm_z_delta_m": mean_z_drop,
        "target_palm_z_criterion_met": bool(z_hit),
        "cube_relative_distance_drops": distance_drops,
        "cube_pairs_over_threshold": distance_agree,
        "cube_criterion_met": bool(any(cube_hit.values())),
        "cubes_meeting_criterion": [cube for cube, hit in cube_hit.items() if hit],
    }
    if z_hit or any(cube_hit.values()):
        return {"decision": "A",
                "reason": "the treatment moved the pre-declared primary metric(s) in the "
                          "favourable direction in at least 2 of 3 paired repeats",
                "evidence": evidence}
    # Inconsistent: the same direction in the repeats but too small, or directions
    # that disagree with each other.
    z_signs = {("lower" if value < 0 else "higher" if value > 0 else "equal") for value in z_deltas
               if value is not None}
    cube_consistent = any(
        all(value is not None and value < 0 for value in values) for values in distance_drops.values())
    if len(z_signs) > 1 and not cube_consistent:
        return {"decision": "C",
                "reason": "the paired repeats disagree in direction and no cube-distance "
                          "criterion is met", "evidence": evidence}
    return {"decision": "B",
            "reason": "the appearance change was verified in the policy's own frames but the "
                      "pre-declared primary metrics did not move beyond the baseline's own "
                      "repeat spread", "evidence": evidence}


# ------------------------------------------------------------------ stages


def cell_rollouts(out: Path):
    """The cell's repeats in slot order, read once per process.

    A cell may draw its repeats from more than one session: cell G's rejected
    first repeat was re-run in its own session, and the replacement takes slot 1.
    """
    cache = globals().setdefault("_rollouts", {})
    key = str(out)
    if key not in cache:
        module = warm_ab()
        rollouts: dict[str, list[Any]] = {}
        for cell in CELLS:
            collected = []
            for session_name, index in CELL_REPEATS[cell]:
                directory = out / "runs" / session_name / "rollouts" / f"rollout-{index:02d}"
                if not directory.is_dir():
                    raise AbError(f"the declared repeat is missing: {directory}")
                rollout = module.Rollout(session_name, out, directory)
                rollout.session_name = session_name
                rollout.cell = cell
                collected.append(rollout)
            for slot, rollout in enumerate(collected, start=1):
                rollout.slot = slot
                rollout.label = f"{cell}{slot}"
            rollouts[cell] = collected
        cache[key] = rollouts
    return cache[key]


def rejected_runs(out: Path) -> list[dict[str, Any]]:
    """Every run this experiment put aside, with the reason it was put aside."""
    rows = []
    for path in sorted(out.glob("runs/*/_rejected/*/REJECTION.json")):
        record = read_json(path)
        record["record"] = rel(path)
        rows.append(record)
    return rows


def attach_intended(rollouts: dict[str, list[Any]], out: Path) -> dict[str, Any]:
    """The intended initial-pose command, taken from the cell that records it."""
    module = handshake_ab()
    for cell in CELLS:
        for rollout in rollouts[cell]:
            try:
                intended = module.client_initial_pose(rollout)
            except SystemExit:
                continue
            for other_cell in CELLS:
                for other in rollouts[other_cell]:
                    other.intended = intended
            return intended
    raise AbError("no client initial-pose message recorded in either cell")


def session_identity(out: Path) -> dict[str, Any]:
    """What each cell actually ran, and whether the two cells differ in anything else."""
    identity: dict[str, Any] = {}
    for cell in CELLS:
        sessions = {}
        for session_name in cell_session_names(cell):
            session = read_json(out / "runs" / session_name / "session.json")
            sessions[session_name] = {
                "tag": session.get("tag"),
                "prompt": session.get("prompt"),
                "requested_model": session.get("requested_model"),
                "requested_model_id": session.get("requested_model_id"),
                "checkpoints": session.get("checkpoints"),
                "checkpoint_step": session.get("checkpoint_step"),
                "rollout_seconds": session.get("rollout_seconds"),
                "settle_seconds": session.get("settle_seconds"),
                "initial_pose": session.get("initial_pose"),
                "scene_profile": session.get("scene_profile"),
                "rollouts": len(session.get("rollouts") or []),
                "command": session.get("command"),
            }
        identity[cell] = sessions
    problems: list[str] = []
    reference = next(iter(identity["A"].values()))
    for cell in CELLS:
        for session_name, entry in identity[cell].items():
            for field in ("prompt", "requested_model_id", "checkpoint_step",
                          "rollout_seconds", "settle_seconds"):
                if entry.get(field) != reference.get(field):
                    problems.append(f"{cell}/{session_name}: {field} differs: "
                                    f"{entry.get(field)!r} vs {reference.get(field)!r}")
            if entry.get("checkpoints") != reference.get("checkpoints"):
                problems.append(f"{cell}/{session_name}: the checkpoints differ")
            if not (entry.get("initial_pose") or {}).get("handshake"):
                problems.append(f"{cell}/{session_name}: the initial-pose handshake is not enabled")
            if cell == "A" and entry.get("scene_profile") is not None:
                problems.append(f"{cell}/{session_name}: names a scene profile; "
                                f"cell A must run the shipped scene")
            if cell == "G":
                variant = (entry.get("scene_profile") or {}).get("file")
                if not variant or VARIANT_PROFILE.split("/")[-1] not in str(variant):
                    problems.append(f"{cell}/{session_name}: does not name the variant profile: {variant!r}")
    return {
        "cells": identity,
        "shipped_profile": {"path": SHIPPED_PROFILE, "sha256": sha256(REPO_ROOT / SHIPPED_PROFILE)},
        "variant_profile": {"path": VARIANT_PROFILE, "sha256": sha256(REPO_ROOT / VARIANT_PROFILE)},
        "identical_except_scene": not problems,
        "problems": problems,
    }


def stage_colour(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    validation = colour_validation(out)
    write_json(out / "colour-validation.json", validation)
    rows = []
    for cell, entry in validation["cells"].items():
        for name, summary in entry["classes"].items():
            rows.append({"cell": cell, "class": name, **{key: value for key, value in summary.items()
                                                         if not isinstance(value, (dict, list))},
                         "median_rgb": summary.get("median_rgb")})
    write_csv(out / "tables" / "colour-validation.csv", rows)
    print(f"[colour] treatment applied: {validation['treatment_applied']} "
          f"{validation['problems'] or ''}")
    for cell in CELLS:
        entry = validation["cells"][cell]
        sampled = sum(session["frames_in_valid_policy_windows"]
                      for session in entry["sessions"].values())
        print(f"[colour] {cell}: third cube ({entry['third_cube_class']}) "
              f"rgb {entry.get('third_cube_median_rgb')} hue {entry.get('third_cube_median_hue')} "
              f"area {entry['third_cube'].get('median_area_px')}px "
              f"frames {entry['third_cube'].get('frames')}/{sampled} "
              f"whole-cube {entry.get('third_cube_whole_cube_rgb')}")
    return validation


def stage_analyse(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    rollouts = cell_rollouts(out)
    intended = attach_intended(rollouts, out)
    identity = session_identity(out)

    validity = [validity_record(rollout) for cell in CELLS for rollout in rollouts[cell]]
    # The hand-off gate is measured against the largest first step the two cells
    # themselves show, so a cell is not judged by a ceiling its own reference
    # cell exceeds: experiment 08's canonical hand-off already did.
    reference_jump = max((row["gates"]["G5_handoff_bounded"]["target_jump_rad"]
                          for row in validity), default=None)
    validity = [validity_record(rollout, baseline_jump=reference_jump)
                for cell in CELLS for rollout in rollouts[cell]]

    metrics = {cell: [rollout_metrics(rollout) for rollout in rollouts[cell]] for cell in CELLS}
    pairs = []
    for index in range(min(len(metrics["A"]), len(metrics["G"]))):
        pairs.append(paired_effect(metrics["A"][index], metrics["G"][index]))

    def values(cell: str, window: str, key: str) -> list[float]:
        return [entry["windows"][window].get(key) for entry in metrics[cell]]

    summaries: dict[str, Any] = {}
    for window in WINDOWS_S:
        for key in ("target_palm_z_mean_m", "acting_target_palm_z_median_m",
                    "measured_palm_z_mean_m", "target_palm_distance_any_cube_min_m",
                    "acting_target_palm_distance_third_min_m",
                    "acting_measured_palm_distance_any_cube_min_m"):
            deltas = [pair.get(f"{window}_{_metric_name(key)}_delta_m") for pair in pairs]
            summaries[f"{window}_{_metric_name(key)}"] = summarise_metric(
                deltas, values("A", window, key))
        for cube in CUBES:
            deltas = [pair.get(f"{window}_target_palm_distance_{cube}_relative_delta") for pair in pairs]
            summaries[f"{window}_target_palm_distance_{cube}_relative"] = summarise_metric(deltas, [])
        support_deltas = [pair.get(f"{window}_token_support_median_delta") for pair in pairs]
        summaries[f"{window}_token_support_median"] = summarise_metric(
            support_deltas, [entry["windows"][window]["groot_token_support"].get("median")
                             for entry in metrics["A"]])

    colour = read_json(out / "colour-validation.json") if (out / "colour-validation.json").is_file() else None
    if colour is None:
        colour = colour_validation(out)
        write_json(out / "colour-validation.json", colour)
    decision = decide(pairs, colour, validity)

    start_states = {cell: [np.asarray(rollout.start_state_43d, dtype=float) for rollout in rollouts[cell]]
                    for cell in CELLS}
    module = warm_ab()
    # The pairing is by slot, so what matters for validity is how far apart the
    # two cells' start states are within each slot, next to each cell's own spread.
    paired_start_states = {}
    for index in range(min(len(start_states["A"]), len(start_states["G"]))):
        paired_start_states[f"A{index + 1}/G{index + 1}"] = {
            "arms_rmse_mrad": handshake_ab().joint_rmse_mrad(
                [start_states["A"][index], start_states["G"][index]], module.ARMS),
            "upper_body_rmse_mrad": handshake_ab().joint_rmse_mrad(
                [start_states["A"][index], start_states["G"][index]], slice(0, 43)),
        }
    start_repeatability = {
        "arms_block": {
            cell: {"joint_rmse_mrad": handshake_ab().joint_rmse_mrad(states, module.ARMS),
                   "per_joint_sd_mrad": handshake_ab().per_joint_sd_mrad(states, module.ARMS)}
            for cell, states in start_states.items()},
        "upper_body_block": {
            cell: {"joint_rmse_mrad": handshake_ab().joint_rmse_mrad(states, slice(0, 43)),
                   "per_joint_sd_mrad": handshake_ab().per_joint_sd_mrad(states, slice(0, 43))}
            for cell, states in start_states.items()},
        "measured_palm_z_at_start_m": {
            cell: [row["start_state"]["measured_palm_z_m"] for row in validity
                   if row["cell"] == cell] for cell in CELLS},
        "start_arms_support": {
            cell: [row["start_state"]["support"]["arms"] for row in validity if row["cell"] == cell]
            for cell in CELLS},
        "paired_start_states": paired_start_states,
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_VERSION,
        "experiment": "09-groot-cube-color-ab",
        "question": ("does reproducing the demonstration-matched green third cube in simulation "
                     "improve GR00T's high target palm behaviour or its approach to the cubes?"),
        "prompt": PROMPT,
        "cells": {cell: CELL_TREATMENT[cell] for cell in CELLS},
        "third_cube_identity": THIRD_CUBE,
        "shipped_third_albedo": list(SHIPPED_THIRD_ALBEDO),
        "variant_third_albedo": list(VARIANT_THIRD_ALBEDO),
        "acceptance": ACCEPTANCE,
        "windows_s": {name: [low, high] for name, (low, high) in WINDOWS_S.items()},
        "session_identity": identity,
        "repeats": {cell: [f"{session}:rollout-{index:02d}" for session, index in CELL_REPEATS[cell]]
                    for cell in CELLS},
        "rejected": rejected_runs(out),
        "intended_token": intended,
        "validity": validity,
        "start_repeatability": start_repeatability,
        "metrics": metrics,
        "pairs": pairs,
        "effect_summaries": summaries,
        "colour_validation": {
            "treatment_applied": colour["treatment_applied"],
            "problems": colour["problems"],
            "third_cube_rgb_delta_G_minus_A": colour.get("third_cube_rgb_delta_G_minus_A"),
            "third_cube_rgb_delta_G_minus_demo": colour.get("third_cube_rgb_delta_G_minus_demo"),
            "control_cubes": colour.get("control_cubes"),
            "scale_controls": colour.get("scale_controls"),
        },
        "decision": decision,
    }
    write_json(out / "summary.json", summary)
    write_csv(out / "tables" / "validity.csv", [
        {"label": row["label"], "cell": row["cell"], "rollout": row["rollout"],
         "integrity_passed": row["gates"]["integrity_passed"],
         "delivery_passed": row["gates"]["G1_delivery"]["passed"],
         "passed": row["gates"]["passed"],
         "onset_s": round(row["onset_seconds_after_start"], 3),
         "initial_pose_messages": row["gates"]["G1_delivery"]["initial_pose_messages"],
         "rate_hz": row["gates"]["G1_delivery"]["rate_hz"],
         "matches_intended": row["gates"]["G1_delivery"]["matches_intended_token"],
         "max_cube_displacement_m": row["gates"]["G4_scene_intact"]["max_cube_displacement_m"],
         "handoff_jump_rad": row["gates"]["G5_handoff_bounded"]["target_jump_rad"],
         "start_arms_mahalanobis": row["start_state"]["support"]["arms"]["mahalanobis"],
         "start_arms_in_p95": row["start_state"]["support"]["arms"]["inside_p95_mahalanobis"],
         } for row in validity])
    metric_rows = []
    for cell in CELLS:
        for entry in metrics[cell]:
            for window, values_ in entry["windows"].items():
                row = {"label": entry["label"], "cell": cell, "rollout": entry["rollout"],
                       "window": window}
                for key, value in values_.items():
                    if isinstance(value, (int, float, bool)) or value is None:
                        row[key] = value
                metric_rows.append(row)
    write_csv(out / "tables" / "window-metrics.csv", metric_rows)
    write_csv(out / "tables" / "paired-effects.csv", pairs)
    print(f"[analyse] decision: {decision['decision']} -- {decision['reason']}")
    return summary


def _metric_name(key: str) -> str:
    return {
        "target_palm_z_mean_m": "target_palm_z_mean",
        "acting_target_palm_z_median_m": "acting_target_palm_z",
        "measured_palm_z_mean_m": "measured_palm_z_mean",
        "target_palm_distance_any_cube_min_m": "target_palm_distance_any_cube",
        "acting_target_palm_distance_third_min_m": "target_palm_distance_third",
        "acting_measured_palm_distance_any_cube_min_m": "measured_palm_distance_any_cube",
    }[key]


def metric_series(rollout: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two figure-2 series: mean target palm z, and the third cube's distance.

    Both are the policy's own output: the commanded palm, not the realised one.
    """
    mask = rollout.relative >= 0.0
    relative = rollout.relative[mask]
    mean_z = 0.5 * (rollout.target_palm["left"][mask][:, 2] + rollout.target_palm["right"][mask][:, 2])
    distances = per_cube_distances(rollout, "left", mask)
    other = per_cube_distances(rollout, "right", mask)
    return relative, mean_z, np.minimum(distances[THIRD_CUBE], other[THIRD_CUBE])


def stage_figures(args: argparse.Namespace) -> list[str]:
    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = read_json(out / "colour-validation.json")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. The treatment, measured in the policy's own view.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = []
    for cell in CELLS:
        entry = colour["cells"][cell]
        rgb = entry["classes"][entry["third_cube_class"]]["median_rgb"]
        colours.append(np.clip(np.array(rgb, dtype=float) / 255.0, 0.0, 1.0))
    axes[0].bar(["A (blue third cube)", "G (green third cube)"], [1.0, 1.0], color=colours)
    axes[1].bar(["A", "G"], [colour["cells"][cell]["classes"][colour["cells"][cell]["third_cube_class"]]
                             ["median_hue"] for cell in CELLS], color=colours)
    axes[1].axhline(colour["demo_third_cube_hue"], color="black", linestyle="--", linewidth=1)
    axes[1].annotate(f"demonstration hue {colour['demo_third_cube_hue']:.1f}",
                     (0.02, colour["demo_third_cube_hue"] + 1.5), fontsize=8)
    for index, cell in enumerate(CELLS):
        entry = colour["cells"][cell]
        published = entry["classes"][entry["third_cube_class"]]
        rgb = entry["third_cube_median_rgb"]
        delta = [round(rgb[channel] - DEMO_THIRD_SRGB[channel]) for channel in range(3)]
        delta_text = ", ".join(f"{value:+d}" for value in delta)
        axes[0].annotate(f"rgb {[round(value) for value in published['median_rgb']]}\n"
                         f"hue {published['median_hue']:.1f}  {published['median_area_px']:.0f} px\n"
                         f"vs demonstration {delta_text}",
                         (index, 0.5), ha="center", va="center", fontsize=8, color="white")
        axes[1].annotate(f"hue {published['median_hue']:.1f}\n"
                         f"off demo by {published['median_hue'] - DEMO_THIRD_HUE:+.1f}",
                         (index, 6), ha="center", fontsize=8)
    axes[0].set_title("third cube, median blob colour in the policy head camera")
    axes[1].set_title("third cube hue against the demonstration")
    axes[1].set_ylim(0, 140)
    figure.tight_layout()
    path = figures / "figure-1-treatment-colour.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(rel(path))

    # 2. The primary metrics over the policy window.
    rollouts = cell_rollouts(out)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    grid = np.linspace(0.0, float(read_json(run_dir("A", out) / "session.json")["rollout_seconds"]), 300)
    for cell, color in (("A", "#1f77b4"), ("G", "#2ca02c")):
        resampled = []
        for rollout in rollouts[cell]:
            relative, mean_z, third = metric_series(rollout)
            axes[0].plot(relative, mean_z, color=color, alpha=0.4, linewidth=0.8)
            axes[1].plot(relative, third, color=color, alpha=0.4, linewidth=0.8)
            resampled.append((np.interp(grid, relative, mean_z), np.interp(grid, relative, third)))
        axes[0].plot(grid, np.median([entry[0] for entry in resampled], axis=0), color=color,
                     linewidth=2, label=f"{cell}: {CELL_TREATMENT[cell]}")
        axes[1].plot(grid, np.median([entry[1] for entry in resampled], axis=0), color=color,
                     linewidth=2, label=f"{cell}: {CELL_TREATMENT[cell]}")
    axes[0].set_ylabel("target palm z (mean of sides) [m]")
    axes[1].set_ylabel(f"target palm - {THIRD_CUBE} cube min distance [m]")
    axes[1].set_xlabel("seconds after Start")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.axvspan(0.0, 1.0, color="grey", alpha=0.12)
        axis.axvspan(1.0, 5.0, color="grey", alpha=0.06)
        axis.legend(loc="best", fontsize=9)
    axes[0].set_title("policy output: target palm height and approach (thin = single repeats, thick = median)")
    figure.tight_layout()
    path = figures / "figure-2-policy-metrics.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(rel(path))

    # 3. The head camera itself, both cells at matched times after Start.
    montage_rows = []
    for cell in CELLS:
        rollout = cell_rollouts(out)[cell][0]
        frames = [row for row in frame_rows(rollout.session)
                  if rollout.reset_wall - 1.0 <= row["wall_time"] <= rollout.stop_wall]
        onset = rollout.onset_wall
        picks = []
        for offset in (0.0, 5.0, 15.0, 30.0):
            target = onset + offset
            picks.append(min(frames, key=lambda row: abs(row["wall_time"] - target)))
        montage_rows.append((cell, picks))
    tiles = []
    for cell, picks in montage_rows:
        row = []
        for frame in picks:
            image = cv2.imread(str(frame["path"]))
            if image is None:
                continue
            image = cv2.resize(image, (480, 360))
            treatment = "blue third cube" if cell == "A" else "green third cube"
            label = f"{cell} ({treatment})  t+{int(frame['wall_time'] - picks[0]['wall_time'])}s"
            cv2.putText(image, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            row.append(image)
        if row:
            tiles.append(np.hstack(row))
    if tiles:
        width = max(tile.shape[1] for tile in tiles)
        padded = [np.pad(tile, ((0, 0), (0, width - tile.shape[1]), (0, 0)), constant_values=32)
                  for tile in tiles]
        montage = np.vstack(padded)
        path = figures / "figure-3-head-camera-montage.jpg"
        cv2.imwrite(str(path), montage)
        written.append(rel(path))
    print(f"[figures] wrote {len(written)} figures")
    return written


def stage_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    artifacts: dict[str, str | None] = {}
    for pattern in ("REPORT.md", "summary.json", "colour-validation.json", "tables/*.csv", "figures/*"):
        for path in sorted(out.glob(pattern)):
            artifacts[rel(path)] = sha256(path)
    for cell in CELLS:
        for session_name in cell_session_names(cell):
            run = out / "runs" / session_name
            for name in ("session.json", "verification.json", "manifest.json"):
                artifacts[rel(run / name)] = sha256(run / name)
            for rollout in sorted((run / "rollouts").glob("rollout-*")):
                for name in ("rollout.json", "tracking.jsonl", "video.mp4", "contact-sheet.jpg"):
                    artifacts[rel(rollout / name)] = sha256(rollout / name)
            for rejected in sorted((run / "_rejected").glob("*")):
                for name in ("REJECTION.json", "rollout.json", "contact-sheet.jpg"):
                    artifacts[rel(rejected / name)] = sha256(rejected / name)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_VERSION,
        "experiment": "09-groot-cube-color-ab",
        "cells": {cell: CELL_TREATMENT[cell] for cell in CELLS},
        "repeats": {cell: [f"{session}:rollout-{index:02d}" for session, index in CELL_REPEATS[cell]]
                    for cell in CELLS},
        "rejected": rejected_runs(out),
        "scene_profiles": {
            "shipped": {"path": SHIPPED_PROFILE, "sha256": sha256(REPO_ROOT / SHIPPED_PROFILE)},
            "variant": {"path": VARIANT_PROFILE, "sha256": sha256(REPO_ROOT / VARIANT_PROFILE)},
        },
        "script_sha256": sha256(REPO_ROOT / "scripts" / "groot-cube-color-ab.py"),
        "artifacts": artifacts,
    }
    write_json(out / "manifest.json", manifest)
    print(f"[manifest] {len(artifacts)} artifacts hashed")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T third-cube colour A/B (experiment 09)")
    parser.add_argument("stage", choices=("colour", "analyse", "figures", "manifest", "all"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("colour", "all"):
        stage_colour(args)
    if args.stage in ("analyse", "all"):
        stage_analyse(args)
    if args.stage in ("figures", "all"):
        stage_figures(args)
    if args.stage in ("manifest", "all"):
        stage_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
