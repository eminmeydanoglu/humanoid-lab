"""Pull grounding coordinates out of a model answer and map them onto image pixels.

ER-1 answers grounding prompts with coordinate lists such as ``[(255, 415)]``. The
numbers are not reliably in the frame's own pixel grid: on the 640x480 G1 frames we
have seen x up to 848 and y up to 867, which no pixel of a 640x480 image can have.
So the raw numbers are always kept, and the pixel mapping is an explicit choice
(``auto`` by default) that the caller can override from the UI.
"""

import re

SCALE_MODES = ("auto", "raw", "norm1000", "half")

# <|box_start|>(x1,y1),(x2,y2)<|box_end|> — Qwen-style box tags, if the model ever emits them.
BOX_TAG_RE = re.compile(r"<\|box_start\|>([^<]*?)<\|box_end\|>")
# A bracketed or parenthesised run of 2 or 4 numbers: [(255, 415)], (255,415), [255, 415, 300, 470]
GROUP_RE = re.compile(r"[\[(]\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){1,3})\s*[\])]")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text):
    return [float(n) for n in NUM_RE.findall(text)]


def _map_pair(values, w, h, mode):
    out = []
    for i, v in enumerate(values):
        dim = w if i % 2 == 0 else h
        if mode == "raw":
            out.append(v)
        elif mode == "half":
            out.append(v / 2.0)
        else:
            out.append(v / 1000.0 * dim)
    return out


def _pick_mode(raw_groups, w, h, requested):
    if requested != "auto":
        return requested
    xs = [g[0::2] for g in raw_groups]
    ys = [g[1::2] for g in raw_groups]
    flat_x = [v for g in xs for v in g]
    flat_y = [v for g in ys for v in g]
    negative = any(v < 0 for v in flat_x + flat_y)
    inside = all(v <= w * 1.02 for v in flat_x) and all(v <= h * 1.02 for v in flat_y)
    if inside and not negative:
        return "raw"
    if all(0 <= v <= 1000 for v in flat_x + flat_y):
        return "norm1000"
    return "half"


def extract(answer, width, height, mode="auto"):
    """Return markers for every coordinate group found in ``answer``.

    Each marker carries the numbers exactly as printed (``raw``) and the mapped
    pixel position (``px``), plus whether it landed inside the frame.
    """
    if mode not in SCALE_MODES:
        raise ValueError(f"unknown scale mode {mode!r}")

    boxes = []
    rest = answer
    for match in BOX_TAG_RE.finditer(answer):
        nums = _numbers(match.group(1))
        if len(nums) >= 4:
            boxes.append(nums[:4])
    high = boxes if boxes else None
    if high:
        rest = BOX_TAG_RE.sub(" ", answer)

    points = []
    for match in GROUP_RE.finditer(rest):
        nums = _numbers(match.group(1))
        if len(nums) == 2:
            points.append(nums)
        elif len(nums) == 4:
            boxes.append(nums)

    raw_groups = points + boxes
    if not raw_groups:
        return {"mode": None, "markers": [], "warnings": [], "n_raw": 0}

    used = _pick_mode(raw_groups[: len(points)] or raw_groups, width, height, mode)
    markers = []
    n_point = len(points)
    for idx, raw in enumerate(raw_groups):
        px = _map_pair(raw, width, height, used)
        xs, ys = px[0::2], px[1::2]
        in_bounds = all(0 <= v <= width for v in xs) and all(0 <= v <= height for v in ys)
        label = f"P{idx + 1}" if idx < n_point else f"B{idx - n_point + 1}"
        markers.append(
            {
                "kind": "point" if idx < n_point else "box",
                "label": label,
                "raw": [round(v, 1) for v in raw],
                "px": [round(v, 1) for v in px],
                "in_bounds": in_bounds,
            }
        )

    warnings = []
    outside = [m["label"] for m in markers if not m["in_bounds"]]
    if outside:
        warnings.append(
            f"{len(outside)} işaret karenin dışına düştü ({', '.join(outside)}) — ölçek kipi yanlış olabilir."
        )
    if used != "raw":
        warnings.append(f"koordinatlar '{used}' kipine göre piksele çevrildi (ham sayılar korundu).")
    return {"mode": used, "markers": markers, "warnings": warnings, "n_raw": len(raw_groups)}
