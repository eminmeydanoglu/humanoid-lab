#!/usr/bin/env python3
"""Build self-contained HTML previews for the G1 Dex3 sample frames.

Consumes the manifest written by g1-dex3-sample-frames.py, reads the matching
episode data rows from the local LeRobot parquet mirror, and renders one HTML
page per dataset plus an index page. Camera frames are embedded as base64 so a
page can be opened on its own.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

STYLE = """body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:20px} h2{font-size:16px;color:#444}
section.ep{border:1px solid #ddd;border-radius:8px;padding:12px;margin:16px 0;background:#fff}
.camrow{display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start}
figure{margin:6px} figcaption{font-size:11px;color:#555}
img.frame{width:320px;height:240px;object-fit:cover;border:1px solid #ccc;border-radius:4px}
.spark{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px;margin-top:8px}
.spark div{font-size:11px;color:#555}
table{border-collapse:collapse;font-size:13px} td,th{border:1px solid #ddd;padding:4px 8px;text-align:left}
.tblwrap{overflow-x:auto}
.card{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;background:#fff;display:flex;gap:14px}
.card img{width:200px;height:150px;object-fit:cover;border:1px solid #ccc;border-radius:4px}
a{color:#1a73e8;text-decoration:none} a:hover{text-decoration:underline}"""


def b64_jpeg(path: Path, max_width: int | None, quality: int) -> str:
    image = Image.open(path).convert("RGB")
    if max_width and image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sparkline(values, label: str, color: str) -> str:
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    width = 220.0
    points = " ".join(
        f"{i * width / max(1, len(values) - 1):.1f},{42 - (v - lo) / span * 42:.1f}" for i, v in enumerate(values)
    )
    return (
        f'<div class="sparkitem">{html.escape(label)}'
        f'<svg width="220" height="44" viewBox="0 0 220 44" '
        f'style="background:#f6f6f6;border:1px solid #ddd;display:block"><polyline points="{points}" fill="none" '
        f'stroke="{color}" stroke-width="1.4"/></svg></div>'
    )


def load_episode_data(dataset_dir: Path, episode: int, cache: dict | None = None):
    key = str(dataset_dir)
    tables = (cache or {}).get(key)
    if tables is None:
        tables = [
            pq.read_table(
                path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            )
            for path in sorted(dataset_dir.glob("data/chunk-*/file-*.parquet"))
        ]
        if cache is not None:
            cache[key] = tables
    for table in tables:
        rows = table.filter(pc.equal(table["episode_index"], episode))
        if rows.num_rows:
            return rows.to_pydict()
    return None


def feature_names(info: dict) -> dict[str, list[str]]:
    names = {}
    for key in ("observation.state", "action"):
        raw = info.get("features", {}).get(key, {}).get("names")
        if raw and isinstance(raw[0], list):
            names[key] = list(raw[0])
        elif raw:
            names[key] = list(raw)
    return names


def episode_sparklines(rows, names: dict[str, list[str]] | None = None) -> str:
    parts = []
    for key, color in (("observation.state", "#1a73e8"), ("action", "#e8710a")):
        series = rows[key]
        if not series or series[0] is None:
            continue
        dims = len(series[0])
        labels = (names or {}).get(key) or [f"{key}[{i}]" for i in range(dims)]
        for dim in range(dims):
            values = [float(row[dim]) for row in series]
            label = labels[dim] if dim < len(labels) else f"{key}[{dim}]"
            parts.append(sparkline(values, label, color))
    return f'<details style="margin-top:10px"><summary style="cursor:pointer;font-size:13px">state/action signal preview</summary><div class="spark">{"".join(parts)}</div></details>'


def dataset_page(slug: str, frames: list[dict], fps: float, frames_root: Path, datasets_root: Path) -> str:
    by_episode: dict[int, dict[str, list[dict]]] = {}
    for frame in frames:
        by_episode.setdefault(frame["episode"], {}).setdefault(frame["cam"], []).append(frame)
    dataset_name = frames[0]["dataset"]
    dataset_dir = datasets_root / dataset_name
    data_cache: dict = {}
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    task_list = sorted({f["task"] for f in frames})
    body = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Dataset preview · {html.escape(slug)}</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<p><a href=\"index.html\">← all datasets</a></p>",
        f"<h1>Dataset: {html.escape(slug)}</h1>",
        (
            f'<p style="font-size:13px;color:#555">HF: {html.escape(frames[0]["repo"])} · '
            f"{info['total_episodes']} episodes · {info['total_frames']} frames @ {fps:g} fps · "
            f"cameras: {html.escape(', '.join(sorted({f['cam'] for f in frames})))}<br>"
            f"task label(s) in meta: {html.escape('; '.join(task_list))}</p>"
        ),
    ]
    for episode in sorted(by_episode):
        any_frame = next(iter(by_episode[episode].values()))[0]
        seconds = any_frame["episode_to"] - any_frame["episode_from"]
        body.append('<section class="ep">')
        body.append(
            f"<h3>Episode {episode:06d} · {any_frame['length']} frames @ {fps:g} fps · {seconds:.1f}s</h3>"
        )
        body.append(f"<p><b>Task:</b> {html.escape(any_frame['task'])}</p>")
        for cam in sorted(by_episode[episode]):
            body.append('<div class="camrow">')
            for frame in sorted(by_episode[episode][cam], key=lambda f: f["frame_slot"]):
                encoded = b64_jpeg(Path(frame["out"]), max_width=640, quality=78)
                caption = f"t={frame['timestamp'] - frame['episode_from']:.2f}s of {frame['episode_to'] - frame['episode_from']:.1f}s"
                body.append(
                    f'<figure><img class="frame" src="data:image/jpeg;base64,{encoded}"/>'
                    f"<figcaption>{html.escape(cam)}<br/>{caption}</figcaption></figure>"
                )
            body.append("</div>")
        rows = load_episode_data(dataset_dir, episode, data_cache)
        if rows:
            body.append(episode_sparklines(rows, feature_names(info)))
        body.append("</section>")
    body.append("</body></html>")
    return "\n".join(body)


def index_page(pages: list[dict]) -> str:
    body = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Unitree G1 Dex3 · dataset previews</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>Unitree G1 Dex3 datasets (mindchain collection)</h1>",
        '<p style="font-size:13px;color:#555">Sample camera frames pulled from the Hugging Face repos; one page per dataset. '
        "Task labels below are copied from each dataset's meta and are unreliable in this collection "
        '(seven pick-* datasets share "Pick up the red cup on the table.", GraspSquare is labelled "camera packaging"); '
        "the dataset name is the trustworthy task identifier.</p>",
        "<div class='tblwrap'><table><tr><th>dataset</th><th>episodes</th><th>frames</th><th>fps</th><th>cameras</th><th>task label</th><th>page</th></tr>",
    ]
    for page in pages:
        body.append(
            f"<tr><td>{html.escape(page['slug'])}</td><td>{page['episodes']}</td><td>{page['frames_total']}</td>"
            f"<td>{page['fps']:g}</td><td>{html.escape(', '.join(page['cams']))}</td>"
            f"<td>{html.escape(page['task'])}</td><td><a href=\"{page['slug']}.html\">open</a></td></tr>"
        )
    body.append("</table></div>")
    for page in pages:
        thumbs = "".join(
            f'<img src="data:image/jpeg;base64,{b64_jpeg(Path(path), max_width=240, quality=70)}"/>'
            for path in page["thumbs"]
        )
        body.append(
            f'<div class="card">{thumbs}<div><h2 style="margin:0 0 6px"><a href="{page["slug"]}.html">'
            f"{html.escape(page['slug'])}</a></h2>"
            f'<p style="font-size:13px;color:#555;margin:0">{html.escape(page["repo"])}<br/>'
            f"{page['episodes']} episodes · {len(page['cams'])} cameras · task: "
            f"{html.escape(page['task'])}</p></div></div>"
        )
    body.append("</body></html>")
    return "\n".join(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/outputs/dataset-previews/g1-dex3/manifest.json")
    parser.add_argument("--datasets-root", default="data/datasets/first_tur_ham/unitree-g1-dex3")
    parser.add_argument("--out-root", default="data/outputs/dataset-previews/g1-dex3")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.out_root)
    datasets_root = Path(args.datasets_root)
    frames_root = Path(manifest["frames_root"])

    by_slug: dict[str, list[dict]] = {}
    for frame in manifest["frames"]:
        by_slug.setdefault(frame["slug"], []).append(frame)

    pages = []
    for slug, frames in sorted(by_slug.items()):
        fps = frames[0]["fps"]
        page_html = dataset_page(slug, frames, fps, frames_root, datasets_root)
        (out_root / f"{slug}.html").write_text(page_html)
        thumbs = [f["out"] for f in sorted(frames, key=lambda f: (f["episode"], f["frame_slot"]))[:2]]
        info = json.loads((datasets_root / frames[0]["dataset"] / "meta/info.json").read_text())
        pages.append(
            {
                "slug": slug,
                "repo": frames[0]["repo"],
                "episodes": info["total_episodes"],
                "frames_total": info["total_frames"],
                "fps": fps,
                "cams": sorted({f["cam"] for f in frames}),
                "task": sorted({f["task"] for f in frames})[0],
                "thumbs": thumbs,
            }
        )
        print(f"wrote {out_root / (slug + '.html')} ({len(frames)} frames)")

    (out_root / "index.html").write_text(index_page(pages))
    print(f"wrote {out_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
