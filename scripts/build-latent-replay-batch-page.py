#!/usr/bin/env python3
"""One page for a simulated SONIC-latent replay batch.

Each card pairs the recorded head camera with the same episode replayed in Isaac
from the stored latent, plus the A/B/C fidelity numbers that gate the replay.
The page only reports; it never recomputes or repairs a result.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_review_module():
    """Reuse the review page's stylesheet and grouped-video script verbatim."""
    spec = importlib.util.spec_from_file_location("sonic_pilot_review_page",
                                                  ROOT / "scripts/build-sonic-pilot-review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def fmt(value, places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def badge(value: str) -> str:
    kind = {"PASS": "pass", "FAIL": "fail"}.get(value, "unknown")
    return f"<b class='badge {kind}'>{html.escape(value)}</b>"


def card(run_dir: Path, *, page_dir: Path, sync_group: str) -> dict:
    manifest = read_json(run_dir / "run_manifest.json")
    fidelity = read_json(run_dir / "sonic_fidelity.json")
    prefix = run_dir.relative_to(page_dir).as_posix()
    source = (run_dir / "source.mp4").is_file()
    replay = (run_dir / "sonic_latent_motion.mp4").is_file()
    pairs = fidelity.get("pairs") or {}
    coverage = fidelity.get("coverage") or {}
    transport = fidelity.get("transport") or {}
    wrist = (fidelity.get("left_wrist") or {}).get("reference_to_response") or {}
    result = str(fidelity.get("result", "UNVERIFIED"))

    facts = [
        ("kare", f"{manifest.get('frames', '—')} (kaynak {fmt(manifest.get('source_fps'), 0)} Hz → {fmt(manifest.get('processed_fps'), 0)} Hz)"),
        ("süre", f"{fmt(manifest.get('duration_s'), 1)} s"),
        ("A→B reference → SONIC komutu (body MAE)", f"{fmt((pairs.get('reference_to_command') or {}).get('body_mae_rad'))} rad"),
        ("A→C reference → robot (body MAE)", f"{fmt((pairs.get('reference_to_response') or {}).get('body_mae_rad'))} rad"),
        ("B→C komut → robot (body MAE)", f"{fmt((pairs.get('command_to_response') or {}).get('body_mae_rad'))} rad"),
        ("A→C sol bilek yörünge MAE", f"{fmt(wrist.get('path_mae_m'))} m"),
        ("tam hareket gözlendi", str(coverage.get("full_motion", "—"))),
        ("son gözlenen kare", f"{coverage.get('last_observed_frame', '—')}/{coverage.get('expected_frames', '—')}"),
        ("SONIC'e ulaşan tekil kare", f"{transport.get('received_unique_frames', '—')}/{coverage.get('expected_frames', '—')}"),
    ]

    def video(name_en: str, title: str, note: str, present: bool) -> str:
        if not present:
            return (f"<figure><div class='missing-box'>{html.escape(title)}: video yok</div>"
                    f"<figcaption>{html.escape(note)}</figcaption></figure>")
        return (
            "<figure>"
            f"<video controls preload='metadata' playsinline data-sync-group='{html.escape(sync_group, quote=True)}' "
            f"src='{html.escape(prefix)}/{html.escape(name_en)}'></video>"
            f"<figcaption><strong>{html.escape(title)}</strong> · {html.escape(note)}</figcaption></figure>"
        )

    return {
        "result": result,
        "collection": manifest.get("dataset_name", run_dir.parent.name),
        "episode_index": manifest.get("episode_index"),
        "pilot": manifest.get("pilot", run_dir.name),
        "frames": manifest.get("frames", 0),
        "paragraph": pairs.get("reference_to_response", {}).get("body_mae_rad"),
        "html": f"""<section class='card' id='{html.escape(str(manifest.get('pilot', run_dir.name)))}'>
<h2>{html.escape(str(manifest.get('dataset_name', '—')))} · ep {manifest.get('episode_index', '—')} {badge(result)}</h2>
<p class='note'>A = kanonik 50 Hz referans, B = SONIC'in eklem hedefi, C = simülasyondaki ölçülen robot.
Replay, kayıtta saklanan latent ile sürüldü; hiçbir token yeniden üretilmedi.</p>
<div class='syncer' data-sync-controls='{html.escape(sync_group, quote=True)}'>
<button type='button' class='sync-toggle' aria-pressed='false'>▶︎ Oynat</button>
<input type='range' class='sync-seek' min='0' max='1' step='0.01' value='0' aria-label='Ortak zaman çizelgesi (saniye)'>
<span class='sync-clock'>0.00 / 0.00 s</span>
<span class='sync-hint' role='status'></span></div>
<div class='pair'>
{video('source.mp4', '1 · Kayıtlı baş kamerası', 'ham episode, kayıt', source)}
{video('sonic_latent_motion.mp4', '2 · SONIC latent → simülasyon (dış kamera)', 'latent ile sürülen robot', replay)}
</div>
<table>{''.join(f'<tr><th>{html.escape(a)}</th><td>{html.escape(b)}</td></tr>' for a, b in facts)}</table>
<p class='links'><a href='{html.escape(prefix)}/review.html'>tam kanıt sayfası</a>
 · <a href='{html.escape(prefix)}/source.mp4'>source.mp4</a>
 · <a href='{html.escape(prefix)}/sonic_latent_motion.mp4'>sonic_latent_motion.mp4</a>
 · <a href='{html.escape(prefix)}/sonic_fidelity.json'>sonic_fidelity.json</a></p>
</section>""",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="SONIC latent → simülasyon toplu kontrol")
    args = parser.parse_args()

    batch_root = args.batch_root.resolve()
    output = (args.output or batch_root / "index.html").resolve()
    batch = read_json(batch_root / "batch_manifest.json")
    run_dirs = sorted(path for path in batch_root.glob("*/episode_*") if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"no episode directories under {batch_root}")

    items = [card(run_dir, page_dir=output.parent, sync_group=f"ep-{run_dir.parent.name}-{run_dir.name}")
             for run_dir in run_dirs]
    passed = sum(1 for item in items if item["result"] == "PASS")
    failed = sum(1 for item in items if item["result"] == "FAIL")
    unverified = len(items) - passed - failed
    total_frames = sum(int(item["frames"] or 0) for item in items)

    by_collection: dict[str, list[dict]] = {}
    for item in items:
        by_collection.setdefault(str(item["collection"]), []).append(item)
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{len(group)}</td>"
        f"<td>{sum(1 for item in group if item['result'] == 'PASS')}</td>"
        f"<td>{sum(1 for item in group if item['result'] != 'PASS')}</td></tr>"
        for name, group in sorted(by_collection.items())
    )

    review = load_review_module()
    sampling = ""
    if batch:
        sampling = (f"örnekleme: {batch.get('per_collection')} episode/koleksiyon, "
                    f"maks {batch.get('max_seconds')} s, seed <code>{html.escape(str(batch.get('seed')))}</code>, "
                    f"toplam {batch.get('total_frames')} kare")
    sections = ""
    for name, group in sorted(by_collection.items()):
        sections += f"<h2 class='collection'>{html.escape(name)}</h2>" + "".join(
            sorted(group, key=lambda item: item["episode_index"] or 0) and
            [item["html"] for item in sorted(group, key=lambda item: item["episode_index"] or 0)]
        )

    page = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>{review.STYLE}
.pair{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}}
.pair figure{{flex:1 1 380px;margin:0}}
.pair video{{width:100%;background:#000;border-radius:6px}}
figcaption{{color:#9aa2b1;font-size:12px;margin-top:6px}}
.card h2{{margin-bottom:2px}}
h2.collection{{margin:34px 0 6px;border-bottom:1px solid #262a32;padding-bottom:4px}}
.links{{font-size:12px}}
</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p class='sub'>{len(items)} episode · {total_frames} kare · {total_frames / 50 / 60:.1f} dakika hareket ·
 PASS {passed} · FAIL {failed}{f" · UNVERIFIED {unverified}" if unverified else ""}</p>
<p class='sub'>Her kartta solda kayıtlı baş kamerası, sağda aynı episode'un latent ile sürülen simülasyonu.
Kareler <code>unitree-sonic-v1.1-78d</code> korpusundaki 78D aksiyondan dilimlendi
(64D gövde token'ı + 7+7 Dex3 el); {sampling}.
{''.join(f"<br>konverter <code>{html.escape(str(item))}</code>" for item in [ (batch.get('runs') or [{}])[0].get('action_npz_sha256', '')[:0] ] if False)}
</p>
<section><h2>Koleksiyon özeti</h2>
<table><tr><th>koleksiyon</th><th>episode</th><th>PASS</th><th>PASS olmayan</th></tr>{rows}</table></section>
{sections}
<script>{review.SCRIPT}</script>
</body></html>
"""
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"html": str(output), "episodes": len(items), "pass": passed,
                      "fail": failed, "unverified": unverified}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
