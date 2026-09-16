#!/usr/bin/env python3
"""Write a minimal per-episode review page: three videos, nothing else.

Deliberately much plainer than ``build-sonic-pilot-review.py``.  For one dataset
episode it shows exactly the three artifacts that answer "is this conversion
right?":

1. the recorded head/ego camera clip (what the human actually did),
2. the frame-exact kinematic replay (the canonical 50 Hz reference written
   straight into the simulator, no controller in the loop),
3. the free SONIC run driven by the persisted offline latent tokens.

Each section carries the numbers that justify it, read from the artifacts on
disk rather than restated by hand, so the page cannot drift from the run it
describes.  Videos use ``preload="metadata"``; serve the directory with
``scripts/serve-sonic-review.py`` so the timeline can seek.

Usage:
  write-sonic-pilot-page.py --pilot-dir DIR --output FILE [--title TEXT]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def seconds(path: Path) -> float | None:
    """Duration of a media file, via ffprobe; None when unavailable."""
    import subprocess

    if not path.is_file():
        return None
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{html.escape(text)}</span>'


def video_block(
    label: str,
    path: Path,
    *,
    note: str,
    facts: list[tuple[str, str]],
    relative_to: Path,
) -> str:
    """One titled video with its measured facts, or a clear 'missing' notice."""
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>" for key, value in facts
    )
    if not path.is_file():
        body = f'<p class="missing">not produced: <code>{html.escape(path.name)}</code></p>'
    else:
        href = path.relative_to(relative_to).as_posix() if path.is_relative_to(relative_to) else path.as_posix()
        duration = seconds(path)
        meta = f"{duration:.2f} s" if duration is not None else "unknown duration"
        body = (
            f'<video controls preload="metadata" playsinline src="{html.escape(href)}"></video>'
            f'<p class="file"><code>{html.escape(path.name)}</code> · {meta}</p>'
        )
    return (
        f'<section><h2>{html.escape(label)}</h2><p class="note">{html.escape(note)}</p>'
        f"{body}<table>{rows}</table></section>"
    )


def build_page(pilot_dir: Path, *, title: str | None = None) -> str:
    manifest = load_json(pilot_dir / "run_manifest.json")
    encoder = load_json(pilot_dir / "encoder_manifest.json")
    kinematic = load_json(pilot_dir / "completed_reference_kinematic_metrics.json")
    sonic = load_json(pilot_dir / "sonic_latent_metrics.json")
    fidelity = load_json(pilot_dir / "sonic_fidelity.json")
    qc = manifest.get("qc") or {}
    pilot_name = str(manifest.get("pilot", pilot_dir.parent.name))
    page_title = title or f"SONIC pilot — {pilot_name}"

    verdict = str(qc.get("episode_result", "UNKNOWN"))
    verdict_kind = {"PASS": "pass", "FAIL": "fail"}.get(verdict, "unknown")
    action = (manifest.get("final_action_78d") or {}).get("status", "unknown")
    hand_schema = str(manifest.get("hand_schema_status", "unknown"))

    decisions = "".join(
        "<tr><th>{}</th><td class=\"{}\">{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(item.get("metric"))),
            "ok" if item.get("result") == "PASS" else "bad",
            html.escape(str(item.get("result"))),
            fmt(item.get("value")),
            fmt(item.get("limit")),
        )
        for item in qc.get("decisions", [])
    )

    k = kinematic.get("kinematic") or {}
    write_error = k.get("write_error_rad") or {}
    controller = sonic.get("controller") or {}
    tracking = controller.get("body_tracking_error_rad") or {}
    pairs = fidelity.get("pairs") or {}
    coverage = fidelity.get("coverage") or {}
    transport = fidelity.get("transport") or {}
    wrist = fidelity.get("left_wrist") or {}
    world_wrist = fidelity.get("world_left_wrist") or {}

    # 1. recorded camera
    source_facts = [
        ("frames", fmt(manifest.get("frames"), 0)),
        ("source fps", fmt(manifest.get("source_fps"), 1)),
        ("processed fps", fmt(manifest.get("processed_fps"), 1)),
        ("duration", f"{fmt(manifest.get('duration_s'), 2)} s"),
        ("camera", str((manifest.get("source_video") or {}).get("camera", "—"))),
    ]

    # 2. kinematic replay: zero write error is the whole point of this video
    kinematic_facts = [
        ("frames replayed", fmt(k.get("frames_replayed"), 0)),
        ("physics ticks / frame", fmt(k.get("physics_ticks_per_frame"), 0)),
        ("body write error (max)", f"{fmt(write_error.get('body_max'), 8)} rad"),
        ("hand write error (max)", f"{fmt(write_error.get('left_hand_max'), 8)} rad"),
        ("root write error (max)", f"{fmt((k.get('root_write_error') or {}).get('position_max_m'), 8)} m"),
        ("root source", str(k.get("root_source", "—"))),
        ("hands applied", str(k.get("hands_applied", "—"))),
    ]

    # 3. free SONIC run from the persisted latent
    sonic_facts = [
        ("A→B reference to SONIC body MAE", f"{fmt((pairs.get('reference_to_command') or {}).get('body_mae_rad'))} rad"),
        ("A→C reference to robot body MAE", f"{fmt((pairs.get('reference_to_response') or {}).get('body_mae_rad'))} rad"),
        ("B→C SONIC to robot body MAE", f"{fmt((pairs.get('command_to_response') or {}).get('body_mae_rad'))} rad"),
        ("A→B left arm MAE", f"{fmt((pairs.get('reference_to_command') or {}).get('left_arm_mae_rad'))} rad"),
        ("A→C left arm MAE", f"{fmt((pairs.get('reference_to_response') or {}).get('left_arm_mae_rad'))} rad"),
        ("A→B left wrist path MAE", f"{fmt((wrist.get('reference_to_command') or {}).get('path_mae_m'))} m"),
        ("A→C left wrist path MAE", f"{fmt((wrist.get('reference_to_response') or {}).get('path_mae_m'))} m"),
        ("A / B left wrist lateral move", f"{fmt((wrist.get('reference_to_command') or {}).get('reference_delta_y_m'))} / {fmt((wrist.get('reference_to_command') or {}).get('other_delta_y_m'))} m"),
        ("A→C world left wrist path MAE", f"{fmt((world_wrist.get('reference_to_response') or {}).get('path_mae_m'))} m"),
        ("A / C world left wrist lateral move", f"{fmt((world_wrist.get('reference_to_response') or {}).get('reference_delta_y_m'))} / {fmt((world_wrist.get('reference_to_response') or {}).get('other_delta_y_m'))} m"),
        ("full motion observed", str(coverage.get("full_motion", "UNVERIFIED"))),
        ("last observed token frame", f"{fmt(coverage.get('last_observed_frame'), 0)}/{fmt(coverage.get('expected_frames'), 0)}"),
        ("SONIC received unique frames", f"{fmt(transport.get('received_unique_frames'), 0)}/{fmt(transport.get('expected_frames'), 0)}"),
        ("fidelity gate", str(fidelity.get("result", "UNVERIFIED"))),
        ("result", str(sonic.get("result", "—"))),
        ("video frames", fmt((sonic.get("video") or {}).get("frames"), 0)),
        ("body tracking MAE", f"{fmt(tracking.get('mean'))} rad"),
        ("body tracking max", f"{fmt(tracking.get('max'))} rad"),
        ("control mode", str(controller.get("control_mode", "—"))),
        ("hand binding", json.dumps(controller.get("hand_binding", {})) if controller.get("hand_binding") else "—"),
        ("real time factor", fmt(sonic.get("real_time_factor"), 3)),
    ]

    encoder_facts = [
        ("encoder sha256", str((encoder.get("encoder") or {}).get("sha256", "—"))[:16] + "…"),
        ("motion token dim", fmt(encoder.get("motion_token_dim"), 0)),
        ("action dim", fmt(encoder.get("action_dim"), 0)),
        ("token norm p50", fmt((encoder.get("token_norm") or {}).get("p50"))),
        ("orientation policy", str(encoder.get("encoder_orientation_policy", "—"))),
    ]

    sections = [
        video_block(
            "1 · Recorded head camera",
            pilot_dir / "source.mp4",
            note="The raw source episode as recorded. This is the ground truth the conversion must preserve.",
            facts=source_facts,
            relative_to=pilot_dir,
        ),
        video_block(
            "2 · Whole-body trajectory written directly into the simulator",
            pilot_dir / "completed_reference_kinematic.mp4",
            note=(
                "The canonical 50 Hz reference written straight into Isaac with write_joint_state_to_sim: "
                "no controller, no PD, no support band. A non-zero write error here would mean the reference "
                "is not what the sim shows."
            ),
            facts=kinematic_facts,
            relative_to=pilot_dir,
        ),
        video_block(
            "3 · SONIC controller driven by the produced latent",
            pilot_dir / "sonic_latent_motion.mp4",
            note=(
                "Complete token-stream interval only. A is the canonical reference, B is SONIC's joint-position "
                "target q, and C is the measured robot. B's velocity, gains and torque fields are retained in the "
                "tracking trace. The fidelity gate fails if recording ends before the final token."
            ),
            facts=sonic_facts,
            relative_to=pilot_dir,
        ),
    ]

    fidelity_plots = "".join(
        f'<img src="{html.escape(name)}" alt="{html.escape(name)}">'
        for name in ("sonic_left_wrist_world_abc.png", "sonic_left_wrist_abc.png", "sonic_left_arm_abc.png", "sonic_left_hand_abc.png")
        if (pilot_dir / name).is_file()
    )
    if fidelity_plots:
        sections.append(
            '<section><h2>Primary motion-fidelity tests: A → B → C</h2>'
            '<p class="note">Publisher wall time aligns each robot sample to the latest sent token. '
            'Transport delay is not measured; judge fast transitions with that limit in mind.</p>'
            + fidelity_plots + '</section>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page_title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0 auto; max-width: 1100px; padding: 24px 16px 64px;
         font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
         background: #14161a; color: #e7e9ee; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin: 0 0 6px; }}
  .sub {{ color: #9aa2b1; margin: 0 0 16px; font-size: 13px; }}
  .badges {{ margin: 0 0 20px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .badge {{ padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .badge.pass {{ background: #16351f; color: #6ee7a0; }}
  .badge.fail {{ background: #3a1a1c; color: #ff8f8f; }}
  .badge.unknown {{ background: #33373f; color: #cfd4de; }}
  section {{ margin: 0 0 28px; padding: 16px; border: 1px solid #262a32; border-radius: 10px; background: #191c21; }}
  video {{ width: 100%; max-height: 60vh; background: #000; border-radius: 6px; display: block; }}
  img {{ width: 100%; margin: 8px 0; background: white; border-radius: 6px; }}
  .note {{ color: #9aa2b1; font-size: 13px; margin: 0 0 10px; }}
  .file {{ color: #7d8798; font-size: 12px; margin: 6px 0 10px; }}
  .missing {{ color: #ff8f8f; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 3px 8px 3px 0; border-bottom: 1px solid #23262d; vertical-align: top; }}
  th {{ color: #9aa2b1; font-weight: 500; white-space: nowrap; width: 42%; }}
  code {{ font-family: ui-monospace, monospace; font-size: 12px; color: #b6c0d0; }}
  td.ok {{ color: #6ee7a0; }} td.bad {{ color: #ff8f8f; }}
  details {{ margin-top: 10px; }}
  summary {{ cursor: pointer; color: #9aa2b1; font-size: 13px; }}
</style>
</head>
<body>
<h1>{html.escape(page_title)}</h1>
<p class="sub">dataset <code>{html.escape(str(manifest.get('dataset_name', '—')))}</code>
 · episode {html.escape(str(manifest.get('episode_index', '—')))}
 · {html.escape(str(manifest.get('frames', '—')))} frames
 · {html.escape(str(manifest.get('created_utc', '—')))}</p>
<div class="badges">
  {badge(f"episode QC {verdict}", verdict_kind)}
  {badge(f"78D action {action}", "pass" if action == "available" else "unknown")}
  {badge(f"hand schema {hand_schema}", "pass" if hand_schema == "verified" else "unknown")}
</div>
{''.join(sections)}
<section>
  <h2>Encoder and QC detail</h2>
  <table>{encoder_facts and ''.join(f'<tr><th>{html.escape(a)}</th><td>{html.escape(b)}</td></tr>' for a, b in encoder_facts)}</table>
  <details><summary>QC decisions ({len(qc.get('decisions', []))})</summary>
    <table><tr><th>metric</th><th>result</th><th>value</th><th>limit</th></tr>{decisions}</table>
  </details>
</section>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    pilot_dir = args.pilot_dir.resolve()
    if not (pilot_dir / "run_manifest.json").is_file():
        print(f"error: no run_manifest.json in {pilot_dir}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(pilot_dir, title=args.title), encoding="utf-8")
    print(f"{args.output}: {args.output.stat().st_size} bytes from {pilot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
