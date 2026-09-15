#!/usr/bin/env python3
"""Build one responsive review page per pilot run: videos, plots, verdicts.

Each card shows the source head/ego camera next to the fixed-base direct joint
oracle and the free SONIC run, the per-joint plots of both simulations, the QC
and encoder verdicts, and every assumption the conversion made.  Missing
artifacts are shown as NOT RUN / UNVERIFIED instead of being hidden.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.controllers.sonic import (  # noqa: E402
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    BODY_MOTOR_FAMILY,
    SONIC_REFERENCE_JOINT_ORDER,
    deploy_gains,
)
from humanoid_lab.datasets.sonic.joints import reorder  # noqa: E402
from humanoid_lab.datasets.sonic.quality import evaluate_thresholds, overall_result  # noqa: E402
from humanoid_lab.datasets.sonic.tracking import (  # noqa: E402
    DIRECT_THRESHOLDS,
    FREE_THRESHOLDS,
    load_tracking,
    max_command_step,
    motion_start_index,
    tracking_metrics,
)

DEFAULT_PROCESSED_ROOT = Path("/data/datasets/first_tur_processed/sonic_v1_1")
BADGE = {"PASS": "pass", "FAIL": "fail", "UNVERIFIED": "unverified", "NOT RUN": "missing"}


def badge(value: str) -> str:
    return f'<span class="badge {BADGE.get(value, "missing")}">{html.escape(value)}</span>'


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def moov_is_early(path: Path) -> bool:
    """True when the moov atom sits before the payload (faststart, streamable)."""
    with path.open("rb") as handle:
        offset = 0
        while True:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) < 8:
                return True
            size = int.from_bytes(header[:4], "big")
            kind = header[4:8]
            if size == 1:
                size = int.from_bytes(handle.read(8), "big")
            if size < 8:
                return True
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            offset += size


def playable_copy(source: Path, media_dir: Path) -> Path:
    """Path to hand to the player: the source, or a faststart remux when needed.

    The sim recorders and the dataset extractor write moov last, so a browser
    must read the tail of the file before it knows the duration.  Remuxing with
    ``-c copy`` fixes that losslessly without rewriting the run artifact (whose
    hash may be recorded elsewhere), so the copy lives under the review output
    and is keyed by run directory to keep stale copies unreachable.
    """
    if not source.is_file() or moov_is_early(source):
        return source
    if shutil.which("ffmpeg") is None:
        return source
    media_dir.mkdir(parents=True, exist_ok=True)
    copy = media_dir / source.name
    if copy.is_file() and copy.stat().st_mtime >= source.stat().st_mtime:
        return copy
    temporary = media_dir / f"{source.name}.remux.mp4"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-c", "copy", "-movflags", "+faststart", str(temporary)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        return source
    temporary.replace(copy)
    return copy


def plot_tracking(
    run_dir: Path, tracking_path: Path, stem: str, title: str, *, reference: np.ndarray | None, window: str
) -> dict:
    """Per-joint error bars, target-vs-measured traces, and root stability."""
    tracking = load_tracking(tracking_path)
    metrics = tracking_metrics(tracking, window=window)
    metrics["command_step"] = max_command_step(tracking)
    names = list(tracking["body_joint_names"])
    time = tracking["sim_s"]
    mask = ~tracking["support_active"] if not np.all(tracking["support_active"]) else np.ones_like(time, bool)
    # The reference is stored in official SONIC reference/IsaacLab order; the
    # simulator traces use hardware/MuJoCo order, so remap by name before drawing.
    aligned_reference = None
    if reference is not None:
        aligned_reference = reorder(reference, SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER)
        start = motion_start_index(tracking)
        if start is not None:
            time = time - time[start]
            aligned_reference = aligned_reference[: len(time)]
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)

    mae = np.array([metrics["body"]["per_joint"][name]["mae"] for name in names])
    p95 = np.array([metrics["body"]["per_joint"][name]["p95"] for name in names])
    order = np.argsort(-mae)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    positions = np.arange(len(names))
    axis.bar(positions - 0.2, mae[order], width=0.4, label="MAE")
    axis.bar(positions + 0.2, p95[order], width=0.4, label="p95")
    axis.set_xticks(positions)
    axis.set_xticklabels([names[index] for index in order], rotation=90, fontsize=7)
    axis.set_ylabel("rad")
    axis.set_title(f"{title}: per-joint tracking error ({metrics['window']} window)")
    axis.legend()
    figure.tight_layout()
    error_plot = plots / f"{stem}_per_joint_error.png"
    figure.savefig(error_plot, dpi=150)
    plt.close(figure)

    worst = [names[index] for index in order[:6]]
    figure, axes = plt.subplots(len(worst), 1, figsize=(12, 1.6 * len(worst)), sharex=True)
    axes = np.atleast_1d(axes)
    measured = tracking["body_measured"]
    target = tracking["body_target"]
    for axis, name in zip(axes, worst):
        index = names.index(name)
        axis.plot(time, measured[:, index], label="measured", linewidth=1.0)
        axis.plot(time, target[:, index], label="sim target", linewidth=1.0, linestyle="--")
        if aligned_reference is not None:
            axis.plot(time[: len(aligned_reference)], aligned_reference[:, index], label="reference", linewidth=0.8, alpha=0.7)
        axis.set_ylabel(name.replace("_joint", ""), fontsize=8)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("time since motion start (s)")
    figure.suptitle(f"{title}: worst six body joints")
    figure.tight_layout()
    trace_plot = plots / f"{stem}_tracking_traces.png"
    figure.savefig(trace_plot, dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
    axes[0].plot(time, tracking["root_position"][:, 2])
    axes[0].set_ylabel("root z (m)")
    axes[1].plot(time, np.degrees(np.arccos(tracking["root_up_z"])))
    axes[1].set_ylabel("root tilt (deg)")
    axes[1].set_xlabel("time since motion start (s)")
    if not np.all(mask):
        for axis in axes:
            axis.axvline(time[mask][0], color="tab:red", linestyle="--", label="support release")
        axes[0].legend()
    figure.suptitle(f"{title}: root stability")
    figure.tight_layout()
    root_plot = plots / f"{stem}_root.png"
    figure.savefig(root_plot, dpi=150)
    plt.close(figure)

    metrics["plots"] = {
        "per_joint_error": str(error_plot.relative_to(run_dir)),
        "tracking_traces": str(trace_plot.relative_to(run_dir)),
        "root": str(root_plot.relative_to(run_dir)),
    }
    return metrics


def triple_video(run_dir: Path, *, direct_trim: float, free_trim: float) -> str | None:
    source = run_dir / "source.mp4"
    direct = run_dir / "direct_fixed.mp4"
    free = run_dir / "sonic_free.mp4"
    if not (source.is_file() and direct.is_file() and free.is_file()):
        return None
    output = run_dir / "triple_comparison.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-ss", f"{direct_trim:.3f}", "-i", str(direct),
        "-ss", f"{free_trim:.3f}", "-i", str(free),
        "-filter_complex",
        "[0:v]scale=480:-2,pad=480:480:(ow-iw)/2:(oh-ih)/2[a];"
        "[1:v]scale=480:-2,pad=480:480:(ow-iw)/2:(oh-ih)/2[b];"
        "[2:v]scale=480:-2,pad=480:480:(ow-iw)/2:(oh-ih)/2[c];"
        "[a][b][c]hstack=inputs=3:shortest=1[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        # One keyframe per second plus a leading moov: the review page scrubs
        # these clips, and a coarse GOP makes every seek land seconds away.
        "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
        "-movflags", "+faststart", "-shortest", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not output.is_file():
        return None
    return output.name


def verdict_table(title: str, decisions: list[dict]) -> str:
    rows = []
    for item in decisions:
        value = "" if item["value"] is None else f"{item['value']:.6g}"
        rows.append(
            f"<tr><td>{html.escape(str(item['metric']))}</td><td>{value}</td>"
            f"<td>{item['limit']:.6g}</td><td>{badge(item['result'])}</td></tr>"
        )
    return (
        f"<h4>{html.escape(title)}</h4><table class='metrics'><thead><tr>"
        "<th>metric</th><th>value</th><th>limit</th><th>result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def per_joint_table(metrics: dict, side: str) -> str:
    entries = metrics["per_joint"]
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{item['mae']:.4f}</td><td>{item['p95']:.4f}</td>"
        f"<td>{item['max']:.4f}</td></tr>"
        for name, item in entries.items()
    )
    return (
        f"<table class='metrics small'><thead><tr><th>{html.escape(side)} joint</th><th>MAE</th>"
        f"<th>p95</th><th>max</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def sync_bar_html(group: str) -> str:
    """Shared play/pause and timeline for every video tagged with this group.

    The bar is inert markup: the page script binds it to ``video[data-sync-group]``
    elements, so videos added later (whole-body kinematics, debug renders) join
    the same control by carrying the attribute — no markup or id coupling.
    """
    return (
        f"<div class='syncer' data-sync-controls='{html.escape(group, quote=True)}'>"
        "<button type='button' class='sync-toggle' aria-pressed='false'>▶︎ Oynat</button>"
        "<input type='range' class='sync-seek' min='0' max='1' step='0.01' value='0' "
        "aria-label='Ortak zaman çizelgesi (saniye)'>"
        "<span class='sync-clock'>0.00 / 0.00 s</span>"
        "<span class='sync-hint' role='status'></span></div>"
    )


def pd_wrist_diagnostic(tracking_path: Path, reference: np.ndarray | None) -> dict | None:
    """Why the PD oracle lags on the wrists: index swap or drive authority?

    Two questions are answered numerically: does a wrist joint's measurement
    track *another* joint's command (an index/binding swap would show a mutual
    pair), and how much torque the error demands compared with the pinned effort
    limit of that joint's motor family (drive saturation).
    """
    if not tracking_path.is_file():
        return None
    tracking = load_tracking(tracking_path)
    names = list(tracking["body_joint_names"])
    target, measured = tracking["body_target"], tracking["body_measured"]
    kp, _kd = deploy_gains()
    wrists = [name for name in names if name.startswith(("left_wrist", "right_wrist"))]
    joints: dict[str, dict] = {}
    for joint in wrists:
        index = names.index(joint)
        own = float(np.abs(measured[:, index] - target[:, index]).mean())
        alternatives = sorted(
            (float(np.abs(measured[:, index] - target[:, other]).mean()), names[other])
            for other in range(len(names))
            if other != index
        )
        error = np.abs(measured[:, index] - target[:, index])
        tau = kp[index] * error
        limit = float(BODY_EFFORT_LIMIT_NM[index])
        joints[joint] = {
            "own_mae_rad": own,
            "best_other_joint": alternatives[0][1],
            "best_other_mae_rad": alternatives[0][0],
            "kp_nm_per_rad": float(kp[index]),
            "tau_p99_nm": float(np.percentile(tau, 99)),
            "effort_limit_nm": limit,
            "saturated_frames": int((tau > limit).sum()),
            "frames": int(len(tau)),
        }
    # A swap needs mutual evidence: measured[i] tracks target[j] *and* the other
    # way round.  One-sided similarity is what a small, near-constant command
    # looks like, so it is not treated as evidence.
    swaps = []
    for first in wrists:
        for second in wrists:
            if first >= second:
                continue
            i, j = names.index(first), names.index(second)
            forward = float(np.abs(measured[:, i] - target[:, j]).mean())
            backward = float(np.abs(measured[:, j] - target[:, i]).mean())
            own_first = float(np.abs(measured[:, i] - target[:, i]).mean())
            own_second = float(np.abs(measured[:, j] - target[:, j]).mean())
            if forward < own_first * 0.5 and backward < own_second * 0.5:
                swaps.append({"pair": [first, second], "forward_mae": forward, "backward_mae": backward})
    return {
        "joints": joints,
        "index_swap_pairs": swaps,
        "index_swap_refuted": not swaps,
        "motor_families": {name: BODY_MOTOR_FAMILY[names.index(name)] for name in wrists},
        "note": (
            "kp comes from the pinned deployment gains (deploy_gains); the effort limit is the pinned "
            "per-joint limit of the same motor family.  A joint whose demanded torque stays below its "
            "limit while its error does not decay is an authority/binding question, not a saturating drive."
        ),
    }


def pilot_card(run_dir: Path, *, relative_base: str, relative_to: Path, media_root: Path, make_triple: bool) -> dict:
    base = relative_base.rstrip("/")
    prefix = f"{base}/" if base else ""
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest is None:
        raise SystemExit(f"{run_dir} has no run_manifest.json")
    group = manifest["pilot"]
    media_dir = media_root / group / run_dir.name
    encoder = read_json(run_dir / "encoder_manifest.json")
    direct_metrics = read_json(run_dir / "direct_metrics.json")
    sonic_metrics = read_json(run_dir / "sonic_metrics.json")
    sonic_latent_metrics = read_json(run_dir / "sonic_latent_metrics.json")
    reference_path = run_dir / "reference.npz"
    reference = np.load(reference_path)["joint_pos"] if reference_path.is_file() else None

    sections: list[str] = []
    artifacts: dict[str, object] = {}
    qc = manifest["qc"]
    sections.append(
        "<p class='assume'><strong>Kaynak:</strong> "
        f"{html.escape(manifest['dataset_name'])} episode {manifest['episode_index']} · "
        f"{manifest['frames']} kare · {manifest['duration_s']:.2f}s · "
        f"{manifest['source_fps']}→{manifest['processed_fps']} Hz · "
        f"pilot <code>{html.escape(manifest['pilot'])}</code></p>"
    )
    verdicts = [
        ("pilot QC", qc.get("episode_result", qc["result"])),
        ("78D action", manifest["final_action_78d"]["status"].upper().replace("AVAILABLE", "PASS").replace("BLOCKED", "UNVERIFIED")),
        ("hand schema", manifest["hand_schema_status"].upper().replace("VERIFIED", "PASS").replace("UNRESOLVED", "UNVERIFIED")),
    ]
    bulk_conversion = manifest.get("bulk_conversion", {})
    if bulk_conversion.get("status") == "excluded":
        verdicts.append(("bulk conversion", "FAIL"))
        sections.append(
            "<p class='missing-box'><strong>PROCESSED DÖNÜŞÜMÜNDEN ELENDİ.</strong> "
            + html.escape(str(bulk_conversion.get("reason", "")))
            + " Ham veri ve bu pilot kanıtları arşivde tutulur.</p>"
        )
    if encoder:
        verdicts.append(("encoder finite", "PASS" if encoder["finite"] else "FAIL"))
        verdicts.append(("encoder repeatable", "PASS" if encoder["repeatability"]["bitwise_identical"] else "FAIL"))
    sections.append("<p>" + " ".join(f"{html.escape(name)} {badge(value)}" for name, value in verdicts) + "</p>")

    if qc.get("decisions"):
        sections.append(verdict_table("Pilot QC thresholds", qc["decisions"]))

    direct_analysis = None
    sonic_analysis = None
    if (run_dir / "direct_tracking.parquet").is_file():
        direct_analysis = plot_tracking(
            run_dir,
            run_dir / "direct_tracking.parquet",
            "direct",
            "Fixed-base direct joint oracle",
            reference=reference,
            window="motion",
        )
        decisions = evaluate_thresholds(direct_analysis["values"], DIRECT_THRESHOLDS)
        direct_analysis["decisions"] = decisions
        direct_analysis["result"] = overall_result(decisions)
        artifacts["direct"] = True
    wrist_diagnostic = pd_wrist_diagnostic(run_dir / "direct_tracking.parquet", reference)
    sonic_tracking_path = (
        run_dir / "sonic_latent_tracking.parquet"
        if (run_dir / "sonic_latent_tracking.parquet").is_file()
        else run_dir / "sonic_tracking.parquet"
    )
    sonic_analysis_title = (
        "Offline latent → Free SONIC"
        if sonic_tracking_path.name == "sonic_latent_tracking.parquet"
        else "Live-encoder Free SONIC"
    )
    if sonic_tracking_path.is_file():
        sonic_analysis = plot_tracking(
            run_dir, sonic_tracking_path, "sonic_latent" if sonic_latent_metrics else "sonic",
            sonic_analysis_title, reference=None, window="free"
        )
        decisions = evaluate_thresholds(sonic_analysis["values"], FREE_THRESHOLDS)
        sonic_analysis["decisions"] = decisions
        sonic_analysis["result"] = overall_result(decisions)
        artifacts["sonic"] = True

    videos = []
    playable_videos = 0
    whole_body = manifest["kind"] in {"fruits", "apple"}
    kinematic_metrics = read_json(run_dir / "recorded_state_kinematic_metrics.json")
    kinematic_action_metrics = read_json(run_dir / "encoder_action_kinematic_metrics.json")
    completed_reference_metrics = read_json(run_dir / "completed_reference_kinematic_metrics.json")
    # The kinematic replay is the dataset view: it writes the reference state
    # straight into the simulation, so it shows what the recording/command
    # actually contains.  The PD/free runs stay below as controller debugging.
    if whole_body:
        main_slots = (
            ("1. Kaynak ego/head kamera", "source.mp4"),
            ("2. Kaydedilmiş state (kinematik)", "recorded_state_kinematic.mp4"),
            ("3. Encoder action (kinematik)", "encoder_action_kinematic.mp4"),
        )
        debug_slots = (
            ("A. Sabit taban doğrudan joint oracle (PD)", "direct_fixed.mp4"),
            ("B. Serbest SONIC (PD + fizik)", "sonic_free.mp4"),
        )
    else:
        main_slots = (
            ("1. Kaynak ego/head kamera", "source.mp4"),
            ("2. Oluşturulan tüm-beden referansı (frame-exact kinematik)", "completed_reference_kinematic.mp4"),
            ("3. Offline latent → SONIC controller + fizik", "sonic_latent_motion.mp4"),
        )
        debug_slots = (
            ("A. Sabit taban PD takip denemesi (ana veri görseli değil)", "direct_fixed.mp4"),
            ("B. Aynı referans → canlı C++ encoder → SONIC (karşılaştırma)", "sonic_free.mp4"),
        )

    def slot(label: str, name: str, sync_group: str) -> str:
        nonlocal playable_videos
        path = run_dir / name
        if path.is_file():
            source = os.path.relpath(playable_copy(path, media_dir), relative_to)
            playable_videos += 1
            return (
                f"<div class='video-slot'><h3>{html.escape(label)}</h3>"
                f"<video controls preload='metadata' playsinline data-sync-group='{html.escape(sync_group, quote=True)}' "
                f"src='{html.escape(source)}'></video></div>"
            )
        return (
            f"<div class='video-slot'><h3>{html.escape(label)}</h3>"
            f"<p class='missing-box'>NOT RUN · {html.escape(name)} yok</p></div>"
        )

    # Render slots first because slot() determines whether the group has any
    # playable media. The sync bar must be emitted only after that count exists.
    main_video_html = "".join(slot(label, name, group) for label, name in main_slots)
    if playable_videos:
        sections.append(sync_bar_html(group))
    # The syncer binds by attribute, not by name: adding a video to this card
    # with the same data-sync-group joins the shared timeline automatically.
    sections.append("<div class='videos'>" + main_video_html + "</div>")
    if whole_body:
        sections.append(
            "<p class='assume'>Ana grup: kaynak kamera + kaydedilmiş state kinematik + encoder action kinematik "
            "(aynı sync grubu; ortak slider üçünü birlikte sarar).</p>"
        )
    else:
        sections.append(
            "<p class='assume'>Ana grup: kaynak kamera + kayıtta olmayan bacak/bel/root kanalları sabit standing ile "
            "tamamlandıktan sonra oluşan 50 Hz tüm-beden referansın controller'sız, frame-exact oynatımı + aynı "
            "referanstan offline ONNX encoder ile kaydedilmiş 64D tokenlar resmi Protocol v4 girişinden SONIC "
            "controller'a verilince robotun fizik içindeki hareketi. Canlı C++ encoder'ın aynı referanstan ürettiği "
            "ayrı koşu aşağıdaki debugging grubunda tutulur.</p>"
        )
    if debug_slots:
        sections.append("<h3>Controller debugging (PD + fizik; ana veri görseli değil)</h3>")
        debug_group = f"{group}-controller-debug"
        sections.append(sync_bar_html(debug_group))
        sections.append(
            "<div class='videos'>" + "".join(slot(label, name, debug_group) for label, name in debug_slots) + "</div>"
        )

    kinematic_rows = []
    for label, payload in (
        ("oluşturulan tüm-beden referansı", completed_reference_metrics),
        ("recorded state", kinematic_metrics),
        ("encoder action", kinematic_action_metrics),
    ):
        kinematic = (payload or {}).get("kinematic")
        if not kinematic:
            continue
        errors = kinematic["write_error_rad"]
        root_error = kinematic.get("root_write_error", {})
        root_max = float(root_error.get("position_max_m", 0.0))
        result = "PASS" if max(errors["body_max"], errors["left_hand_max"], errors["right_hand_max"], root_max) <= 1e-5 else "FAIL"
        kinematic_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{badge(result)}</td>"
            f"<td>{kinematic['frames_replayed']}/{kinematic['reference_frames']}</td>"
            f"<td>{kinematic['source_fps']:.0f}</td><td>{'evet' if kinematic['hands_applied'] else 'hayır (blocked)'}</td>"
            f"<td>{html.escape(str(kinematic.get('root_source', 'legacy fallback')))}</td>"
            f"<td>{root_max:.2e}</td><td>{errors['body_max']:.2e}</td><td>{errors['body_mean']:.2e}</td>"
            f"<td>{errors['left_hand_max']:.2e}</td><td>{errors['right_hand_max']:.2e}</td>"
            f"<td>{kinematic['min_sole_z_m']['min']:.4f}</td></tr>"
        )
    if kinematic_rows:
        sections.append(
            "<h3>Kinematik replay (frame-exact)</h3><table class='metrics'><thead><tr>"
            "<th>referans</th><th>sonuç</th><th>kare</th><th>fps</th><th>eller</th>"
            "<th>root kaynağı</th><th>root max (m)</th><th>body max</th><th>body ort.</th>"
            "<th>sol el max</th><th>sağ el max</th><th>min sole z (m)</th>"
            f"</tr></thead><tbody>{''.join(kinematic_rows)}</tbody></table>"
            "<p class='assume'>write error = hedef ile simülasyondaki ölçüm arasındaki fark; her karede state "
            "yazıldıktan sonra ölçülür, arada fizik adımı yoktur.</p>"
        )

    triple = None
    if make_triple and len(videos) == 3 and all((run_dir / name).is_file() for name in ("source.mp4", "direct_fixed.mp4", "sonic_free.mp4")):
        direct_trim = 1.0
        if (run_dir / "direct_tracking.parquet").is_file():
            direct_tracking = load_tracking(run_dir / "direct_tracking.parquet")
            start = motion_start_index(direct_tracking)
            direct_trim = 0.0 if start is None else float(direct_tracking["sim_s"][start])
        free_trim = 0.0
        if sonic_metrics and sonic_metrics.get("controller", {}).get("support", {}).get("release_tick"):
            free_trim = float(sonic_metrics["controller"]["support"]["release_tick"]) * 0.005
        triple = triple_video(run_dir, direct_trim=direct_trim, free_trim=free_trim)
    if triple:
        sections.append(f"<p>Senkron üçlü video: <a href='{prefix}{triple}'>{triple}</a></p>")

    for title, analysis in (("Fixed-base direct oracle", direct_analysis), (sonic_analysis_title, sonic_analysis)):
        if analysis is None:
            sections.append(f"<p class='missing-box'>{html.escape(title)}: NOT RUN</p>")
            continue
        sections.append(
            f"<h3>{html.escape(title)} — {badge(analysis['result'])} "
            f"({analysis['frames']} kare, {analysis['sim_seconds']:.2f}s {html.escape(analysis['window'])} pencere)</h3>"
        )
        sections.append(verdict_table(f"{title} thresholds ({analysis['window']} window)", analysis["decisions"]))
        group_rows = "".join(
            f"<tr><td>{html.escape(group)}</td><td>{item['mae']:.4f}</td><td>{item['p95']:.4f}</td>"
            f"<td>{item['max']:.4f}</td></tr>"
            for group, item in analysis["groups"].items()
        )
        sections.append(
            "<table class='metrics small'><thead><tr><th>joint group</th><th>MAE</th><th>p95</th><th>max</th>"
            f"</tr></thead><tbody>{group_rows}</tbody></table>"
        )
        sections.append(
            "<p class='assume'>root min/final z: "
            f"{analysis['stability']['min_root_height_m']:.3f}/{analysis['stability']['final_root_height_m']:.3f} m · "
            f"max tilt {analysis['stability']['max_tilt_deg']:.1f}° · fell: {bool(analysis['stability']['fell'])} · "
            f"largest single-tick command step: {analysis['command_step']['value']:.3f} rad at "
            f"{analysis['command_step']['sim_s']:.1f}s</p>"
        )
        deployment = direct_metrics if title.startswith("Fixed-base") else (sonic_latent_metrics or sonic_metrics)
        if deployment and deployment.get("controller", {}).get("body_tracking_error_rad"):
            reported = deployment["controller"]["body_tracking_error_rad"]
            sections.append(
                "<p class='assume'>deployment-reported command tracking (all commands it applied, including "
                f"mode switches): mean {reported['mean']:.4f} rad · max {reported['max']:.4f} rad</p>"
            )
        plots = analysis["plots"]
        sections.append(
            "<div class='plots'>"
            f"<img src='{prefix}{plots['per_joint_error']}'><img src='{prefix}{plots['root']}'></div>"
            f"<div class='plots'><img src='{prefix}{plots['tracking_traces']}'></div>"
        )
        sections.append("<details><summary>Per-joint body table</summary>" + per_joint_table(analysis["body"], "body") + "</details>")

    if wrist_diagnostic:
        rows = []
        for joint, item in wrist_diagnostic["joints"].items():
            rows.append(
                f"<tr><td>{html.escape(joint)}</td><td>{item['own_mae_rad']:.3f}</td>"
                f"<td>{html.escape(item['best_other_joint'])} ({item['best_other_mae_rad']:.3f})</td>"
                f"<td>{item['kp_nm_per_rad']:.1f}</td><td>{item['tau_p99_nm']:.2f}</td>"
                f"<td>{item['effort_limit_nm']:.1f}</td><td>{item['saturated_frames']}/{item['frames']}</td></tr>"
            )
        verdict = "çürütüldü" if wrist_diagnostic["index_swap_refuted"] else "şüpheli"
        sections.append(
            "<h3>PD bilek teşhisi — controller debugging</h3>"
            "<table class='metrics'><thead><tr><th>eklem</th><th>kendi MAE (rad)</th><th>en iyi başka hedef</th>"
            "<th>kp (Nm/rad)</th><th>tau p99 (Nm)</th><th>limit (Nm)</th><th>doyan kare</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"<p class='assume'>index swap: <strong>{verdict}</strong> (karşılıklı eşleşme aranır) · "
            f"{html.escape(wrist_diagnostic['note'])}</p>"
        )
    if encoder:
        sections.append(
            "<details><summary>Encoder provenance</summary><table class='metrics small'>"
            f"<tr><td>model</td><td colspan='3'>{html.escape(encoder['encoder']['path'])}</td></tr>"
            f"<tr><td>sha256</td><td colspan='3'><code>{html.escape(encoder['encoder']['sha256'])}</code></td></tr>"
            f"<tr><td>onnxruntime</td><td colspan='3'>{html.escape(encoder['encoder']['onnxruntime_version'])} "
            f"({html.escape(', '.join(encoder['encoder']['providers']))})</td></tr>"
            f"<tr><td>token norm p50/p99</td><td>{encoder['token_norm']['p50']:.4f}</td>"
            f"<td colspan='2'>{encoder['token_norm']['p99']:.4f}</td></tr>"
            f"<tr><td>orientation policy</td><td colspan='3'>{html.escape(encoder['encoder_orientation_policy'])}</td></tr>"
            "</table><p class='assume'>"
            + html.escape(encoder["parity_note"])
            + "</p></details>"
        )
    else:
        sections.append("<p class='missing-box'>Encoder: NOT RUN (action_tokens.npz yok)</p>")

    assumptions = [
        f"orientation policy: {manifest['encoder_orientation_policy']} — {manifest['encoder_orientation_policy_note']}",
        f"standing completion: {manifest['standing_completion_policy']} (scope {manifest['standing_completion_scope']})",
        f"hand schema: {manifest['hand_schema_status']} — {manifest['hand_schema_reason']}",
    ]
    assumptions += [f"note: {note}" for note in manifest.get("notes", [])]
    assumptions += [f"provenance: {key} = {json.dumps(value)}" for key, value in manifest["provenance"].items()]
    sections.append(
        "<details open><summary>Assumptions &amp; provenance</summary><ul class='assume'>"
        + "".join(f"<li>{html.escape(item)}</li>" for item in assumptions)
        + "</ul></details>"
    )
    links = [
        f"<a href='{prefix}{name}'>{name}</a>"
        for name in ("run_manifest.json", "encoder_manifest.json", "reference.parquet", "action_tokens.npz",
                     "direct_tracking.parquet", "sonic_tracking.parquet", "sonic_latent_tracking.parquet",
                     "direct_metrics.json", "sonic_metrics.json", "sonic_latent_metrics.json", "human_review.json")
        if (run_dir / name).is_file()
    ]
    sections.append("<p>" + " · ".join(links) + "</p>")
    sections.append(
        f"<p class='assume'>human review: <strong>{html.escape(str(manifest.get('human_review')))}</strong> — "
        f"<code>{html.escape(str(run_dir / 'human_review.json'))}</code></p>"
    )

    card = (
        f"<section id='{html.escape(manifest['pilot'])}'><h2>{html.escape(manifest['pilot'])} "
        f"<span class='tag'>{html.escape(manifest['kind'])}</span></h2>"
        + "".join(sections)
        + "</section>"
    )
    return {
        "card": card,
        "name": manifest["pilot"],
        "run_dir": str(run_dir),
        "qc_result": qc.get("episode_result", qc["result"]),
        "direct_result": None if direct_analysis is None else direct_analysis["result"],
        "sonic_result": None if sonic_analysis is None else sonic_analysis["result"],
        "direct_metrics": direct_analysis,
        "wrist_diagnostic": wrist_diagnostic,
        "sonic_metrics": sonic_analysis,
    }


STYLE = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{font:16px/1.5 system-ui,sans-serif;background:#101216;color:#e8e8e8;margin:0;padding:20px;max-width:1800px;margin-inline:auto}
h1{font-size:1.6rem}h2{font-size:1.25rem;margin-top:0}h3{font-size:1rem}h4{margin:16px 0 6px;font-size:.95rem;color:#9fb3c8}
h1,h2,h3,p,li,a,code{overflow-wrap:anywhere;word-break:break-word}
section{background:#181b21;border-radius:14px;padding:20px;margin:18px 0;min-width:0}
.tag{font-size:.7rem;background:#2a3f5f;padding:2px 8px;border-radius:8px;vertical-align:middle}
.badge{font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:8px;white-space:nowrap}
.badge.pass{background:#14432a;color:#7ee2a8}.badge.fail{background:#4a1717;color:#ff9d9d}
.badge.unverified{background:#4a3a12;color:#ffd479}.badge.missing{background:#333;color:#bbb}
.videos{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));align-items:start}
.syncer{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:12px 0;padding:10px 12px;background:#12151a;border:1px solid #232b36;border-radius:12px}
.syncer button{font:inherit;font-size:.85rem;padding:6px 14px;border-radius:9px;border:1px solid #33415a;background:#1d2a3d;color:#e8e8e8;cursor:pointer;white-space:nowrap}
.syncer button:hover{background:#24344b}
.syncer button:focus-visible,.sync-seek:focus-visible{outline:2px solid #7dcfff;outline-offset:2px}
.sync-seek{flex:1 1 220px;min-width:140px;height:24px;accent-color:#7dcfff;background:transparent}
.sync-seek:disabled{opacity:.45}
.sync-clock{font-variant-numeric:tabular-nums;font-size:.85rem;color:#c8d3df;white-space:nowrap}
.sync-hint{flex:1 1 100%;font-size:.8rem;color:#ffd479}
.sync-hint:empty{display:none}
video{width:100%;background:#000;border-radius:10px;aspect-ratio:4/3;object-fit:contain}
.plots{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));margin:12px 0}
img{width:100%;border-radius:10px;background:#fff}
table.metrics{width:100%;border-collapse:collapse;margin:8px 0;font-size:.85rem;table-layout:fixed}
table.metrics th,table.metrics td{text-align:left;padding:4px 8px;border-bottom:1px solid #262b33;overflow-wrap:anywhere}
table.metrics.small{font-size:.78rem}
details summary{cursor:pointer;margin:8px 0;color:#9fb3c8}
.assume{color:#a9b6c4;font-size:.88rem}
.missing-box{border:1px dashed #444;border-radius:10px;padding:18px;color:#c9a227;text-align:center}
code{background:#0d0f13;padding:1px 5px;border-radius:5px;font-size:.85em}
a{color:#7dcfff}
@media (max-width:700px){body{padding:10px}section{padding:14px}}
"""

# Grouped video control.  Bars carry data-sync-controls, videos carry
# data-sync-group; anything added to a card with that attribute joins the same
# play/pause + timeline, so the component does not depend on file names or on
# the current page structure.
SCRIPT = """
(function () {
  'use strict';
  // Only review-sized clips are pulled into a blob; the fallback exists for
  // servers that ignore byte ranges, where HTMLMediaElement.seekable stays
  // empty and every seek is discarded.
  var BLOB_LIMIT = 64 * 1024 * 1024;
  var objectUrls = [];
  var bars = {};
  var seenVideos = new WeakSet();

  function videosOf(group) {
    return Array.prototype.slice.call(document.querySelectorAll('video[data-sync-group]'))
      .filter(function (video) { return video.dataset.syncGroup === group; });
  }

  function usable(duration) { return typeof duration === 'number' && isFinite(duration) && duration > 0; }

  // A server without byte-range support reports a degenerate [0, 0] range, so
  // length alone says nothing; only a non-empty span means the video can seek.
  function seekableSpan(video) {
    var ranges = video.seekable;
    if (!ranges || ranges.length === 0) return 0;
    return ranges.end(ranges.length - 1) - ranges.start(0);
  }

  function groupDuration(videos) {
    return videos.reduce(function (max, video) {
      return usable(video.duration) ? Math.max(max, video.duration) : max;
    }, 0);
  }

  function seekVideo(video, seconds) {
    if (!usable(video.duration) || seekableSpan(video) <= 0) return;
    try { video.currentTime = Math.max(0, Math.min(seconds, video.duration)); } catch (error) { /* not seekable */ }
  }

  function playVideo(video) {
    var attempt = video.play();
    if (attempt && typeof attempt.catch === 'function') attempt.catch(function () { /* autoplay blocked */ });
  }

  function setText(node, text) { if (node && node.textContent !== text) node.textContent = text; }
  function setValue(seek, value) { if (seek.value !== value) seek.value = value; }

  function clockText(time, duration) {
    var fmt = function (value) { return (isFinite(value) ? value : 0).toFixed(2); };
    return fmt(time) + ' / ' + fmt(duration) + ' s';
  }

  function refresh(group) {
    var bar = bars[group];
    if (!bar) return;
    var videos = videosOf(group);
    var duration = groupDuration(videos);
    bar.seek.max = duration > 0 ? String(duration) : '1';
    bar.seek.disabled = duration <= 0;
    setText(bar.clock, clockText(Number(bar.seek.value), duration));
    var blocked = videos.filter(function (video) {
      return video.readyState >= 1 && usable(video.duration) && seekableSpan(video) <= 0;
    });
    setText(bar.hint, blocked.length
      ? 'seek kapalı: sunucu byte-range göndermiyor (' + blocked.length + '/' + videos.length +
        ' video) — sayfayı ./dev.sh sonic-review-serve ile servis edin'
      : '');
  }

  function applyTime(group, seconds) {
    var bar = bars[group];
    if (!bar || bar.scrubbing) return;
    setValue(bar.seek, String(Math.max(0, seconds)));
    setText(bar.clock, clockText(seconds, groupDuration(videosOf(group))));
  }

  function setToggle(group) {
    var bar = bars[group];
    if (!bar) return;
    var playing = videosOf(group).some(function (video) { return !video.paused && !video.ended; });
    setText(bar.toggle, playing ? '⏸ Duraklat' : '▶︎ Oynat');
    bar.toggle.setAttribute('aria-pressed', playing ? 'true' : 'false');
  }

  function startScrub(group) {
    var bar = bars[group];
    if (!bar || bar.scrubbing) return;
    bar.scrubbing = true;
    var videos = videosOf(group);
    bar.resume = videos.some(function (video) { return !video.paused && !video.ended; });
    videos.forEach(function (video) { video.pause(); });
  }

  function endScrub(group) {
    var bar = bars[group];
    if (!bar || !bar.scrubbing) return;
    bar.scrubbing = false;
    if (bar.resume) {
      videosOf(group).forEach(function (video) {
        if (!usable(video.duration) || video.currentTime < video.duration - 0.05) playVideo(video);
      });
    }
    bar.resume = false;
    setToggle(group);
  }

  function bindBar(element) {
    var group = element.dataset.syncControls;
    if (bars[group]) return;
    var bar = bars[group] = {
      toggle: element.querySelector('.sync-toggle'),
      seek: element.querySelector('.sync-seek'),
      clock: element.querySelector('.sync-clock'),
      hint: element.querySelector('.sync-hint'),
      scrubbing: false,
      resume: false,
    };
    if (!bar.toggle || !bar.seek) return;

    bar.toggle.addEventListener('click', function () {
      var videos = videosOf(group);
      var playing = videos.some(function (video) { return !video.paused && !video.ended; });
      videos.forEach(function (video) {
        if (playing) { video.pause(); return; }
        if (usable(video.duration) && video.currentTime >= video.duration - 0.05) seekVideo(video, 0);
        playVideo(video);
      });
      setToggle(group);
    });

    bar.seek.addEventListener('pointerdown', function () { startScrub(group); });
    bar.seek.addEventListener('keydown', function () { startScrub(group); });
    bar.seek.addEventListener('input', function () {
      var seconds = Number(bar.seek.value);
      videosOf(group).forEach(function (video) { seekVideo(video, seconds); });
      setText(bar.clock, clockText(seconds, groupDuration(videosOf(group))));
    });
    // 'change' covers both pointerup and the keyboard commit; the extra
    // listeners just make the resume deterministic across browsers.
    bar.seek.addEventListener('change', function () { endScrub(group); });
    bar.seek.addEventListener('pointerup', function () { endScrub(group); });
    bar.seek.addEventListener('keyup', function () { endScrub(group); });

    refresh(group);
    setToggle(group);
  }

  // A server that ignores Range leaves seekable empty even once the file is
  // fully buffered.  Review clips are small, so fetch once and play from a blob
  // URL, which is always seekable; anything too large is left alone.
  function upgradeToBlob(video) {
    if (seekableSpan(video) > 0 || video.dataset.blobFallback === 'done') return Promise.resolve();
    if (location.protocol !== 'http:' && location.protocol !== 'https:') return Promise.resolve();
    video.dataset.blobFallback = 'done';
    var url;
    try { url = new URL(video.currentSrc || video.src, location.href); } catch (error) { return Promise.resolve(); }
    if (url.origin !== location.origin) return Promise.resolve();
    return fetch(url.href)
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.blob();
      })
      .then(function (blob) {
        if (!blob.size || blob.size > BLOB_LIMIT) throw new Error('unsuitable size');
        var objectUrl = URL.createObjectURL(blob.type ? blob : new Blob([blob], { type: 'video/mp4' }));
        objectUrls.push(objectUrl);
        var time = video.currentTime;
        var wasPlaying = !video.paused && !video.ended;
        var loaded = new Promise(function (resolve) {
          var settled = false;
          var finish = function () { if (!settled) { settled = true; resolve(); } };
          video.addEventListener('loadeddata', finish, { once: true });
          setTimeout(finish, 5000);
        });
        video.src = objectUrl;
        video.load();
        return loaded.then(function () {
          if (time > 0) seekVideo(video, time);
          if (wasPlaying) playVideo(video);
        });
      })
      .catch(function () { /* keep the native element; the hint explains why seek is off */ });
  }

  function onMediaEvent(video, event) {
    var group = video.dataset.syncGroup;
    if (!bars[group]) return;
    if (event === 'play' || event === 'pause' || event === 'ended') setToggle(group);
    if (event !== 'timeupdate' && event !== 'seeked') return;
    var videos = videosOf(group);
    var playing = videos.filter(function (item) { return !item.paused && !item.ended; });
    var lead = (playing.length ? playing : videos).reduce(function (best, item) {
      return usable(item.duration) && (!best || item.duration > best.duration) ? item : best;
    }, null);
    if (lead === video) applyTime(group, video.currentTime);
  }

  function attach(video) {
    if (seenVideos.has(video)) return;
    seenVideos.add(video);
    var group = video.dataset.syncGroup;
    var onMetadata = function () { upgradeToBlob(video).then(function () { refresh(group); }); };
    if (video.readyState >= 1) onMetadata();
    else video.addEventListener('loadedmetadata', onMetadata, { once: true });
    ['play', 'pause', 'ended', 'seeked', 'timeupdate', 'durationchange'].forEach(function (event) {
      video.addEventListener(event, function () { onMediaEvent(video, event); });
    });
  }

  function init() {
    Array.prototype.slice.call(document.querySelectorAll('[data-sync-controls]')).forEach(bindBar);
    Array.prototype.slice.call(document.querySelectorAll('video[data-sync-group]')).forEach(attach);
    Object.keys(bars).forEach(refresh);
  }

  // Late videos (whole-body kinematics, debug renders) only need the attribute.
  new MutationObserver(init).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener('DOMContentLoaded', init);
  window.addEventListener('pagehide', function () {
    objectUrls.forEach(function (url) { URL.revokeObjectURL(url); });
  });
  init();
})();
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pilot-dir", action="append", type=Path, default=[],
                        help="explicit pilot run directory (repeatable); defaults to the newest of each pilot")
    parser.add_argument("--no-triple", action="store_true")
    args = parser.parse_args()

    if args.pilot_dir:
        run_dirs = [path.resolve() for path in args.pilot_dir]
    else:
        root = args.processed_root / "pilots"
        run_dirs = [
            sorted(path for path in pilot.iterdir() if (path / "run_manifest.json").is_file())[-1]
            for pilot in sorted(root.iterdir())
            if pilot.is_dir() and any((child / "run_manifest.json").is_file() for child in pilot.iterdir())
        ]
    if not run_dirs:
        raise SystemExit(f"no pilot runs found under {args.processed_root / 'pilots'}")
    output = (args.output or args.processed_root.parent / "sonic_pilot_review.html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    media_root = output.parent / "sonic_pilot_review_media"
    results = [
        pilot_card(
            run_dir,
            relative_base=os.path.relpath(run_dir, output.parent),
            relative_to=output.parent,
            media_root=media_root,
            make_triple=not args.no_triple,
        )
        for run_dir in run_dirs
    ]
    summary_rows = "".join(
        f"<tr><td><a href='#{html.escape(item['name'])}'>{html.escape(item['name'])}</a></td>"
        f"<td>{badge(item['qc_result'])}</td><td>{badge(item['direct_result'] or 'NOT RUN')}</td>"
        f"<td>{badge(item['sonic_result'] or 'NOT RUN')}</td><td>{html.escape(os.path.relpath(item['run_dir'], output.parent))}</td></tr>"
        for item in results
    )
    document = (
        "<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>SONIC pilot review</title><style>" + STYLE + "</style></head><body>"
        "<h1>SONIC v1.1 pilot review</h1>"
        "<p class='assume'>Her kart tek bir pilot episode'u gösterir. Whole-body pilotların ana karşılaştırması "
        "kaynak ego kamera, kaydedilmiş state'in frame-exact kinematik replay'i ve encoder'a verilen action "
        "trajectory'sinin frame-exact kinematik replay'idir; PD ve serbest SONIC yalnız controller debugging "
        "bölümündedir. Unitree kartı kaynak kamera, standing ile tamamlanmış direct replay ve serbest SONIC'i "
        "gösterir. Eski başarısız koşular yeni kanıtla karıştırılmaz.</p>"
        "<p class='assume'>Kart içindeki videolar ortak play/pause ve zaman çizelgesi ile senkron sürülür; "
        "süresi kısa olan video seçilen zamana clamp edilir. Native kontroller tek tek oynatmak için durur. "
        "Seek için sayfayı byte-range destekleyen bir sunucudan açın: <code>./dev.sh sonic-review-serve</code> "
        "(<code>python3 -m http.server</code> <code>Accept-Ranges</code> göndermez ve videolar seek edilemez).</p>"
        "<table class='metrics'><thead><tr><th>pilot</th><th>QC</th><th>direct oracle</th><th>free SONIC</th>"
        f"<th>run dir</th></tr></thead><tbody>{summary_rows}</tbody></table>"
        + "".join(item["card"] for item in results)
        + f"<script>{SCRIPT}</script>"
        + "</body></html>\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    (output.parent / "sonic_pilot_review.json").write_text(
        json.dumps(
            [
                {
                    "pilot": item["name"],
                    "run_dir": item["run_dir"],
                    "qc": item["qc_result"],
                    "direct": item["direct_result"],
                    "sonic": item["sonic_result"],
                    "wrist_diagnostic": item.get("wrist_diagnostic"),
                    "direct_metrics": None
                    if item["direct_metrics"] is None
                    else {
                        "window": item["direct_metrics"]["window"],
                        "values": item["direct_metrics"]["values"],
                        "groups": item["direct_metrics"]["groups"],
                        "per_joint": item["direct_metrics"]["body"]["per_joint"],
                        "hands": {
                            side: item["direct_metrics"]["hands"][side]["overall"] for side in ("left", "right")
                        },
                        "stability": item["direct_metrics"]["stability"],
                        "command_step": item["direct_metrics"]["command_step"],
                        "decisions": item["direct_metrics"]["decisions"],
                    },
                    "sonic_metrics": None
                    if item["sonic_metrics"] is None
                    else {
                        "window": item["sonic_metrics"]["window"],
                        "values": item["sonic_metrics"]["values"],
                        "groups": item["sonic_metrics"]["groups"],
                        "per_joint": item["sonic_metrics"]["body"]["per_joint"],
                        "hands": {
                            side: item["sonic_metrics"]["hands"][side]["overall"] for side in ("left", "right")
                        },
                        "stability": item["sonic_metrics"]["stability"],
                        "command_step": item["sonic_metrics"]["command_step"],
                        "decisions": item["sonic_metrics"]["decisions"],
                    },
                }
                for item in results
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"review page: {output}")
    for item in results:
        print(f"  {item['name']}: qc={item['qc_result']} direct={item['direct_result']} sonic={item['sonic_result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
