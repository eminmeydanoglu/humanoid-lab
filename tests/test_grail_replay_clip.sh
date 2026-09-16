#!/usr/bin/env bash
# Acceptance run for the kinematic GRAIL replay path.
#
# Renders one real pickup_table motion headless and checks the contract the
# replay path promises: 43-DOF trajectory, two interpolated video frames per
# motion frame, cameras that actually move through the clip, columns whose
# order matches the articulation the replay drives, and a measured target
# visibility report around the source right-hand grasp.
#
#   tests/test_grail_replay_clip.sh [sequence-key]
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="${1:-pickup_table__apple_0__000}"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
DATA_ROOT="$(sed -n 's/^HUMANOID_DATA_ROOT=//p' .env 2>/dev/null | tail -1)"
DATA_ROOT="${DATA_ROOT:-data}"

# A stale run must not be able to satisfy the checks below.
rm -rf "${DATA_ROOT}/outputs/grail-replay/$KEY"
./dev.sh isaac-g1 grail-replay "$KEY" --headless \
  --visibility --visibility-threshold 0.5 --visibility-before 1.0 --visibility-after 1.0 2>&1 | tee "$LOG"

grep -q "positional_order=True" "$LOG" || {
  echo "FAIL: trajectory columns do not match the articulation's joint order" >&2
  rm -f "$LOG"
  exit 1
}
rm -f "$LOG"

docker compose --env-file .env exec -T dev bash -lc '
  source /opt/humanoid-lab/entrypoint.sh >/dev/null 2>&1
  use-isaac-sonic >/dev/null 2>&1
  KEY="$1" python - <<"PY"
import hashlib, json, math, os, pathlib
import imageio

key = os.environ["KEY"]
out = pathlib.Path("/outputs/grail-replay") / key
manifest = json.loads((out / "manifest.json").read_text())
assert manifest["sequence_key"] == key, manifest["sequence_key"]
assert manifest["result"] == "COMPLETED", json.dumps(manifest, indent=2)
source_frames = manifest["frames"]
frames = manifest["render_frames"]
shape = manifest["trajectory"]["dof_pos_shape"]
assert shape == [source_frames, 43], shape
assert frames == source_frames * 2, (source_frames, frames)
assert manifest["render_fps"] == manifest["fps"] * 2, manifest
assert manifest["articulation"]["positional_order"] is True, manifest["articulation"]
assert manifest["articulation"]["joint_names"] == manifest["trajectory"]["joint_names"]
assert manifest["tracking_error"]["joint_radians"] <= 1e-4, manifest["tracking_error"]
assert manifest["tracking_error"]["root_metres"] <= 1e-5, manifest["tracking_error"]
# The grasp event comes from the source right-hand command, not from replay state.
assert manifest["grasp"] == {"source_frame": 96, "source_time_seconds": 3.84, "render_frame": 192}, manifest["grasp"]
print("OK trajectory: %s source frames @ %g Hz -> %s render frames @ %g Hz, dof shape %s"
      % (source_frames, manifest["fps"], frames, manifest["render_fps"], shape))
print("OK grasp: source frame %d at %.2f s -> render frame %d"
      % (manifest["grasp"]["source_frame"], manifest["grasp"]["source_time_seconds"], manifest["grasp"]["render_frame"]))
# The head camera has to render the D435i stream angles. The rendered pair comes
# from the square-pixel projection (measured by tests/test_head_camera_projection.sh);
# these checks only prove that the prim carries the parameters that pair implies.
camera = manifest["cameras"]["head"]
assert camera["resolution"] == [640, 480], camera
rendered_h, rendered_v = camera["rendered_fov_deg"]
assert abs(rendered_h - 54.9) < 1e-9, camera
assert abs(rendered_v - 42.5712) < 1e-3, camera
assert camera["nominal_fov_deg"] == [54.9, 42.5], camera["nominal_fov_deg"]
usd = camera["usd_attributes"]
assert usd, "replay did not read the head camera prim back"
for name in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm"):
    assert abs(usd[name] - camera[name]) <= 1e-4, (name, usd[name], camera[name])
attributes_h = math.degrees(2.0 * math.atan(usd["horizontal_aperture_mm"] / (2.0 * usd["focal_length_mm"])))
attributes_v = math.degrees(2.0 * math.atan(usd["vertical_aperture_mm"] / (2.0 * usd["focal_length_mm"])))
assert abs(attributes_h - rendered_h) < 0.01, (attributes_h, usd)
assert abs(attributes_v - rendered_v) < 0.01, (attributes_v, usd)
print("OK head camera: prim attributes consistent (focal %.4f mm, apertures %.4f x %.4f mm, fov %.2f x %.2f deg)"
      % (usd["focal_length_mm"], usd["horizontal_aperture_mm"], usd["vertical_aperture_mm"], attributes_h, attributes_v))
for name in ("external", "head"):
    path = out / (name + ".mp4")
    assert path.stat().st_size > 0, "%s is empty" % path
    reader = imageio.get_reader(path)
    expected_size = tuple(manifest["cameras"][name]["resolution"])
    actual_size = tuple(reader.get_meta_data()["size"])
    assert actual_size == expected_size, "%s.mp4 size %s, expected %s" % (name, actual_size, expected_size)
    count = reader.count_frames()
    digests = {hashlib.blake2b(reader.get_data(i).tobytes(), digest_size=8).digest() for i in range(count)}
    reader.close()
    assert count == frames, "%s.mp4 has %d frames, expected %d" % (name, count, frames)
    assert len(digests) > frames // 2, "%s.mp4 has only %d distinct frames of %d" % (name, len(digests), frames)
    print("OK %s.mp4: %d frames, %d distinct" % (name, count, len(digests)))
assert manifest["head_camera_distinct_frames"] > frames // 2, manifest["head_camera_distinct_frames"]
print("OK manifest: %d distinct head frames during the run" % manifest["head_camera_distinct_frames"])

# Visibility analysis. It is opt-in, but when it runs the report has to be
# complete, measured (no unmeasured frames) and internally consistent, and the
# grasp-window summary has to describe the real apple behaviour: visible on
# the table before the hand arrives, hidden during the grasp itself.
report = json.loads((out / "visibility.json").read_text())
assert report["schema_version"] == 1, report["schema_version"]
assert report["sequence_key"] == key, report["sequence_key"]
assert report["result"] == "COMPLETED", json.dumps(report["summary"])
metric = report["metric"]
assert metric["name"] == "visible_fraction", metric
assert metric["occlusion_counted_as_not_visible"] is True, metric
assert metric["image_boundary_clipping_counted_as_not_visible"] is True, metric
assert metric["analysis_view"]["resolution"] == [1280, 960], metric["analysis_view"]
assert metric["analysis_view"]["view_scale"] == 2, metric["analysis_view"]
assert report["head_camera"]["resolution"] == [640, 480], report["head_camera"]
assert report["grasp"]["source_frame"] == 96, report["grasp"]
assert report["grasp"]["render_frame"] == 192, report["grasp"]
window = report["window"]
assert window["threshold"] == 0.5, window
assert (window["start_render_frame"], window["end_render_frame"]) == (142, 242), window
assert window["frames"] == 101 and window["clamped"] is False, window
rows = report["frames"]
assert len(rows) == frames, (len(rows), frames)
assert report["frames_recorded"] == frames, report["frames_recorded"]
for row in rows:
    if row["valid"]:
        assert 0.0 <= row["visible_fraction"] <= 1.0, row
        assert row["visible_pixels"] >= 0, row
        assert row["unoccluded_projected_pixels"] >= 0.0, row
        if row["visible_pixels"] > 0:
            assert row["unoccluded_projected_pixels"] > 0.0, row
    else:
        # An unmeasured frame carries no number at all, so it cannot be
        # mistaken for a measured zero.
        assert row["visible_fraction"] is None, row
    assert abs(row["source_time_seconds"] - row["render_frame"] / manifest["render_fps"]) < 1e-6, row
    expected_relative = row["source_time_seconds"] - report["grasp"]["source_time_seconds"]
    assert abs(row["relative_to_grasp_seconds"] - expected_relative) < 1e-6, row
summary = report["summary"]
assert summary["frames"] == frames, summary
assert summary["valid_frames"] == frames, summary
assert summary["invalid_frames"] == 0, summary
assert summary["window_frames"] == 101 and summary["window_valid_frames"] == 101, summary
assert 0.0 <= summary["window_min_fraction"] <= summary["window_mean_fraction"] <= 1.0, summary
assert abs(summary["grasp_fraction"] - rows[192]["visible_fraction"]) < 1e-9, summary
# The target is on the table and clearly visible well before the grasp.
early = max(row["visible_fraction"] for row in rows if row["render_frame"] < 50)
assert early > 0.9, early
# At the grasp the hand covers it, so the window is below the 0.5 threshold.
assert summary["grasp_fraction"] < 0.5, summary["grasp_fraction"]
assert summary["object_below_threshold_for_whole_window"] is True, summary
assert summary["object_below_threshold_at_any_window_frame"] is True, summary
intervals = summary["below_threshold_intervals"]
assert intervals and intervals[0]["start_render_frame"] == 142, intervals
print("OK visibility: %s frames, grasp %.3f, window mean %.3f, below-threshold frames %d/%d"
      % (summary["frames"], summary["grasp_fraction"], summary["window_mean_fraction"],
         summary["window_below_threshold_frames"], summary["window_frames"]))
reference = manifest["visibility"]
assert reference["enabled"] is True, reference
assert reference["report"] == str(out / "visibility.json"), reference
assert reference["threshold"] == window["threshold"], reference
assert reference["window_start_render_frame"] == window["start_render_frame"], reference
assert reference["window_end_render_frame"] == window["end_render_frame"], reference
assert reference["object_below_threshold_for_whole_window"] == summary["object_below_threshold_for_whole_window"], reference
assert reference["invalid_frames"] == 0, reference
print("OK manifest: visibility report referenced with %d window frames" % reference["window_frames"])
PY
' -- "$KEY"
