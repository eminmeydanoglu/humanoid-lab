#!/usr/bin/env bash
# Acceptance run for the kinematic GRAIL replay path.
#
# Renders one real pickup_table motion headless and checks the contract the
# replay path promises: 43-DOF trajectory, two interpolated video frames per
# motion frame, cameras that actually move through the clip, and columns whose
# order matches the articulation the replay drives.
#
#   tests/test_grail_replay_clip.sh [sequence-key]
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="${1:-pickup_table__apple_0__000}"
LOG="$(mktemp)"
DATA_ROOT="$(sed -n 's/^HUMANOID_DATA_ROOT=//p' .env 2>/dev/null | tail -1)"
DATA_ROOT="${DATA_ROOT:-data}"

# A stale run must not be able to satisfy the checks below.
rm -rf "${DATA_ROOT}/outputs/grail-replay/$KEY"
./dev.sh isaac-g1 grail-replay "$KEY" --headless 2>&1 | tee "$LOG"

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
import hashlib, json, os, pathlib
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
print("OK trajectory: %s source frames @ %g Hz -> %s render frames @ %g Hz, dof shape %s"
      % (source_frames, manifest["fps"], frames, manifest["render_fps"], shape))
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
PY
' -- "$KEY"
