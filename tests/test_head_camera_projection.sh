#!/usr/bin/env bash
# Renderer-behaviour probe for the head camera.
#
# A profile declares the head camera's horizontal angle; Isaac Sim projects
# square pixels, so the vertical angle follows from the image size. This script
# measures that instead of assuming it: it renders one motion a second time at
# 640x360 and checks that the smaller frame is the centre crop of the 640x480
# frame. A vertical squeeze of the whole picture would mean the renderer derives
# the vertical angle from something else, and the camera contract would have to
# change with it.
#
# Run the clip acceptance first so the 640x480 reference exists:
#
#   tests/test_grail_replay_clip.sh [sequence-key]
#   tests/test_head_camera_projection.sh [sequence-key]
#
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="${1:-pickup_table__apple_0__000}"
DATA_ROOT="$(sed -n 's/^HUMANOID_DATA_ROOT=//p' .env 2>/dev/null | tail -1)"
DATA_ROOT="${DATA_ROOT:-data}"
DATA_ROOT_ABS="$(cd "$DATA_ROOT" && pwd)"
STAMP="$$"
# The container mounts the data root piecewise (/data/*, /outputs), so the probe
# profile has to live beside the shipped ones to be readable there. It is
# generated for this run only and removed on any exit.
PROFILE="configs/profiles/zz-head-camera-projection-probe.$STAMP.json"
PROFILE_CONTAINER="/workspace/humanoid-lab/$PROFILE"
BASE="$DATA_ROOT_ABS/outputs/grail-replay/$KEY"
OUT_CONTAINER="/outputs/grail-replay/zz-projection-probe.$STAMP"

cleanup() { rm -f "$PROFILE"; rm -rf "$DATA_ROOT_ABS/outputs/grail-replay/zz-projection-probe.$STAMP"; }
trap cleanup EXIT

[ -f "$BASE/manifest.json" ] || {
  echo "error: no 640x480 reference at $BASE — run tests/test_grail_replay_clip.sh $KEY first" >&2
  exit 2
}

# The reference has to be a completed run of this contract, not a stale clip.
python3 - "$BASE/manifest.json" "$KEY" <<'PY'
import json, sys

manifest = json.loads(open(sys.argv[1]).read())
key = sys.argv[2]
assert manifest["result"] == "COMPLETED", manifest["result"]
assert manifest["sequence_key"] == key, manifest["sequence_key"]
camera = manifest["cameras"]["head"]
assert camera["resolution"] == [640, 480], camera["resolution"]
rendered_h, rendered_v = camera["rendered_fov_deg"]
assert abs(rendered_h - 54.9) < 1e-9, rendered_h
assert abs(rendered_v - 42.5712) < 1e-3, rendered_v
print("OK reference: %s, %s frames, rendered %g x %g deg"
      % (key, manifest["render_frames"], rendered_h, rendered_v))
PY

python3 - "$PROFILE" <<'PY'
import json, pathlib, sys

source = json.loads(pathlib.Path("configs/profiles/isaac-g1-dex3.json").read_text())
source["profile_id"] = "isaac-g1-29dof-dex3-projection-probe"
source["camera"]["resolution"] = [640, 360]
source["camera"]["provenance"] = "probe: is the vertical angle the image-size one?"
pathlib.Path(sys.argv[1]).write_text(json.dumps(source, indent=2) + "\n")
PY

docker compose --env-file .env exec -T dev bash -lc '
  source /opt/humanoid-lab/entrypoint.sh >/dev/null 2>&1
  use-isaac-sonic >/dev/null 2>&1
  mkdir -p /tmp/humanoid-lab-kit-cwd && cd /tmp/humanoid-lab-kit-cwd
  exec 9>/tmp/humanoid-lab-isaac-g1.lock
  flock -n 9 || { echo "error: another Isaac G1 run holds the lock" >&2; exit 3; }
  python /workspace/humanoid-lab/scripts/run-isaac-g1.py \
    --profile "$1" --replay "$2" --replay-output-dir "$3" --headless
' -- "$PROFILE_CONTAINER" "$KEY" "$OUT_CONTAINER" >/dev/null

docker compose --env-file .env exec -T dev bash -lc '
  source /opt/humanoid-lab/entrypoint.sh >/dev/null 2>&1
  use-isaac-sonic >/dev/null 2>&1
  KEY="$1" PROBE="$2" python - <<"PY"
import json, os, pathlib
import imageio.v2 as imageio
import numpy as np

key = os.environ["KEY"]
base = pathlib.Path("/outputs/grail-replay") / key
probe = pathlib.Path(os.environ["PROBE"])
reference = json.loads((base / "manifest.json").read_text())
manifest = json.loads((probe / "manifest.json").read_text())
assert manifest["result"] == "COMPLETED", manifest["result"]
assert manifest["sequence_key"] == key, manifest["sequence_key"]
camera = manifest["cameras"]["head"]
assert camera["resolution"] == [640, 360], camera
frames = reference["render_frames"]
assert manifest["render_frames"] == frames, (manifest["render_frames"], frames)


def read(path, index):
    return imageio.get_reader(path).get_data(index).astype(np.float64)


def bilinear(frame, shape):
    rows = (np.arange(shape[0]) + 0.5) * frame.shape[0] / shape[0] - 0.5
    cols = (np.arange(shape[1]) + 0.5) * frame.shape[1] / shape[1] - 0.5
    rows = np.clip(rows, 0, frame.shape[0] - 1)
    cols = np.clip(cols, 0, frame.shape[1] - 1)
    r0, c0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    r1 = np.minimum(r0 + 1, frame.shape[0] - 1)
    c1 = np.minimum(c0 + 1, frame.shape[1] - 1)
    wr, wc = (rows - r0)[:, None, None], (cols - c0)[None, :, None]
    top = frame[r0][:, c0] * (1 - wc) + frame[r0][:, c1] * wc
    bottom = frame[r1][:, c0] * (1 - wc) + frame[r1][:, c1] * wc
    return top * (1 - wr) + bottom * wr


for index in (0, frames // 2):
    wide, small = read(base / "head.mp4", index), read(probe / "head.mp4", index)
    top = (wide.shape[0] - small.shape[0]) // 2
    crop = wide[top : top + small.shape[0]]
    squeezed = bilinear(wide, small.shape[:2])
    crop_diff = float(np.abs(small - crop).mean())
    squeeze_diff = float(np.abs(small - squeezed).mean())
    print("frame %3d: 640x360 vs centre crop %.3f | vs vertical squeeze %.3f"
          % (index, crop_diff, squeeze_diff))
    assert crop_diff < 4.0, "the smaller render is not a centre crop: %.3f" % crop_diff
    assert crop_diff < squeeze_diff, "vertical squeeze fits at least as well: %.3f vs %.3f" % (crop_diff, squeeze_diff)
print("OK projection: the vertical angle is the image-size one (square pixels), not an authored aperture")
PY
' -- "$KEY" "$OUT_CONTAINER"
