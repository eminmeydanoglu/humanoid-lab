#!/usr/bin/env bash
# Build the pinned upstream SONIC C++ deploy for the Isaac simulation path.
#
# Repo patches applied to the pinned source before building:
#   sonic-deploy-sim-domain.patch  - move DDS off domain 0 (the physical G1 default)
#   sonic-f310-bridge-input.patch  - add the --input-type f310_bridge adapter
#
# Select a subset with DEPLOY_PATCHES (space separated "patch:base" entries).
#
# Must run in an image providing TensorRT + ONNX Runtime + a C++20 compiler
# (humanoid-lab/dev:cloudwalk-sonic-native-* provides all three).
set -euo pipefail

SONIC_ROOT="${SONIC_ROOT:-/opt/src/sonic}"
PATCH_DIR="${PATCH_DIR:-/workspace/humanoid-lab/patches}"
BUILD_DIR="${BUILD_DIR:-$SONIC_ROOT/gear_sonic_deploy/build}"
BINARY="$SONIC_ROOT/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime-linux-x64-1.20.1}"
JOBS="${JOBS:-$(nproc)}"

# Each entry is "<patch file>:<base dir relative to SONIC_ROOT>". The domain
# patch is written against the SONIC root; the F310 patch against
# gear_sonic_deploy/.
read -r -a PATCHES <<< "${DEPLOY_PATCHES:-sonic-deploy-sim-domain.patch: sonic-f310-bridge-input.patch:gear_sonic_deploy}"

cd "$SONIC_ROOT"

want_f310=0
for entry in "${PATCHES[@]}"; do
  patch="${entry%%:*}"
  base="${entry##*:}"
  path="$PATCH_DIR/$patch"
  [ -f "$path" ] || { echo "error: missing patch $path" >&2; exit 2; }
  args=(-p1)
  [ -n "$base" ] && args+=(--directory="$base")
  if git apply "${args[@]}" --reverse --check "$path" >/dev/null 2>&1; then
    echo "[build-deploy] already applied: $patch"
  elif git apply "${args[@]}" --check "$path" >/dev/null 2>&1; then
    git apply "${args[@]}" "$path"
    echo "[build-deploy] applied: $patch"
  else
    echo "error: $patch neither applies nor is applied — the tree is not the pinned source" >&2
    exit 3
  fi
  [ "$patch" = "sonic-f310-bridge-input.patch" ] && want_f310=1
done

# A physical G1 defaults to DDS domain 0; the build must never leave that in.
grep -n 'ChannelFactory::Instance()->Init(42' \
  "$SONIC_ROOT/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp" >/dev/null \
  || { echo "error: deploy is not pinned to the simulation DDS domain" >&2; exit 4; }

mkdir -p "$BUILD_DIR"
# The vendored Findonnxruntime.cmake searches ENV onnxruntime_ROOT (and a fixed
# list that does not include the image's versioned onnxruntime directory).
export onnxruntime_ROOT="$ONNXRUNTIME_ROOT"
cmake -S "$SONIC_ROOT/gear_sonic_deploy" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_INCLUDE_DIR="$ONNXRUNTIME_ROOT/include" \
  -Donnxruntime_LIBRARY="$ONNXRUNTIME_ROOT/lib/libonnxruntime.so"

cmake --build "$BUILD_DIR" -j "$JOBS"

[ -x "$BINARY" ] || { echo "error: build produced no binary at $BINARY" >&2; exit 5; }
echo "[build-deploy] built: $BINARY"

if [ "$want_f310" = 1 ]; then
  # The bridge input type only exists once the patch is compiled in.
  if "$BINARY" 2>&1 | grep -q 'f310_bridge'; then
    echo "[build-deploy] f310_bridge input type is present"
  else
    echo "error: f310_bridge input type missing from the built binary" >&2
    exit 6
  fi
fi
