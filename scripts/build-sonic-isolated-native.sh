#!/usr/bin/env bash
set -euo pipefail

: "${SONIC_ROOT:?set SONIC_ROOT to the pinned SONIC export}"
: "${ONNXRUNTIME_ROOT:=/opt/onnxruntime-linux-x64-1.20.1}"
: "${PROJECT_ROOT:=/workspace/humanoid-lab}"

harness=${SONIC_NATIVE_HARNESS:-isolated}
case "$harness" in
  isolated) source_file="$PROJECT_ROOT/tests/native/sonic_isolated_harness.cpp"; output=/tmp/sonic-isolated-native ;;
  closed-loop) source_file="$PROJECT_ROOT/tests/native/sonic_closed_loop_harness.cpp"; output=/tmp/sonic-closed-loop-native ;;
  *) echo "unsupported SONIC_NATIVE_HARNESS: $harness" >&2; exit 2 ;;
esac
subscriber="$SONIC_ROOT/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp"
compiler=/opt/cyclonedds/bin/c++

[[ -f "$source_file" && -f "$subscriber" && -x "$compiler" ]]
[[ -f "$ONNXRUNTIME_ROOT/include/onnxruntime_cxx_api.h" && -f "$ONNXRUNTIME_ROOT/lib/libonnxruntime.so" ]]
dpkg-query -W -f='${Status}\n' cppzmq-dev | grep -qx 'install ok installed'
"$compiler" -std=c++20 -O2 -Wall -Wextra -Werror \
  -I"$SONIC_ROOT/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include" \
  -I"$ONNXRUNTIME_ROOT/include" "$source_file" \
  -L"$ONNXRUNTIME_ROOT/lib" -lonnxruntime -lzmq -pthread -o "$output"
printf 'built %s\n' "$output"
