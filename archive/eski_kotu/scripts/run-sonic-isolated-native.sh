#!/usr/bin/env bash
set -euo pipefail

container=${SONIC_ISOLATED_CONTAINER:-humanoid-lab-dev}
project_container=${SONIC_ISOLATED_PROJECT_ROOT:-/workspace/humanoid-lab}
sonic_root=${SONIC_ROOT:-/opt/src/sonic}
models=${SONIC_MODELS_ROOT:-/data/runtime/sonic-deploy-models/sonic_v1_1}

docker inspect "$container" >/dev/null
docker exec \
  -e PROJECT_ROOT="$project_container" \
  -e SONIC_ROOT="$sonic_root" \
  -e ENCODER_MODEL="$models/model_encoder.onnx" \
  -e DECODER_MODEL="$models/model_decoder.onnx" \
  -e SAFE_REFERENCE="$project_container/tests/fixtures/sonic_isolated_standing_reference.json" \
  -e LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.20.1/lib:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu \
  "$container" sh -lc '
    set -eu
    "$PROJECT_ROOT/scripts/build-sonic-isolated-native.sh"
    exec /tmp/sonic-isolated-native
  '
