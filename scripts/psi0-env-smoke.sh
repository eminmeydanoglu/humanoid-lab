#!/usr/bin/env bash
# Psi0 training-environment smoke: interpreter, PyTorch build, upstream import
# stack, configuration schema and the released SONIC warm-start checkpoint.
# Runs no dataset and no optimizer step.
set -euo pipefail

readonly EXPECTED_COMMIT="${PSI0_EXPECTED_COMMIT:-4f3720d45e102b36d7c3e9465ab8062274170518}"
readonly SOURCE_DIR="${PSI0_SOURCE_DIR:-/workspace/humanoid-lab/third_party/Psi0}"
readonly PYTHON_BIN="${PSI0_PYTHON:-/opt/venvs/psi0/bin/python}"
readonly PSI_HOME_DIR="${PSI_HOME:-/hfm}"
readonly CKPT_DIR="${PSI0_CKPT_DIR:-$PSI_HOME_DIR/cache/checkpoints/psi0/postpre.sonic1.0.unifolm.2609092156.40k}"
readonly DREAM_DIR="${PSI0_DREAM_DIR:-$PSI_HOME_DIR/cache/checkpoints/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156}"
readonly LOG_DIR="${PSI0_SMOKE_LOG_DIR:-/outputs/psi0-env-smoke}"
readonly CONFIG_MODULE="${PSI0_CONFIG_MODULE:-finetune_real_psi0_config}"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

blocked() {
  echo "BLOCKED: $*" >&2
  exit 3
}

require_file() {
  [[ -s "$1" ]] || fail "$2 is missing or empty: $1"
}

check_source() {
  [[ -d "$SOURCE_DIR" ]] || fail "Psi0 checkout is missing: $SOURCE_DIR (git submodule update --init third_party/Psi0)"
  local actual
  actual="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null)" || fail "cannot read the Psi0 revision at $SOURCE_DIR"
  [[ "$actual" == "$EXPECTED_COMMIT" ]] || fail "Psi0 revision is $actual; expected $EXPECTED_COMMIT"
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]] || \
    fail "Psi0 checkout has tracked modifications: $SOURCE_DIR"
  echo "[ok] source   $SOURCE_DIR @ ${actual:0:12}"
}

check_interpreter() {
  [[ -x "$PYTHON_BIN" ]] || fail "psi0 interpreter is missing: $PYTHON_BIN (run ./dev.sh sync)"
  local version
  version="$("$PYTHON_BIN" --version 2>&1)"
  [[ "$version" == "Python 3.11."* ]] || fail "psi0 interpreter is $version; upstream pins 3.11"
  echo "[ok] python   $version ($PYTHON_BIN)"
}

check_stack() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util

import torch

assert torch.__version__.startswith("2.7.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable to the psi0 environment"

import psi

from psi.config.train.finetune_real_psi0_config import DynamicLaunchConfig
from psi.data.lerobot.compat import LEROBOT_LAYOUT
from psi.trainers import Trainer
from accelerate import Accelerator, DeepSpeedPlugin  # noqa: F401  (accelerate imports deepspeed when installed)
import transformers

# The image ships no CUDA toolkit, and deepspeed's package init calls
# installed_cuda_version(): keeping it installed breaks `import accelerate`.
assert importlib.util.find_spec("deepspeed") is None, "deepspeed is installed and unimportable in this runtime"

name = torch.cuda.get_device_name(0)
arch = torch.cuda.get_arch_list()
assert any("sm_120" in entry for entry in arch), f"torch build has no sm_120 kernels: {arch}"
print(f"[ok] psi      {psi.__version__} / transformers {transformers.__version__} / lerobot layout {LEROBOT_LAYOUT if isinstance(LEROBOT_LAYOUT, str) else type(LEROBOT_LAYOUT).__name__}")
print(f"[ok] torch    {torch.__version__} (cuda {torch.version.cuda}) on {name}")
print(f"[ok] arch     {arch}")
print("[ok] trainer  accelerate + psi.trainers + config schema import cleanly")
PY
}

check_config_cli() {
  local log="$LOG_DIR/config-cli.log"
  require_file "$SOURCE_DIR/.env" "Psi0 .env (PSI_HOME and HF/W&B settings)"
  if ! (cd "$SOURCE_DIR" && timeout 600 "$PYTHON_BIN" scripts/train.py "$CONFIG_MODULE" --help) >"$log" 2>&1; then
    tail -20 "$log" >&2
    fail "the training entry point cannot resolve $CONFIG_MODULE; see $log"
  fi
  # tyro prints nested flags in dashed form but accepts both spellings.
  while IFS='|' read -r name pattern; do
    grep -Eq -- "$pattern" "$log" || fail "resolved configuration is missing $name; see $log"
  done <<'EOF'
--data.root_dir|--data\.root[-_]dir
--model.action-chunk-size|--model\.action[-_]chunk[-_]size
--train.train_batch_size|--train\.train[-_]batch[-_]size
EOF
  echo "[ok] config   scripts/train.py $CONFIG_MODULE --help ($(wc -l <"$log") lines)"
}

check_checkpoint() {
  local f
  for f in config.json model.safetensors action_header.safetensors; do
    require_file "$CKPT_DIR/$f" "warm-start checkpoint file"
  done
  "$PYTHON_BIN" - "$CKPT_DIR" "${PSI0_SKIP_HASHES:-0}" <<'PY'
import hashlib
import json
import os
import sys

from safetensors import safe_open

root, skip_hashes = sys.argv[1], sys.argv[2] == "1"

config = json.load(open(os.path.join(root, "config.json"), encoding="utf-8"))
assert config["architectures"] == ["Qwen3VLForConditionalGeneration"], config["architectures"]

for name in ("model.safetensors", "action_header.safetensors"):
    path = os.path.join(root, name)
    with safe_open(path, framework="pt") as handle:
        print(f"[ok] ckpt     {name}: {len(handle.keys())} tensors, "
              f"{os.path.getsize(path) / 1e9:.2f} GB")

provenance = os.path.join(root, "MODEL_PROVENANCE.json")
if not os.path.exists(provenance):
    print("[--] skeleton MODEL_PROVENANCE.json absent; hashes not verified")
elif skip_hashes == "1":
    print("[--] hash verification skipped (PSI0_SKIP_HASHES=1)")
else:
    entries = json.load(open(provenance, encoding="utf-8"))["files"]
    for entry in entries:
        path = os.path.join(root, entry["path"])
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"], f"sha256 mismatch: {entry['path']}"
    print(f"[ok] hashes   {len(entries)} files match MODEL_PROVENANCE.json")
PY
  echo "[ok] ckpt     $CKPT_DIR"
}

# The released multi-task checkpoint the unified evaluation UI can serve as its
# ψ-Dream option.  It is a graded artifact: absent means "not downloaded", which
# is a note, and a present-but-broken tree is a failure.
check_dream_checkpoint() {
  if [[ ! -d "$DREAM_DIR" ]]; then
    echo "[--] dream    not downloaded (./dev.sh fetch-psi0-ckpt psi0_sonic_dream)"
    return 0
  fi
  local f
  for f in run_config.json argv.txt checkpoints/ckpt_40000/model.safetensors; do
    require_file "$DREAM_DIR/$f" "ψ-Dream run file"
  done
  "$PYTHON_BIN" - "$DREAM_DIR" <<'PY'
import json
import os
import sys

from safetensors import safe_open

root = sys.argv[1]
config = json.load(open(os.path.join(root, "run_config.json"), encoding="utf-8"))
repack = config["data"]["transform"]["repack"]
image_keys = repack["image_keys"]
assert len(image_keys) == 1, f"the bridge serves one camera; run declares {image_keys}"
resize = config["data"]["transform"]["model"]["resize"]["size"]
ckpt = os.path.join(root, "checkpoints", "ckpt_40000", "model.safetensors")
with safe_open(ckpt, framework="pt") as handle:
    names = list(handle.keys())
prefixes = {name.split(".")[0] for name in names}
assert {"vlm_model", "action_header"} <= prefixes, f"deploy loader prefixes missing: {sorted(prefixes)}"
print(f"[ok] ckpt     ψ-Dream: {len(names)} tensors, {os.path.getsize(ckpt) / 1e9:.2f} GB, "
      f"camera {image_keys[0]}, resize {resize}")
PY
  echo "[ok] ckpt     $DREAM_DIR"
}

mkdir -p "$LOG_DIR"
echo "psi0 environment smoke -> $LOG_DIR"
check_source
check_interpreter
check_stack
check_config_cli
check_checkpoint
check_dream_checkpoint
echo "PASS: psi0 environment and released checkpoints are usable"
