#!/usr/bin/env bash
set -uo pipefail

PASS=0; WARN=0; BLOCK=0; FAIL=0
note() { printf '  [%-7s] %s\n' "$1" "$2"; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

PY_ISAAC=/opt/venvs/isaac-sonic/bin/python
PY_SIM=/opt/venvs/sonic-sim/bin/python
PY_GROOT=/opt/venvs/groot-n17/bin/python

MODEL_ROOT="${HUMANOID_DATA_ROOT:-/data}/models"
GROOT_MODEL_DIR="${GROOT_MODEL_DIR:-$MODEL_ROOT/groot_n17_base}"

if have_cmd nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  note PASS "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  PASS=$((PASS+1))
else
  note FAIL "nvidia-smi not working (missing host driver/toolkit?)"
  FAIL=$((FAIL+1))
fi

for env_name in isaac-sonic sonic-sim groot-n17; do
  venv="/opt/venvs/$env_name"
  if [ -x "$venv/bin/python" ]; then
    note PASS "$env_name: python $($venv/bin/python --version 2>&1)"
    PASS=$((PASS+1))
  else
    note FAIL "$env_name: interpreter not installed ($venv/bin/python)"
    FAIL=$((FAIL+1))
  fi
done
HAVE_ISAAC=$([ -x "$PY_ISAAC" ] && echo 1 || echo 0)
HAVE_SIM=$([ -x "$PY_SIM" ] && echo 1 || echo 0)
HAVE_GROOT=$([ -x "$PY_GROOT" ] && echo 1 || echo 0)

if [ "$HAVE_ISAAC" = 1 ]; then
  if "$PY_ISAAC" - <<'PY' >/dev/null 2>&1; then
import torch
assert torch.__version__.startswith("2.7.0"), torch.__version__
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.rand(128, 128, device="cuda"); y = x @ x.T
torch.cuda.synchronize()
import isaaclab, gear_sonic
print(f"torch={torch.__version__} cuda={torch.version.cuda} isaaclab={isaaclab.__version__}")
PY
    note PASS "isaac-sonic: torch 2.7.0 CUDA matmul + isaaclab/gear_sonic imports"
    PASS=$((PASS+1))
  else
    note FAIL "isaac-sonic: CUDA matmul or isaaclab/gear_sonic import failed:"
    "$PY_ISAAC" -c 'import torch; assert torch.cuda.is_available(); import isaaclab, gear_sonic' 2>&1 | tail -5 | sed 's/^/           /'
    FAIL=$((FAIL+1))
  fi
else
  note FAIL "isaac-sonic: env missing — cannot test"
  FAIL=$((FAIL+1))
fi

if [ "$HAVE_SIM" = 1 ]; then
  if MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY_SIM" - <<'PY' >/dev/null 2>&1; then
import pathlib
import mujoco
import gear_sonic
probe = mujoco.MjModel.from_xml_string(
    '<mujoco model="sonic-smoke"><worldbody><body name="g1_probe" pos="0 0 1">'
    '<freejoint/><geom type="sphere" size="0.05"/></body></worldbody></mujoco>')
data = mujoco.MjData(probe)
mujoco.mj_step(probe, data)
renderer = mujoco.Renderer(probe, height=64, width=64)
mujoco.mj_forward(probe, data)
renderer.update_scene(data)
_ = renderer.render()
renderer.close()
g1 = pathlib.Path(gear_sonic.__file__).parent / "data/robot_model/model_data/g1/scene_43dof.xml"
assert g1.is_file(), f"missing G1 scene: {g1}"
g1_model = mujoco.MjModel.from_xml_path(str(g1))
g1_data = mujoco.MjData(g1_model)
mujoco.mj_step(g1_model, g1_data)
import torch
print(f"mujoco={mujoco.__version__} torch={torch.__version__} "
      f"g1_nq={g1_model.nq} g1_nu={g1_model.nu} meshes={g1_model.nmesh}")
PY
    note PASS "sonic-sim: MuJoCo EGL render + real G1 asset step + gear_sonic"
    PASS=$((PASS+1))
  else
    note FAIL "sonic-sim: MuJoCo/G1 asset test failed:"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY_SIM" -c '
import pathlib, mujoco, gear_sonic
g1 = pathlib.Path(gear_sonic.__file__).parent / "data/robot_model/model_data/g1/scene_43dof.xml"
m = mujoco.MjModel.from_xml_path(str(g1))' 2>&1 | tail -5 | sed 's/^/           /'
    FAIL=$((FAIL+1))
  fi
  if "$PY_SIM" -c 'import gear_sonic.scripts.run_sim_loop' >/dev/null 2>&1; then
    note PASS "sonic-sim: gear_sonic.scripts.run_sim_loop import OK"
    PASS=$((PASS+1))
  else
    note FAIL "sonic-sim: run_sim_loop import failed (unitree_sdk2py/cyclonedds binding?)"
    FAIL=$((FAIL+1))
  fi
else
  note FAIL "sonic-sim: env missing — cannot test"
  FAIL=$((FAIL+1))
fi

if [ "$HAVE_GROOT" = 1 ]; then
  if "$PY_GROOT" - <<'PY' >/dev/null 2>&1; then
import torch, flash_attn
assert torch.__version__.startswith("2.9.0"), torch.__version__
import gr00t
print(f"torch={torch.__version__} flash_attn={flash_attn.__version__}")
PY
    note PASS "groot-n17: torch 2.9.0 + flash_attn + gr00t imports"
    PASS=$((PASS+1))
  else
    note FAIL "groot-n17: torch/flash_attn/gr00t import failed:"
    "$PY_GROOT" -c 'import torch, flash_attn, gr00t' 2>&1 | tail -5 | sed 's/^/           /'
    FAIL=$((FAIL+1))
  fi
else
  note FAIL "groot-n17: env missing — cannot test"
  FAIL=$((FAIL+1))
fi

if [ "$HAVE_GROOT" = 1 ]; then
  if [ ! -f "$GROOT_MODEL_DIR/config.json" ]; then
    note BLOCKED "GR00T N1.7 weights missing at $GROOT_MODEL_DIR — run ./dev.sh fetch-models"
    BLOCK=$((BLOCK+1))
  else
    if timeout 120 "$PY_GROOT" - "$GROOT_MODEL_DIR" >/dev/null 2>&1 <<'PY'; then
import sys
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
assert config.model_type.lower() in ("gr00tn1d7", "gr00t_n1d7"), config.model_type
print(f"local config OK: model_type={config.model_type}")
PY
      note PASS "groot-n17: N1.7 local checkpoint metadata loads (config.json + registry)"
      PASS=$((PASS+1))
    else
      note FAIL "groot-n17: N1.7 local checkpoint metadata failed to load from $GROOT_MODEL_DIR"
      FAIL=$((FAIL+1))
    fi
    LOAD_LOG=$(mktemp)
    if timeout 900 "$PY_GROOT" - "$GROOT_MODEL_DIR" >"$LOAD_LOG" 2>&1 <<'PY'; then
import sys, torch
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
model_dir = sys.argv[1]
model = Gr00tN1d7.from_pretrained(model_dir)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"model loaded: {n_params/1e9:.2f}B parameters")
PY
      note PASS "groot-n17: N1.7 model loaded from $GROOT_MODEL_DIR ($(tail -1 "$LOAD_LOG" | sed 's/^model loaded: //'))"
      PASS=$((PASS+1))
    else
      rc=$?
      if [ "$rc" = 124 ]; then
        note WARN "groot-n17: N1.7 model load timed out (900s)"
        WARN=$((WARN+1))
      elif grep -qiE "gated|401 Client Error|403 Client Error|Access to model .* is restricted" "$LOAD_LOG"; then
        note BLOCKED "groot-n17: N1.7 backbone (nvidia/Cosmos-Reason2-2B) is gated — accept the license on HF and run ./dev.sh hf-login, then re-run"
        BLOCK=$((BLOCK+1))
      else
        note FAIL "groot-n17: N1.7 model load failed from $GROOT_MODEL_DIR (rc=$rc):"
        tail -8 "$LOAD_LOG" | sed 's/^/           /'
        FAIL=$((FAIL+1))
      fi
    fi
    rm -f "$LOAD_LOG"
  fi
fi

if [ -f "$MODEL_ROOT/sonic/MODEL_PROVENANCE.json" ] && compgen -G "$MODEL_ROOT/sonic/sonic_v1_1/*.onnx" >/dev/null; then
  note PASS "SONIC model files present ($(find "$MODEL_ROOT/sonic" -type f | wc -l) files)"
  PASS=$((PASS+1))
else
  note BLOCKED "SONIC model missing at $MODEL_ROOT/sonic — run ./dev.sh fetch-models"
  BLOCK=$((BLOCK+1))
fi

if have_cmd ffmpeg; then
  FF=$(ffmpeg -version 2>/dev/null | head -1 | grep -oE 'ffmpeg version [0-9]+' | grep -oE '[0-9]+' || true)
  if [ -n "$FF" ] && [ "$FF" -lt 8 ]; then
    note PASS "ffmpeg $FF (torchcodec-compatible line)"
    PASS=$((PASS+1))
  else
    note FAIL "ffmpeg version $FF — FFmpeg 8 not usable (torchcodec==0.8.0)"
    FAIL=$((FAIL+1))
  fi
else
  note FAIL "ffmpeg missing"
  FAIL=$((FAIL+1))
fi

printf '\nsummary: PASS=%d WARN=%d BLOCKED=%d FAIL=%d\n' "$PASS" "$WARN" "$BLOCK" "$FAIL"
[ "$FAIL" = 0 ] || exit 1
exit 0
