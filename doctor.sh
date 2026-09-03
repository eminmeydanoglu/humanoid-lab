#!/usr/bin/env bash
# doctor.sh — host + container validation report.
# Human-readable summary + timestamped JSON/YAML under ${HUMANOID_DATA_ROOT}/diagnostics.
# Exit: 0 = pass, 1 = warnings (usable), 2 = blocking error.
set -euo pipefail
cd "$(dirname "$0")"

# Host-only mode when .env is missing (report is still produced).
ENV_FILE=""
[ -f .env ] && { set -a; source .env; set +a; ENV_FILE=".env"; }

DATA_ROOT="${HUMANOID_DATA_ROOT:-$HOME/humanoid-lab-data}"
DIAG="$DATA_ROOT/diagnostics"
mkdir -p "$DIAG"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$DIAG/doctor-$STAMP.json"
YML="$DIAG/doctor-$STAMP.yaml"
LOG="$DIAG/doctor-$STAMP.log"

exec > >(tee -a "$LOG") 2>&1

rc=0
warn() { echo "  [warn] $*"; [ "$rc" -lt 1 ] && rc=1; return 0; }
err()  { echo "  [fail] $*"; rc=2; return 0; }

echo "humanoid-lab doctor — $STAMP"
echo "================================================"

echo; echo "== HOST =="
echo "OS: $(. /etc/os-release && echo "$PRETTY_NAME") ($(uname -m))"
echo "kernel: $(uname -r)"
echo "hostname: $(hostname)"
echo "RAM: $(free -g | awk '/Mem:/{print $2" GiB"}')  CPU: $(nproc) logical"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while IFS=',' read -r n m d; do
    echo "GPU: $n | VRAM: $m | driver: $d"
  done
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  BASELINE=$(python3 - <<'PY'
import yaml
print(yaml.safe_load(open('versions.lock.yaml'))['host']['nvidia_driver_tested_baseline'])
PY
)
  if [ "$(printf '%s\n%s\n' "$BASELINE" "$DRIVER" | sort -V | tail -1)" = "$DRIVER" ] && [ "$DRIVER" != "$BASELINE" ]; then
    echo "driver policy: $DRIVER >= baseline $BASELINE — keeping (no downgrade)"
  elif [ "$DRIVER" = "$BASELINE" ]; then
    echo "driver: $DRIVER == baseline $BASELINE"
  else
    err "driver $DRIVER BELOW BASELINE ($BASELINE)"
  fi
else
  err "nvidia-smi missing — NVIDIA driver/toolkit absent"
fi

echo; echo "== DOCKER TOOLCHAIN =="
for c in docker "docker compose" docker-buildx nvidia-ctk containerd; do
  if command -v "$c" >/dev/null 2>&1 || command -v "${c%% *}" >/dev/null 2>&1; then
    v=$( (eval "$c version" 2>/dev/null || eval "$c --version" 2>/dev/null) | head -1 ) || true
    [ -n "$v" ] && echo "$c: $v" || warn "$c version unreadable"
  else
    warn "$c not found"
  fi
done
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "runtime: $(docker info --format '{{.Name}} / {{.ServerVersion}}' 2>/dev/null)"
  docker info 2>/dev/null | grep -i "Runtimes:" | head -1 || true
else
  err "docker daemon unreachable"
fi
if command -v docker >/dev/null 2>&1 && docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then
  echo "docker --gpus all test: OK"
else
  err "docker --gpus all test failed (missing toolkit runtime config?)"
fi

if command -v docker >/dev/null 2>&1 && [ -f .env ] && docker compose --env-file .env config >/dev/null 2>&1; then
  echo "compose config: valid"
else
  warn "compose config not validated (docker or .env may be missing)"
fi

echo; echo "== CONTAINER =="
CID=$(docker compose --env-file .env ps -q dev 2>/dev/null || true)
if [ -n "$CID" ] && command -v docker >/dev/null 2>&1 && [ -n "$(docker ps -q --filter id="$CID" 2>/dev/null)" ]; then
  docker inspect "$CID" --format 'image: {{.Image}}' 2>/dev/null || true
  IMG=$(docker inspect "$CID" --format '{{.Image}}' 2>/dev/null)
  echo "container: $CID"
  docker image inspect "$IMG" --format 'RepoDigests: {{.RepoDigests}}' 2>/dev/null || true
  docker image inspect "$IMG" --format 'labels: {{json .Config.Labels}}' 2>/dev/null || true
  # mount contract: critical paths must be bind mounts (not named volumes / container layer)
  echo "-- mount table (critical paths) --"
  docker inspect "$CID" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Type}}, {{if .RW}}rw{{else}}ro{{end}}){{println}}{{end}}' | tee /tmp/hl-mounts.txt
  MISS=0
  for p in /data/datasets /data/checkpoints /data/models /data/rosbags /outputs /cache/huggingface /cache/uv /cache/isaac /data/diagnostics /data/runtime; do
    grep -q " -> $p " /tmp/hl-mounts.txt || { err "critical mount missing: $p"; MISS=1; }
  done
  [ "$MISS" = 0 ] && echo "all critical mounts present"
else
  warn "dev container not running — skipping in-container checks"
  err "mount contract not auditable (no container) — start it with './dev.sh' and re-run doctor"
fi

echo; echo "== DISK =="
df -h "$DATA_ROOT" | tail -1
for d in datasets checkpoints models hf-cache uv-cache isaac-cache rosbags outputs diagnostics; do
  [ -d "$DATA_ROOT/$d" ] && du -sh "$DATA_ROOT/$d" 2>/dev/null | sed "s|$DATA_ROOT/||"
done

if [ -n "$CID" ] && command -v docker >/dev/null 2>&1; then
  echo; echo "== ENV VERSIONS (container) =="
  docker compose --env-file .env exec -T dev bash -lc '
    source /opt/humanoid-lab/entrypoint.sh 2>/dev/null || true
    for env in isaac-sonic sonic-sim groot-n17; do
      v="/opt/venvs/$env"
      if [ -x "$v/bin/python" ]; then
        echo "-- $env --"
        "$v/bin/python" - <<PY
import sys
print("python:", sys.version.split()[0])
try:
    import torch; print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception: pass
try:
    import torchvision; print("torchvision:", torchvision.__version__)
except Exception: pass
try:
    import flash_attn; print("flash_attn:", flash_attn.__version__)
except Exception: pass
PY
      else
        echo "-- $env: not installed --"
      fi
    done
    ffmpeg -version 2>/dev/null | head -1 || true
  ' || true
fi

echo; echo "== GIT PINS =="
git -C . rev-parse --short HEAD 2>/dev/null || echo "no git repo"
git -C . status --porcelain 2>/dev/null | head -5 || true
echo "dirty: $(git -C . status --porcelain 2>/dev/null | wc -l) files"

# Token never written to the report.
echo; echo "== MODEL PROVENANCE =="
if [ -d "$DATA_ROOT/models" ]; then
  find "$DATA_ROOT/models" -maxdepth 2 -name MODEL_PROVENANCE.json | while read -r f; do
    python3 -c "import json;d=json.load(open('$f'));print(d['repo'], d['revision'], f\"{len(d['files'])} files\")" 2>/dev/null || echo "$f (unreadable)"
  done
else
  echo "no model dir"
fi

# Informational only; never writes to the robot network.
echo; echo "== ROS/DDS (info) =="
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-unset}  RMW: ${RMW_IMPLEMENTATION:-unset}"
for ipath in /sys/class/net/*; do
  dev=$(basename "$ipath")
  [ "$dev" = "lo" ] && continue
  state=$(cat "$ipath/operstate" 2>/dev/null || echo "?")
  echo "NIC $dev: $state"
done

# NOTE: `exit $rc` can be lost under the tee subprocess; close the stream first,
# then exit normally at the very end (and give tee a moment).

TMPD="${TMPDIR:-/tmp}"
cat > "$TMPD/hl-host.json" <<EOF
{"os": "$( (. /etc/os-release && echo "$PRETTY_NAME") 2>/dev/null || echo unknown)",
 "kernel": "$(uname -r)", "hostname": "$(hostname)",
 "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)",
 "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)",
 "driver_baseline_ok": $([ "$(printf '%s\n%s\n' "$BASELINE" "$DRIVER" | sort -V | tail -1)" = "$DRIVER" ] && echo true || echo false)}
EOF

DOCKER_OK=0
command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && DOCKER_OK=1
cat > "$TMPD/hl-docker.json" <<EOF
{"installed": $([ "$DOCKER_OK" = 1 ] && echo true || echo false),
 "compose_ok": $([ "$DOCKER_OK" = 1 ] && [ -f .env ] && docker compose --env-file .env config >/dev/null 2>&1 && echo true || echo false),
 "gpus_ok": $([ "$DOCKER_OK" = 1 ] && docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1 && echo true || echo false)}
EOF

CID=""
[ "$DOCKER_OK" = 1 ] && CID=$(docker compose --env-file .env ps -q dev 2>/dev/null || true)
IMG=""
[ -n "$CID" ] && IMG=$(docker inspect "$CID" --format '{{.Image}}' 2>/dev/null || true)
# mounts is null (not an empty list) when there is no container — "not audited".
MOUNTS="null"
MISSING="null"
if [ -n "$CID" ] && [ "$DOCKER_OK" = 1 ]; then
  M=$(docker inspect "$CID" --format '{{range .Mounts}}{{.Source}}|{{.Destination}}|{{.Type}}|{{if .RW}}rw{{else}}ro{{end}}{{println}}{{end}}' 2>/dev/null | awk -F'|' '{printf "{\"source\":\"%s\",\"target\":\"%s\",\"type\":\"%s\",\"rw\":\"%s\"},", $1,$2,$3,$4}' | sed 's/,$//')
  if [ -n "$M" ]; then MOUNTS="[$M]"; fi
  docker inspect "$CID" --format '{{range .Mounts}}{{println .Destination}}{{end}}' 2>/dev/null | sort > "$TMPD/hl-dests.txt" || true
  MISS=""
  for p in /data/datasets /data/checkpoints /data/models /data/rosbags /outputs /cache/huggingface /cache/uv /cache/isaac /data/diagnostics /data/runtime; do
    grep -qx "$p" "$TMPD/hl-dests.txt" || MISS="$MISS\"$p\","
  done
  MISS=$(echo "$MISS" | sed 's/,$//')
  if [ -n "$MISS" ]; then MISSING="[$MISS]"; else MISSING="[]"; fi
fi
cat > "$TMPD/hl-container.json" <<EOF
{"id": "$CID", "image": "$IMG", "mounts": $MOUNTS, "mount_missing": $MISSING}
EOF

GITC=$(git -C . rev-parse --short HEAD 2>/dev/null || echo "")
DIRTY=$(git -C . status --porcelain 2>/dev/null | wc -l || echo 0)
cat > "$TMPD/hl-git.json" <<EOF
{"commit": "$GITC", "dirty": $DIRTY}
EOF

MODELS=""
if [ -d "$DATA_ROOT/models" ]; then
  MODELS=$(find "$DATA_ROOT/models" -maxdepth 2 -name MODEL_PROVENANCE.json 2>/dev/null | while read -r f; do
    python3 -c "import json,sys;d=json.load(open('$f'));print('{\"repo\":\"%s\",\"revision\":\"%s\",\"files\":%d},'%(d['repo'],d['revision'],len(d['files'])))" 2>/dev/null
  done | sed 's/,$//')
fi
if [ -n "$MODELS" ]; then MODELS="[$MODELS]"; else MODELS="[]"; fi
cat > "$TMPD/hl-models.json" <<EOF
$MODELS
EOF

NICS=""
for ipath in /sys/class/net/*; do
  dev=$(basename "$ipath"); [ "$dev" = "lo" ] && continue
  st=$(cat "$ipath/operstate" 2>/dev/null || echo "?")
  NICS="$NICS{\"name\":\"$dev\",\"state\":\"$st\"},"
done
NICS=$(echo "$NICS" | sed 's/,$//')
[ -z "$NICS" ] && NICS="[]" || NICS="[$NICS]"
cat > "$TMPD/hl-dds.json" <<EOF
{"ros_domain_id": "${ROS_DOMAIN_ID:-unset}", "rmw": "${RMW_IMPLEMENTATION:-unset}", "nics": $NICS}
EOF

DISK_JSON=$(df -h "$DATA_ROOT" 2>/dev/null | tail -1 | awk '{printf "{\"fs\":\"%s\",\"size\":\"%s\",\"used\":\"%s\",\"avail\":\"%s\",\"use_pct\":\"%s\",\"mounted\":\"%s\"}", $1,$2,$3,$4,$5,$6}')
[ -z "$DISK_JSON" ] && DISK_JSON="{}"
cat > "$TMPD/hl-disk.json" <<EOF
$DISK_JSON
EOF

python3 scripts/doctor-report.py \
  --out-json "$OUT" --out-yaml "$YML" --rc "$rc" \
  --host-data "$TMPD/hl-host.json" --docker-data "$TMPD/hl-docker.json" \
  --container-data "$TMPD/hl-container.json" --git-data "$TMPD/hl-git.json" \
  --models-data "$TMPD/hl-models.json" --dds-data "$TMPD/hl-dds.json" \
  --disk-data "$TMPD/hl-disk.json"

{ echo; echo "================================================";
  echo "report: $OUT"; echo "log  : $LOG";
  if [ "$rc" = 0 ]; then echo "doctor: PASS"; elif [ "$rc" = 1 ]; then echo "doctor: PASS (with warnings)"; else echo "doctor: FAIL"; fi
} 2>/dev/null

sleep 0.2 2>/dev/null || true
exit "$rc"