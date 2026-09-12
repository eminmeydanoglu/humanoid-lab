#!/usr/bin/env bash
# Start the ER web UI detached, so it survives this shell.
#   webapp/serve.sh            # port 8321
#   PORT=9000 webapp/serve.sh  # another port
set -euo pipefail

DATA_ROOT="${HUMANOID_DATA_ROOT:-/home/aksoy-msi/code/humanoid-lab-main/data}"
PYTHON="$DATA_ROOT/venvs/unifolm-wla/bin/python"
PORT="${PORT:-8321}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-/tmp/er_web.log}"

if ss -tln | grep -q ":$PORT "; then
  echo "port $PORT zaten dinleniyor; önce durdurun: ss -tlnp | grep $PORT"
  exit 1
fi

setsid nohup "$PYTHON" "$HERE/app.py" \
  --port "$PORT" --warm \
  --model-root "$DATA_ROOT/models/unifolm-wla-1.0" \
  --frames-dir "$DATA_ROOT/outputs/unifolm-wla-probe/frames" \
  < /dev/null > "$LOG" 2>&1 &

echo "$!" > "/tmp/er_web_$PORT.pid"
sleep 3
echo "pid $(cat /tmp/er_web_$PORT.pid), log $LOG"
echo "yerel:   http://localhost:$PORT/"
echo "tailnet: http://100.126.18.76:$PORT/"
echo "durdur:  kill \$(cat /tmp/er_web_$PORT.pid)"
