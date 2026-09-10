#!/usr/bin/env bash
set -euo pipefail

if [[ $(id -u) -ne 0 ]]; then
  echo 'run this setup script as root inside the disposable development container' >&2
  exit 1
fi
apt-get update
apt-get install -y --no-install-recommends python3-numpy python3-zmq python3-pip
python3 -m pip install --break-system-packages 'tyro==0.9.35'
python3 - <<'PY'
import numpy
import tyro
import zmq
print(f"python prerequisites numpy={numpy.__version__} pyzmq={zmq.__version__} tyro={tyro.__version__}")
PY
