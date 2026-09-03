#!/usr/bin/env bash
# shell.sh — interactive shell in the container.
# Installed to /opt/humanoid-lab/shell.sh by Dockerfile COPY;
# run by compose.yaml command and dev.sh.
set -euo pipefail

source /opt/humanoid-lab/entrypoint.sh

echo "humanoid-lab dev container"
echo "  env selectors: use-isaac-sonic | use-sonic-sim | use-groot | use-none"
echo "  status:        show-env"
exec /bin/bash
