#!/usr/bin/env bash
set -euo pipefail

source /opt/humanoid-lab/entrypoint.sh

echo "humanoid-lab dev container"
echo "  env selectors: use-isaac-sonic | use-sonic-sim | use-groot | use-none"
echo "  status:        show-env"
exec /bin/bash
