#!/usr/bin/env bash
set -euo pipefail

/opt/humanoid-lab/bootstrap-venvs.sh

# shellcheck source=containers/entrypoint.sh
source /opt/humanoid-lab/entrypoint.sh

echo "humanoid-lab dev container"
echo "  env selectors: use-isaac-sonic | use-sonic-sim | use-groot | use-none"
echo "  status:        show-env"
exec /bin/bash
