"""The simulator wrapper consumes display selection before invoking Isaac."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gui_is_consumed_before_simulator_arguments():
    shell = r'''
source <(sed -n '/^run_isaac_g1() {/,/^}/p' dev.sh)
DC() { printf 'fake-container\n'; }
docker() {
  if [ "$1" = exec ] && [ "$2" = -e ]; then
    printf 'ARG:%s\n' "${@: -2}"
  fi
}
cleanup_isaac_g1() { :; }
LIVESTREAM_ENV=()
DISPLAY=:0
run_isaac_g1 configs/profiles/pick-apple-askida.json --gui --duration 3
'''
    result = subprocess.run(["bash", "-c", shell], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["ARG:--duration", "ARG:3"]
