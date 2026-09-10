#!/usr/bin/env python3
"""Launch the SONIC MuJoCo sim on DDS domain 42 + interface 'lo'.

The staged C++ deploy (data/models/sonic-deploy) is patched to DDS domain 42 so
it can never collide with a physical G1 (domain 0). The upstream MuJoCo sim
defaults to domain 0, so we monkeypatch its WBC config at runtime (no upstream
file is modified) to join the same domain/interface.

Run in the container (sonic-sim env):
  python sim_mujoco_domain42.py --interface sim --simulator mujoco \
      --keyboard-dispatcher-type raw --enable-onscreen
"""

import sys

sys.path.insert(0, "/opt/src/sonic")

from gear_sonic.utils.mujoco_sim import configs as _configs  # noqa: E402

SIM_DDS_DOMAIN_ID = 42
SIM_DDS_INTERFACE = "lo"

_orig_load_wbc_yaml = _configs.BaseConfig.load_wbc_yaml


def _load_wbc_yaml_domain42(self):
    wbc = _orig_load_wbc_yaml(self)
    wbc["DOMAIN_ID"] = SIM_DDS_DOMAIN_ID
    if SIM_DDS_INTERFACE:
        wbc["INTERFACE"] = SIM_DDS_INTERFACE
    return wbc


_configs.BaseConfig.load_wbc_yaml = _load_wbc_yaml_domain42

import tyro  # noqa: E402
from gear_sonic.scripts import run_sim_loop  # noqa: E402

if __name__ == "__main__":
    print(f"[sim42] DDS domain={SIM_DDS_DOMAIN_ID} interface={SIM_DDS_INTERFACE}")
    cfg = tyro.cli(run_sim_loop.ArgsConfig)
    run_sim_loop.main(cfg)
