#!/usr/bin/env python3
"""Gercek G1 MuJoCo asset'i icin gorsel viewer; policy baglantisi ayri process."""
from __future__ import annotations

import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

root = Path(os.environ.get("G1_ASSET_ROOT", "/data/runtime/g1-mujoco-real"))
xml = root / "scene_43dof.xml"
if not xml.is_file():
    raise SystemExit(f"real G1 scene is missing: {xml}")

model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
dt = model.opt.timestep
realtime = float(os.environ.get("MUJOCO_REALTIME", "1.0"))
show_ui = os.environ.get("MUJOCO_SHOW_UI", "0") == "1"
max_catchup_steps = int(os.environ.get("MUJOCO_MAX_CATCHUP_STEPS", "64"))
print(
    f"G1 viewer: dt={dt:.4f}s, realtime={realtime:.2f}x, "
    f"ui={show_ui}; close the viewer window to stop."
)

with mujoco.viewer.launch_passive(model, data, show_left_ui=show_ui, show_right_ui=show_ui) as viewer:
    next_physics_step = time.perf_counter()
    while viewer.is_running():
        # sync() display-vsync-bound; physics adimlari once yetisir
        now = time.perf_counter()
        steps = 0
        while next_physics_step <= now and steps < max_catchup_steps:
            mujoco.mj_step(model, data)
            next_physics_step += dt / realtime
            steps += 1
        if steps == max_catchup_steps:
            next_physics_step = now
        viewer.sync()
        delay = next_physics_step - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
