"""Compare the old sphere apple with the scanned mesh apple on a flat desk.

Headless, no robot.  Both apples sit side by side on one static desk (PhysX
default material, like the realism desk asset) and go through the same phases:

* rest  -- 3 s on the desk: drift and residual speed
* drop  -- released 5 cm above the desk: rebound height, settle time
* push  -- given 0.3 m/s along +x: travel, time to stop, spin at stop
* tilt  -- a horizontal force m*g*tan(theta), theta ramped 1 deg/s, which
            is a tilted desk up to the (irrelevant for Coulomb onset) gravity
            magnitude: angle at which each apple has moved 5 mm

Run inside the dev container:

    source /opt/humanoid-lab/entrypoint.sh; use-isaac-sonic
    python scripts/experiments/apple_physics_check.py --headless --out /outputs/realism/apple-physics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=Path, default=ROOT / "configs/profiles/pick-apple-askida-real.json")
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402

from humanoid_lab.simulators.isaac.contracts import RunProfile  # noqa: E402
from humanoid_lab.simulators.isaac.service import task_object_material  # noqa: E402

DT = 0.005
SURFACE = 0.8357609
profile = RunProfile.load(args.profile)
mesh_apple = profile.scene.object

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cpu"))
sim_utils.CuboidCfg(size=(1.2, 1.2, 0.025), collision_props=sim_utils.CollisionPropertiesCfg()).func(
    "/World/Desk", sim_utils.CuboidCfg(size=(1.2, 1.2, 0.025), collision_props=sim_utils.CollisionPropertiesCfg()),
    translation=(0.5, 0.0, SURFACE - 0.0125))

old_z = SURFACE + 0.0335
old = RigidObject(RigidObjectCfg(
    prim_path="/World/OldApple",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.25, old_z)),
    spawn=sim_utils.SphereCfg(
        radius=0.0335,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=2.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.13),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                                                        friction_combine_mode="max", restitution_combine_mode="min"))))
new_z = mesh_apple.position_m[2]
new = RigidObject(RigidObjectCfg(
    prim_path="/World/MeshApple",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.25, new_z)),
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "data" / Path(mesh_apple.asset_reference).relative_to("/data")),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=mesh_apple.angular_damping_1_s),
        mass_props=sim_utils.MassPropertiesCfg(mass=mesh_apple.mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg())))
material = task_object_material(mesh_apple)
material.func("/World/Materials/MeshApple", material)
sim_utils.bind_physics_material("/World/MeshApple", "/World/Materials/MeshApple")

sim.reset()
apples = {"old_sphere": (old, old_z, -0.25), "mesh": (new, new_z, 0.25)}


def place(z_offset=0.0, velocity=(0.0, 0.0, 0.0)):
    for obj, z, y in apples.values():
        state = obj.data.default_root_state.clone()
        state[:, 2] = z + z_offset
        state[:, 7:10] = torch.tensor(velocity)
        state[:, 10:13] = 0.0
        obj.write_root_state_to_sim(state)
        obj.reset()


def settle_to_rest():
    place()
    for _ in range(int(1.0 / DT)):
        step()
    rest = {}
    for name, (obj, _, _) in apples.items():
        rest[name] = obj.data.root_state_w[0].clone()
    return rest


def write_state(states):
    for name, (obj, _, _) in apples.items():
        obj.write_root_state_to_sim(states[name].unsqueeze(0))
        obj.reset()


def step():
    for obj, _, _ in apples.values():
        obj.write_data_to_sim()
    sim.step(render=False)
    for obj, _, _ in apples.values():
        obj.update(DT)


def pos(name):
    return apples[name][0].data.root_pos_w[0].clone()


def speed(name):
    return float(torch.linalg.norm(apples[name][0].data.root_lin_vel_w[0]))


def spin(name):
    return float(torch.linalg.norm(apples[name][0].data.root_ang_vel_w[0]))


report = {"mesh_apple": {k: getattr(mesh_apple, k) for k in (
    "mass_kg", "size_m", "static_friction", "dynamic_friction", "restitution", "friction_combine_mode",
    "angular_damping_1_s")},
    "old_sphere": {"mass_kg": 0.13, "diameter_m": 0.067, "friction": 1.0, "combine": "max", "angular_damping_1_s": 2.0},
    "desk_material": "PhysX default (0.5/0.5/0, average)"}

# rest
rest_states = settle_to_rest()
start = {n: pos(n) for n in apples}
for _ in range(int(3.0 / DT)):
    step()
report["rest"] = {n: {"drift_mm": round(float(torch.linalg.norm(pos(n) - start[n])) * 1e3, 3),
                      "speed_mm_s": round(speed(n) * 1e3, 3),
                      "rest_center_z_minus_surface_mm": round((float(pos(n)[2]) - SURFACE) * 1e3, 2)} for n in apples}

# drop
write_state({n: s.clone() for n, s in rest_states.items()})
for n, (obj, _, _) in apples.items():
    state = rest_states[n].clone(); state[2] += 0.05
    obj.write_root_state_to_sim(state.unsqueeze(0)); obj.reset()
rest_z = {n: float(rest_states[n][2]) for n in apples}
drop = {n: {"max_rebound_mm": 0.0, "settle_s": None, "contact": False, "moving": False} for n in apples}
for i in range(int(3.0 / DT)):
    step()
    for n in apples:
        z = float(pos(n)[2]); vz = float(apples[n][0].data.root_lin_vel_w[0, 2])
        if not drop[n]["contact"] and z - rest_z[n] < 0.002:
            drop[n]["contact"] = True
        if drop[n]["contact"]:
            drop[n]["max_rebound_mm"] = max(drop[n]["max_rebound_mm"], (z - rest_z[n]) * 1e3)
            if speed(n) < 1e-3 and spin(n) < 0.05 and drop[n]["settle_s"] is None:
                drop[n]["settle_s"] = round(i * DT, 3)
report["drop_5cm"] = {n: {"max_rebound_mm": round(v["max_rebound_mm"], 2), "settle_s": v["settle_s"],
                          "final_xy_offset_mm": round(float(torch.linalg.norm(pos(n)[:2] - rest_states[n][:2])) * 1e3, 2)}
                      for n, v in drop.items()}

# push
for n, (obj, _, _) in apples.items():
    state = rest_states[n].clone(); state[7] = 0.3; state[10:13] = 0.0
    obj.write_root_state_to_sim(state.unsqueeze(0)); obj.reset()
push = {n: {"stop_s": None, "max_spin_rad_s": 0.0} for n in apples}
for i in range(int(4.0 / DT)):
    step()
    for n in apples:
        push[n]["max_spin_rad_s"] = max(push[n]["max_spin_rad_s"], spin(n))
        if push[n]["stop_s"] is None and i > 2 and speed(n) < 2e-3:
            push[n]["stop_s"] = round(i * DT, 3)
report["push_0.3m_s"] = {n: {"travel_mm": round(float(torch.linalg.norm(pos(n)[:2] - rest_states[n][:2])) * 1e3, 1),
                             "stop_s": push[n]["stop_s"], "max_spin_rad_s": round(push[n]["max_spin_rad_s"], 2),
                             "speed_after_4s_mm_s": round(speed(n) * 1e3, 2)} for n in apples}

# tilt
write_state({n: s.clone() for n, s in rest_states.items()})
for _ in range(int(0.5 / DT)):
    step()
origin = {n: pos(n) for n in apples}
onset = {n: None for n in apples}
masses = {n: float(apples[n][0].root_physx_view.get_masses().sum()) for n in apples}
for i in range(int(50.0 / DT)):
    angle = math.radians(1.0 * i * DT)
    for n, (obj, _, _) in apples.items():
        force = torch.tensor([[[masses[n] * 9.81 * math.tan(angle), 0.0, 0.0]]])
        obj.set_external_force_and_torque(force, torch.zeros_like(force), is_global=True)
    step()
    for n in apples:
        if onset[n] is None and float(torch.linalg.norm(pos(n) - origin[n])) > 0.005:
            onset[n] = round(math.degrees(angle), 2)
    if all(v is not None for v in onset.values()):
        break
report["tilt_onset_deg"] = onset
report["tilt_note"] = "atan(mu_s) for the mesh apple against the desk: %.1f deg" % math.degrees(
    math.atan((mesh_apple.static_friction + 0.5) / 2))

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
import os  # noqa: E402

os._exit(0)  # Kit's close can hang after a headless physics-only run.
