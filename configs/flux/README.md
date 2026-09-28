# Flux Dex3 simulation motor config

`flux-dex3-sim-motor-config.json` is the file the `flux_dex3` node loads through
its `motor_output_config` parameter when this repository runs the model against
the Isaac simulator.  The node validates it with its own `load_motor_config()`
before it creates a publisher; this note records where each value comes from and
how the node and simulator enforce the same physical limits.

## Scope: simulation only

The two confirmation flags are **scoped to the simulation**:

- `control_authority_confirmed`: in a run of the `flux_dds` provider the
  simulator is the exclusive owner of `/arm_sdk` and the two hand command
  topics; no SONIC controller process runs and no second command source exists.
  This is a statement about the simulated deployment, not about a physical G1.
- `hand_revision_confirmed`: the hand revision is the pinned asset's
  (`g1_29dof_with_hand_rev_1_0`, Dex3, 7 joints per hand in the training order).
  No physical hand was inspected.

`load_motor_config()` requires both flags to be `true` before the node will
publish, so this file is **not evidence of hardware ownership or hand
revision** and must not be copied to a real robot.  A physical run needs its
own configuration and its own confirmations.

## Physical joint limits

`joint_limits_rad` contains the 28 arm and hand ranges from
`configs/datasets/sonic/g1_joint_limits.json` in model action order.  The node
clips every finite prediction to these ranges when scheduling a chunk, and
`CommandOutput` clips again before publishing.  A limit overshoot therefore
executes at the nearest physical boundary while the task continues.  Invalid
chunks, stale observations, and excessive tracking error still fail their
respective safety checks.  The simulator's `flux_dds.py` adapter also clamps
incoming targets to the pinned physical ranges and reports clipping in its
status.

The model's training data contains hand targets up to 0.3491 rad beyond the
asset's limits.  A 35.2 s closed-loop diagnostic run measured up to 0.4410 rad
of overshoot (`right_hand_index_1_joint`).  These targets now execute at the
physical limit without requiring a larger acceptance envelope.  Previous
configs expanded hand limits by 0.35 or 0.6 rad; the 0.35 rad configuration
aborted a task at seq=1.

The other motor settings retain their independent contracts: arm gains come
from `deploy_gains()[15..28]`, hand gains are 1.5/0.1, and
`max_tracking_error_rad` is 2.5.  The hand watchdog remains enabled and
`arm_mode_machine` is 5 for the pinned G1 asset.

## Simulator asset check

The shipped `g1_29dof_with_hand_rev_1_0.usd` was read directly with usd-core
(no simulation): all 43 limited revolute joints carry exactly the pinned
model's ranges, arms and hands included.  So the adapter's clamp is the asset's
own range, and "physical" means the same numbers on both sides.

## Not verified here

- Live endpoint matching between the node's ROS topics and the simulator's raw
  DDS topics (`/arm_sdk` <-> `rt/arm_sdk`, `/lowstate` <-> `rt/lowstate`,
  `/dex3/*/cmd|state` <-> `rt/dex3/*/cmd|state`). The raw names follow the
  `rmw_cyclonedds_cpp` mapping and the Unitree SDK examples; check live topic
  traffic with ROS 2 while the simulator and launch are running.
- End-to-end model runs with the restarted node and simulator using this
  configuration; the offline tests cover clipping and task continuity.
- Physical command acceptance, CRC handling and hardware stop/hold behaviour.
