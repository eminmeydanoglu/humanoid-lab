# Flux Dex3 motor configs

`flux-dex3-robot-motor-config.json` uses the pinned SONIC joint limits and arm
`deploy_gains()[15:29]`; its `arm_mode_machine=5` matches the robot's LowState.
Hand gains 1.5/0.1 and the motion timeout bit 0 match the robot's
`gear_sonic_deploy/.../include/dex3_hands.hpp::sizeCommand`; its `hold()` keeps
those gains while its relax command sets timeout bit 1. The 4.0 rad
tracking-error limit permits large differences between measured and target positions. Set
`enable_motor_commands=true` with this config to create the command publishers.
`pause_task` repeats the last target at 30 Hz; it requires an executed target.
`stop_task` clears targets and ceases publishing. Confirm control ownership and
physical hand revision before commanding the robot.

## Simulation config

`flux-dex3-sim-motor-config.json` is the file the `flux_dex3` node loads through
its `motor_output_config` parameter when this repository runs the model against
the Isaac simulator.  The node validates it with its own `load_motor_config()`
before it creates a publisher; this note records where each value comes from and
how the node and simulator enforce the same physical limits.

## Simulation scope

The simulation config describes the pinned Dex3 asset. Use the robot-specific
config for physical deployment; simulation settings do not establish hardware
control ownership or hand revision.

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
`max_tracking_error_rad` is 2.5 in the simulation config.  Its hand timeout bit is enabled and
`arm_mode_machine` is 5 for the pinned G1 asset.

## Simulator asset check

The shipped `g1_29dof_with_hand_rev_1_0.usd` was read directly with usd-core
(no simulation): all 43 limited revolute joints carry exactly the pinned
model's ranges, arms and hands included.  So the adapter's clamp is the asset's
own range, and "physical" means the same numbers on both sides.

## Live simulation verification (2026-10-01)

The realistic apple profile on `aksoyy` was tested with V2 raw checkpoint 6000,
3-second model-network and returned-observation age budgets, and motor output
enabled. The simulation launch remaps `/lowcmd` to `/arm_sdk`, matching the
controller's `rt/arm_sdk` reader. Body and both hand states arrived; a late ROS
reader discovered them using the default configuration. Model predictions
produced arm and hand motion. Start, pause, resume, and stop passed; stop ceased
command publication. Camera, body/hand feedback, and all three command topics
were received through the Tailscale Foxglove WebSocket in a real browser.

Run evidence: `/home/aksoyy/code/data/outputs/flux-e2e-aksoyy/control-3s-e2e.json`.

## Physical deployment verification

Physical command acceptance, CRC handling, and hardware stop/hold behaviour
require separate verification on the physical robot.

## Low-latency Foxglove preview

Use `foxglove-live.yaml` for the Tailscale bridge and choose
`/camera/color/image_raw/compressed` in the Image panel. The JPEG preview keeps
one latest frame, uses depth-1 best-effort QoS, drops frames older than 0.5 s,
and publishes at most 15 FPS with quality 75. Capture headers remain unchanged.
The raw image remains available to local inference and is excluded only from
this remote bridge. The shared WebSocket backlog is 32 messages; overflow drops
oldest data. TCP and viewer buffering remain outside that queue limit.
