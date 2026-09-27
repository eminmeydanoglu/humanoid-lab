# Flux Dex3 simulation motor config

`flux-dex3-sim-motor-config.json` is the file the `flux_dex3` node loads through
its `motor_output_config` parameter when this repository runs the model against
the Isaac simulator.  The node validates it with its own `load_motor_config()`
before it creates a publisher; this note records where each value comes from and
what the two different limit contracts in play actually mean.

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

## Two limit contracts, deliberately different

| | Range | Meaning |
| --- | --- | --- |
| Node **prediction envelope** | `joint_limits_rad` in this file | What the model's predicted 32x28 chunk may contain before the node rejects it (`ChunkExecutor` gate and the per-publish check). |
| Simulator **applied range** | the pinned physical model (`configs/datasets/sonic/g1_joint_limits.json`), used by `flux_dds.py` | What the simulated joints can actually do.  The adapter clamps every commanded arm and hand position to this range and reports the first clamp as `flux_command_clipped`. |

The envelope is wider than the applied range for the hands on purpose: the
trained corpus commands the real fingers past the nominal model limits, so a
strict config rejects legitimate predictions.  The reversal is never silent —
an out-of-range-but-accepted target is clamped in the adapter before the torque
law, not left to PhysX.

- Arms keep the **strict physical limits** in both contracts (see the evidence
  below: neither the corpus nor the closed-loop run ever exceeded them).
- Hands get the physical range expanded by **0.6 rad** on both ends.
- The envelope is derived from the same pinned numbers the adapter clamps with,
  so the two documents cannot drift apart about what "physical" means.

The envelope is deliberately **not** the physical range: it is an acceptance
window for a stochastic predictor.  The model samples its targets, so the
relevant evidence is its closed-loop output, not only the training data it was
fit to.  If a future run predicts outside this window, the node rejects the
chunk (task aborts, publishing ceases) rather than the window quietly growing;
widening it is a deliberate, evidence-driven edit of this file.

## Values and evidence

| Field | Value | Evidence |
| --- | --- | --- |
| `joint_limits_rad` (arms) | the 14 pinned physical limits, unmodified | `configs/datasets/sonic/g1_joint_limits.json`; neither the corpus (2,587,515 frames) nor the 35.2 s closed-loop run ever exceeded them |
| `joint_limits_rad` (hands) | physical limit +/- 0.6 rad | two independent measurements.  **Corpus:** the largest hand target excursion past a physical limit is 0.3491 rad (20.0 deg on four finger channels of both hands).  **Real closed loop:** `data/outputs/flux-dex3/flux-e2e-final/tracking.parquet` (35.2 s, 372 rows of the successful broad diagnostic run) shows the model commanding up to **0.4410 rad** past a physical limit (`right_hand_index_1`, target minimum -0.4410 against a 0.0 lower limit), with eight hand channels overshooting at all.  0.6 rad is that observed overshoot plus 0.159 rad of headroom.  The first canonical revision used 0.35 rad and its E2E aborted at seq=1 with `prediction rejected: action exceeds joint limit` (`data/outputs/flux-dex3/canonical-review-20260927/ros-launch.log`) |
| `arm_kp` / `arm_kd` | `deploy_gains()[15..28]` (14 values each) | the same gains the pinned SONIC deployment sends for those joints (`src/humanoid_lab/controllers/sonic.py`), 5020 armature for shoulders/elbows/wrist roll, 4010 for wrist pitch/yaw |
| `hand_kp` / `hand_kd` | `1.5` / `0.1` for all 14 hand joints | the pinned deployment's Dex3 hand driver (`dex3_hands.hpp`); Unitree's own examples bracket this (0.5-2.0 / 0.1) |
| `max_tracking_error_rad` | `2.5` | calibrated from the same corpus: per-frame max \|target - measured\| p99.9 = 2.06 rad, largest non-corrupt 2.37 rad; the bound rejects rows that are not robot poses.  The hand envelope's 0.6 rad clamp stays well inside it |
| `hand_timeout_enabled` | `true` | the node writes this as the hand motor watchdog bit; Unitree's Python example sets it and the SONIC hand driver leaves it clear for normal commands.  In simulation the flag is inert (the `flux_dds` bridge does not interpret mode bits); on hardware it is the fail-safe choice |
| `arm_mode_machine` | `5` | Unitree's variant table gives `g1_29dof_with_hand_rev_1_0` -- this profile's asset -- `mode_machine` 5, and the pinned G1 29DoF teleop config declares `MODE_MACHINE: 5` |

### Hand envelope, joint by joint

Physical range from the pinned model, envelope as written into
`joint_limits_rad`:

| Joint | Physical | Envelope |
| --- | --- | --- |
| `left_hand_thumb_0_joint` | [-1.0472, 1.0472] | [-1.6472, 1.6472] |
| `left_hand_thumb_1_joint` | [-0.724312, 1.0472] | [-1.32431, 1.6472] |
| `left_hand_thumb_2_joint` | [0, 1.74533] | [-0.6, 2.34533] |
| `left_hand_middle_0_joint` | [-1.5708, 0] | [-2.1708, 0.6] |
| `left_hand_middle_1_joint` | [-1.74533, 0] | [-2.34533, 0.6] |
| `left_hand_index_0_joint` | [-1.5708, 0] | [-2.1708, 0.6] |
| `left_hand_index_1_joint` | [-1.74533, 0] | [-2.34533, 0.6] |
| `right_hand_thumb_0_joint` | [-1.0472, 1.0472] | [-1.6472, 1.6472] |
| `right_hand_thumb_1_joint` | [-1.0472, 0.724312] | [-1.6472, 1.32431] |
| `right_hand_thumb_2_joint` | [-1.74533, 0] | [-2.34533, 0.6] |
| `right_hand_index_0_joint` | [0, 1.5708] | [-0.6, 2.1708] |
| `right_hand_index_1_joint` | [0, 1.74533] | [-0.6, 2.34533] |
| `right_hand_middle_0_joint` | [0, 1.5708] | [-0.6, 2.1708] |
| `right_hand_middle_1_joint` | [0, 1.74533] | [-0.6, 2.34533] |

The largest permitted overshoot is 0.6 rad; the widest joint envelope is
3.2944 rad (the symmetric thumbs), against the 5.2 rad of the +/-2.6 rad
diagnostic that was used as a stopgap.  Gross outliers (a unit mix-up, a
garbage row) are still rejected by the node: on `right_hand_index_0_joint` and
`right_hand_index_1_joint`, 2.50 rad and 3.20 rad fail the gate,
`left_hand_middle_1_joint` at -2.50 rad fails it, and every arm target outside
its physical limit fails it too.

## Simulator asset check

The shipped `g1_29dof_with_hand_rev_1_0.usd` was read directly with usd-core
(no simulation): all 43 limited revolute joints carry exactly the pinned
model's ranges, arms and hands included.  So the adapter's clamp is the asset's
own range, and "physical" means the same numbers on both sides.

## Not verified here

- Live endpoint matching between the node's ROS topics and the simulator's raw
  DDS topics (`/arm_sdk` <-> `rt/arm_sdk`, `/lowstate` <-> `rt/lowstate`,
  `/dex3/*/cmd|state` <-> `rt/dex3/*/cmd|state`).  The raw names follow the
  `rmw_cyclonedds_cpp` mapping and the Unitree SDK examples; the wire check is
  the integration owner's `scripts/flux-dds-probe.py`.
- End-to-end model runs: this config is intended to replace the +/-2.6 rad
  diagnostic override without any override.  The first canonical revision
  (0.35 rad hands) aborted at seq=1 on a real model chunk; the 0.6 rad envelope
  is sized from that run's own target trace, but the confirming E2E re-run is
  the integration owner's to execute and record.
- Physical command acceptance, CRC handling and hardware stop/hold behaviour.
