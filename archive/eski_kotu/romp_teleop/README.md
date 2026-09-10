# romp_teleop — ROMP -> SONIC (Protocol v3, SMPL encoder mode 2) -> G1

Bridge between monocular human pose (ROMP) and the SONIC/GR00T-WholeBodyControl
stack, targeting the learned SMPL encoder (encoder mode 2) and a Unitree G1.

```
video/webcam -> ROMP -> SMPL-24 (thetas) --(Stage 1)--> .npz / ZMQ:5558
                                                |
                                                v
                   SONIC conversion (Stage 2, container, gear_sonic helpers)
                   aa->quat -> ytoz(+90x) -> compute_human_joints
                   -> remove_smpl_base_rot -> quat_inv*joints  (root-local Z-up)
                                                |
                                                v
                       SONIC Protocol v3 'pose'  tcp://*:5556
                                                |
                                                v
                        SONIC deploy (SMPL encoder mode 2) -> G1
```

## Environments (kept separate)
* **Stage 1 (ROMP env, host):** `~/romp-venv` (torch 2.11+cu128), repo `~/ROMP`.
* **Stage 2 (SONIC env, container):** `use-sonic-sim` venv, `gear_sonic` at
  `/opt/src/sonic`. Only needs torch + numpy + pyzmq.

Container workspace is bind-mounted from `~/code/humanoid-lab-sonic-isaac`
(=> `/workspace/humanoid-lab`), so these files must live under that worktree.

## Protocol v3 fields sent to SONIC
`smpl_joints (1,24,3)`, `smpl_pose (1,21,3)=0`, `joint_pos (1,29)` (wrists at
`[23:29]`), `joint_vel (1,29)=0`, `body_quat (1,4)`, `frame_index (1,) i64`.
Header 1280 bytes; topic `pose`. Encoder obs (`sonic_v1_1`): 10-frame step-1
windows buffered by the deploy.

## Usage

### 1. Precompute SMPL from a video/webcam (ROMP env, host)
```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
~/romp-venv/bin/python romp_pose_streamer.py \
    --source 0 --seconds 15 --save-smpl webcam_smpl.npz \
    --debug-coordinates --show
# video: --source /path/clip.mp4
```

### 2. Convert + publish v3 (SONIC env, container)
```bash
docker exec -it humanoid-lab-dev bash -lc \
  'cd /workspace/humanoid-lab/romp_teleop && \
   /opt/venvs/sonic-sim/bin/python romp_to_sonic_bridge.py \
     --sonic-root /opt/src/sonic --play webcam_smpl.npz \
     --fps 50 --loop --smooth 0.7 --root-mode yaw'
```

### 3. Validate the wire format without the deploy
```bash
docker exec -it humanoid-lab-dev bash -lc \
  'cd /workspace/humanoid-lab/romp_teleop && \
   /opt/venvs/sonic-sim/bin/python tools/verify_v3_subscriber.py --port 5556 --count 3'
```

## Coordinate convention (empirically calibrated)
ROMP's `global_orient` carries ~180 deg about X (its camera convention is
Y-down); SONIC's SMPL is Y-up. `flip_romp_global_orient()` applies
`diag(1,-1,-1)` to the root before the SONIC pipeline. `--no-flip-x` disables it.
Root modes: `yaw` (default, gravity-aligned), `full`, `identity`.

Diagnostics: `tools/diag_coordinates.py` prints landmark vectors for
ROMP joints / SONIC identity / SONIC flipped / final root-local.

## Limitations / status
* The C++ `gear_sonic_deploy` is **not built** on this host and its CMake
  requires TensorRT (absent). Until a deploy (or Python decoder loop) consumes
  `tcp://*:5556`, packets are validated with the mock subscriber only.
* Root translation is intentionally dropped (root-local pose + heading only).
