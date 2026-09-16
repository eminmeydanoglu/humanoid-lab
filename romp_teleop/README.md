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

The existing `humanoid-lab-dev` container mounts the main checkout, not this
`romp_faruk` worktree. To run this worktree without replacing that container:

```bash
cd ~/code/romp_faruk
HUMANOID_SOURCE_ROOT=$PWD COMPOSE_PROJECT_NAME=romp-faruk \
  docker compose --env-file ../humanoid-lab-sonic-isaac/.env up -d
```

This creates `romp-faruk-dev`, with the worktree available at
`/workspace/humanoid-lab`.

## Protocol v3 fields sent to SONIC
`smpl_joints (1,24,3)`, `smpl_pose (1,21,3)=0`, `joint_pos (1,29)` (wrists at
`[23:29]`), `joint_vel (1,29)=0`, `body_quat (1,4)`, `frame_index (1,) i64`.
Header 1280 bytes; topic `pose`. Encoder obs (`sonic_v1_1`): 10-frame step-1
windows buffered by the deploy.

## Usage

### Upper-body standing mode (SONIC teleop encoder mode 1)

Keep SONIC actively balancing the legs in its neutral standing pose while ROMP
drives the left wrist, right wrist, and torso proxy. The whole stack (domain-42
MuJoCo sim + `zmq_manager` deploy + bridge + Stage-1 webcam) is launched from
this worktree with one command:

```bash
ssh msi_tail
cd ~/code/romp_faruk/romp_teleop
./romp_upper.sh start          # stop | status | logs | restart
```

`start` stops any stray keyboard deploy / sim on the shared DDS domain and GPU,
then brings up (in `romp-faruk-dev`): the MuJoCo sim on domain 42 / `lo`, the
SONIC deploy with `--input-type zmq_manager`, the upper-body bridge
(`--control-mode upper-body --subscribe`), and the host Stage-1 ROMP streamer on
`tcp://*:5558`. Logs land in `/tmp/romp_upper_{sim,deploy,bridge,stage1}.log`
(container-side logs are inside `romp-faruk-dev:/tmp`).

This publishes Sonic's IDLE planner command plus `vr_position` and
`vr_orientation`. The planner holds a standing lower-body reference and the VR
fields make the deploy select encoder mode 1 automatically. Sonic remains free
to make the small lower-body corrections required for balance.

Manual equivalent (no sim/stage-1 management): run
`run_deploy_zmq.py --input-type zmq_manager --print` and
`romp_to_sonic_bridge.py --sonic-root /opt/src/sonic --control-mode upper-body
--subscribe --raw-port 5558 --port 5556 --fps 50 --smooth 0.75` in
`romp-faruk-dev`, with `sim_mujoco_domain42.py` already running on domain 42/lo
and the Stage-1 streamer publishing on 5558.

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
docker exec -it romp-faruk-dev bash -lc \
  'cd /workspace/humanoid-lab/romp_teleop && \
   /opt/venvs/sonic-sim/bin/python romp_to_sonic_bridge.py \
     --sonic-root /opt/src/sonic --play webcam_smpl.npz \
     --fps 50 --loop --smooth 0.75'
```

### 3. Validate the wire format without the deploy
```bash
docker exec -it romp-faruk-dev bash -lc \
     'cd /workspace/humanoid-lab/romp_teleop && \
   /opt/venvs/sonic-sim/bin/python tools/verify_v3_subscriber.py --port 5556 --count 3'

### GEM-X live webcam mode

GEM-X is installed in the same worktree at `third_party/GEM-X`, with its own
`.venv` and the official GEM checkpoint/ONNX assets. The worktree wrapper
`gem_to_sonic_live.py` delegates to SONIC's official
`gear_sonic/examples/live_camera_teleop/webcam_stream.py`; it does not replace
the GEM-to-SMPL conversion logic.

The complete MSI demo (MuJoCo + deploy + live GEM webcam) is started with:

```bash
ssh msi_tail
source ~/.zshrc.local
romp-miror gem
```

Stop it with `romp-miror stop`, inspect it with `romp-miror status`, and follow
the GEM stage with `docker logs -f romp-gem-live`. The GEM process uses the
webcam at `/dev/video0`, publishes official SONIC Protocol v3 on port `5556`,
and enables the 77-joint gravity-aligned SOMA path with no extra standing or
waist constraints from the ROMP bridge.
```

## Coordinate convention (empirically calibrated)
ROMP's `global_orient` carries ~180 deg about X (its camera convention is
Y-down); SONIC's SMPL is Y-up. `flip_romp_global_orient()` applies
`diag(1,-1,-1)` to the root before the SONIC pipeline. `--no-flip-x` disables it.
Root mode is `full` by default and matches GEM's gravity-aligned video teleop.
`yaw` and `identity` remain available only for coordinate diagnostics; using
either one removes part of the operator's body orientation and can suppress
bending/crawling motions.

Diagnostics: `tools/diag_coordinates.py` prints landmark vectors for
ROMP joints / SONIC identity / SONIC flipped / final root-local.

## Limitations / status
* The deploy wrapper adds `/data/models/sonic-deploy/lib` to
  `LD_LIBRARY_PATH` automatically when that staged library directory exists.
* Root translation is intentionally dropped (root-local pose + heading only).
