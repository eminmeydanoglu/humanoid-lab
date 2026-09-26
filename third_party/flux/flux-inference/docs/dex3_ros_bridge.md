# G1/Dex3 ZMQ → ROS 2 bridge (command-disabled by default)

The GPU host runs `examples/dex3/zmq_server.py` with Python 3.12 and the LeRobot/PEFT inference environment. The robot runs the Foxy `flux_dex3` package with Python 3.8. Both import the same pure-Python protocol from `ros2/flux_dex3/flux_dex3/protocol.py`. By default the robot node **creates no motor command publishers**; its 30 Hz scheduler exposes the split targets internally as `last_target` for dry-run validation. An explicitly enabled and hardware-configured publisher exists in the source but has not been deployed or tested on a physical robot. A successful `StartTask` does not establish control authority.

## Contract and routing

- Requests: versioned JSON metadata plus raw contiguous RGB8 `(480,640,3)` or `(192,256,3)` and little-endian float32 `(28,)` frames. Responses: JSON metadata plus little-endian float32 `(32,28)` absolute targets. `STATUS` reports loading/ready/error and the SHA-256 digest of the immutable adapter and its saved processor files.
- State: `/lowstate` (`unitree_hg/msg/LowState`) motor 15–28, `/dex3/left/state` and `/dex3/right/state` (`unitree_hg/msg/HandState`) motor 0–6 each. Training manifest names and order are asserted by `tests/test_dex3_robot_bridge.py`. Each request pairs the newest camera frame with, per joint topic, the buffered sample whose arrival is closest to that frame's capture instant; that offset must stay within `pair_tolerance_s` (default 0.1 s) and the frame within `freshness_s` (default 0.25 s), otherwise the task aborts. Joint samples are kept in a ring buffer covering `freshness_s + pair_tolerance_s`.
- Selected command topics: `/arm_sdk` (`unitree_hg/msg/LowCmd`) motor 15–28; `/dex3/left/cmd` and `/dex3/right/cmd` (`unitree_hg/msg/HandCmd`) motor 0–6. Only explicit `enable_motor_commands=true` with a complete, verified `motor_output_config` creates these publishers. The default launch creates none.
- Services: `/flux_dex3/start_task` (`flux_dex3_interfaces/srv/StartTask`, exact training caption), `/flux_dex3/pause_task` and `/flux_dex3/stop_task` (`std_srvs/srv/Trigger`), `/flux_dex3/get_status` (`flux_dex3_interfaces/srv/GetStatus`). A task requires GPU `READY` and fresh RGB and joint states. Pause latches the **last commanded target** and repeats it at 30 Hz while joint feedback remains fresh; the default mode repeats it internally without ROS motor publishers. Resume with `StartTask` and the same prompt after the previous in-flight request settles; it keeps the latched target until a fresh chunk is ready. Stop clears all local targets and ends publication immediately without waiting for a network response. Neither Stop nor Pause sends a damping-mode transition; the physical controller's behavior after publication stops is unverified. A latched target can still cause motion until the robot reaches it. When a chunk runs out before the next prediction lands, the executor repeats its last executed row at 30 Hz and installs the late chunk at its first row from that instant; the hold ends only with the reply, the network timeout, a non-`READY` model, or a rejected chunk.

## GPU-side preparation

Select a checkpoint and make an immutable local copy of **the adapter and all its checkpoint-owned processor files**. Leave the referenced base policy and encoders available in the LeRobot environment. The server rejects writable/symlinked artifacts; it hashes their content before and after warm-up and resets the model before `READY`. Use `--checkpoint` with an absolute path to that copy. For a localhost-only smoke test:

```sh
cd /home/aksoy-lab/code/flux-training/flux-action
PYTHONPATH="$PWD/src:$PWD/ros2/flux_dex3:$PWD" \
  /tmp/lerobot-peft-env/bin/python examples/dex3/zmq_server.py \
  --checkpoint /absolute/path/to/read-only-checkpoint --bind-ip 127.0.0.1 --port 5557
```

For the wired network, bind the server to the GPU host's **selected wired IP**. Provide `--server-secret-key` (the server's `.key_secret` certificate) and `--client-keys-dir` containing only the robot client's public `.key` certificate. Configure the robot endpoint with that IP, its private client certificate path and the server's public CURVE key. Keep the private certificates readable only by the corresponding service account. Wildcard and unauthenticated non-loopback binds are refused.

## Robot-side prerequisites and build

The robot has Foxy, CycloneDDS and `unitree_hg`. On 2026-09-25 the two packages were installed and built in the isolated `/home/unitree/flux_dex3_ws` workspace; `pyzmq==25.1.2` was installed for Python 3.8 in its `python-deps/` directory. The node and generated service types import successfully when that directory is included in `PYTHONPATH`. A brief command-disabled runtime check in a loopback-only ROS domain called `GetStatus` and received the expected `server unavailable` result; no node was left running. To reproduce a build, source `/opt/ros/foxy/setup.bash` and `/home/unitree/g1_ros_env/install/setup.bash`, prepend the workspace's `python-deps/` to `PYTHONPATH`, then run `colcon build --packages-select flux_dex3_interfaces flux_dex3` in the dedicated workspace. Configure the node's `endpoint`, `client_certificate` (private key file path), `server_public_key` (public key text) and `camera_topic` as ROS parameters; `freshness_s` and `pair_tolerance_s` set the observation gates. Each accepted prediction logs the measured pairing offsets, which is the number to tune `pair_tolerance_s` against. For robot-domain observation the existing CycloneDDS configuration binds to `eth0`; leave that internal network configuration intact. Live robot-domain subscriptions have not yet been exercised.

The live `/camera/color/image_raw` stream was previously observed to stall. Consequently the node rejects missing/stale frames and the live `StartTask` gate remains closed until the camera works. Dry-run replay belongs in a separate test environment. The current robot-installed snapshot predates the PauseTask and expanded logging source changes; those edits remain local until a separate safe deployment/verification. A live motor-output task also requires confirmed ownership of `/arm_sdk`, hand hardware revision and limits, suitable mode/timeout/gain settings, and observed stop/hold behavior. The hardware configuration must explicitly record these confirmations plus 28 joint limits, 14 arm and 14 hand gains and a tracking-error bound. These flags document human verification; the software cannot establish ownership or hardware revision itself. The opt-in command builder uses Unitree's documented arm enable field and a LowCmd CRC checked against the vendor Python reference. The robot's installed native CRC library is an invalid Git LFS placeholder, so physical command acceptance has not been checked.

The ROS node logs model loading/READY/error transitions, first sensor observations, accepted/rejected tasks, prediction request/result latency, chunk changes, pause/stop reasons and a 10-second sensor/status heartbeat. Per-frame warnings are rate-limited. Logs contain metadata and status rather than image data, joint arrays or private CURVE keys.

## Verification

```sh
cd /home/aksoy-lab/code/flux-training/flux-action
PYTHONPATH=src:ros2/flux_dex3 .venv/bin/python -m pytest -q \
  tests/test_dex3_zmq.py tests/test_dex3_network.py tests/test_dex3_reconnect.py \
  tests/test_dex3_robot_bridge.py tests/test_dex3_end_to_end.py \
  tests/test_dex3_node_pause.py tests/test_dex3_command_output.py
```

The fake-model end-to-end test exercises `STATUS` → `READY` → `PREDICT` → 32×28 transport → 30 Hz scheduler → named Unitree target splitting without driving hardware. Fake ROS service tests verify pause/repeat, stop, resume gating and log transitions; fake publisher tests verify arm CRC and hand motor fields. The earlier Foxy package build covered the command-disabled installed snapshot. The edited source needs a fresh sandbox build before use. Real checkpoint load, authenticated wired connection, live ROS subscriptions and physical control require their own verification.
