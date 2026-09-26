# flux-inference

The deployment half of FLUX 3 Action: the ROS 2 node that runs on the robot (Foxy) or next to the
simulator, and the GPU-side serving process that answers it. See `../README.md` for the split rule and
re-sync commands.

```
ros2/flux_dex3/                 the robot-side node (dex3_node), 30 Hz chunk scheduler, motor output gate
ros2/flux_dex3_interfaces/      StartTask / GetStatus service types
examples/dex3/g1_inference.py   model-facing wrapper (load once, predict per observation)
examples/dex3/zmq_server.py     GPU-side ZMQ server; imports flux_dex3.protocol, so both ends share the wire format
tests/                          the Dex3 deployment tests (no hardware, no ROS install needed)
docs/                           dex3_inference.md, dex3_ros_bridge.md, dex3_ros_inference_todo.md
```

## Dependencies

- `flux_action` comes from `../flux-training/src` (library half). Its `data.lerobot.dex3_view` module
  defines the camera key the node and the model share.
- `flux_dex3` comes from `ros2/flux_dex3`.
- Tests additionally need `numpy`; `tests/test_dex3_zmq.py` needs `pyzmq`;
  `tests/test_g1_inference.py` needs `torch` and the LeRobot Flux3 runtime.

`pytest.ini` wires the two paths, so from this directory:

```sh
python -m pytest -q                    # all deployment tests
python -m pytest -q tests/test_dex3_node_pause.py
```

Any interpreter with numpy + pyzmq works, for example the source checkout's environment:

```sh
/home/aksoy-lab/code/flux-training/flux-action/.venv/bin/python -m pytest -q
```

## Running the pieces

GPU side (one process per GPU; the server owns the model). The checkpoint must be read-only and not a
symlink, so copy it first:

```sh
cp -r /home/aksoy-lab/code/flux-training/flux-action/outputs/dex3/peft-train/checkpoint-2500 /tmp/dex3-ckpt
chmod -R a-w /tmp/dex3-ckpt
/tmp/lerobot-peft-env/bin/python examples/dex3/zmq_server.py \
  --bind-ip 127.0.0.1 --port 5557 --checkpoint /tmp/dex3-ckpt
```

Robot / sim side (ROS 2). Same source file in both places; only the launch configuration differs:

```sh
# robot: Foxy, unitree_hg messages, CycloneDDS bound to eth0, domain 0
ros2 run flux_dex3 dex3_node --ros-args -p endpoint:=tcp://<gpu-ip>:5557 ...

# sim (Isaac in humanoid-lab-main): ROS 2 on the host, ROS_DOMAIN_ID=42 and
# CYCLONEDDS_URI=containers/cyclonedds-sim.xml, and /arm_sdk remapped onto the
# simulator's whole-body /lowcmd topic
ros2 run flux_dex3 dex3_node --ros-args \
  -r /arm_sdk:=/lowcmd \
  -p endpoint:=tcp://127.0.0.1:5557 -p enable_motor_commands:=false
```

Motor output stays disabled until a verified `motor_output_config` is supplied; with it disabled the
node creates no publishers at all. `docs/dex3_ros_bridge.md` documents the topics, the pairing and
staleness gates, the hold behaviour on a late chunk, and the two human-confirmed flags the hardware
config must carry.
