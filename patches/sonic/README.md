# SONIC robot-side idle watchdog

This patch targets the exact `gear_sonic_deploy` source installed on `ssh unitree`
on 2026-09-24 (SONIC commit `a0732b6`). It is **not installed** on the robot.
The original `zmq_manager.hpp` and `zmq_endpoint_interface.hpp` SHA-256 values
were respectively:

- `327723b44dcb3cc38c4ca65f9bcb2aebce06c062224f7d1a28738b8fc15e934b`
- `0c182fa3dfe1310a07e07860b4653f56c24ef725eb4695a4e991f499ff03a39f`

The patch adds a 300 ms watchdog for **accepted Protocol v4 tokens** while
`zmq_manager` is in streamed motion. It returns to planner mode on timeout,
requests the existing safety reset, and prevents the manager from forwarding
stale external tokens or hand targets while planner initialization occurs. It
also rejects non-finite or wrong-width tokens and clears hand/VR flags when a
stream is reset. Packet receipt alone cannot refresh the watchdog.

Apply only to a source copy until a complete sim test and a supervised physical
standing-idle test pass:

```bash
scripts/apply-sonic-idle-watchdog.sh --check /path/to/GR00T-WholeBodyControl
scripts/apply-sonic-idle-watchdog.sh --apply /path/to/source-copy
```

The patch does not start or stop SONIC. It does not replace the robot's physical
emergency stop. It has not yet demonstrated that the robot stands safely after
a timeout; that requires measured robot state and the actual deployed binary.
The workstation must stage a first valid token before switching to streamed
motion, because a delayed first inference otherwise triggers the same timeout.

Validation completed on 2026-09-24: the patch applied to a source copy; the
pure watchdog C++ test passed; the patched main C++ translation unit passed
`g++ -std=c++2a -fsyntax-only` on Unitree's aarch64 computer; and the full
`g1_deploy_onnx_ref` binary linked in a separate temporary source tree at
`/tmp/vla-sonic-idle-buildroot/target/release/g1_deploy_onnx_ref`. The source
tree's DDS `.so` files were Git LFS pointers, so that temporary build used the
real DDS libraries already installed under `/home/unitree/sonic_jp5/vendor`.
That temporary ARM binary's SHA-256 is
`80474149eb3abf71a1713b85a1fb62b5dca26dd87b2840ab15ad78f0e3bed300`.
The `tests/cpp/test_zmq_manager_idle.cpp` hardware-free ARM test exercised the
actual manager on local ZMQ: an explicit planner command returned it to idle;
then malformed pose messages arrived, no valid token was accepted, and the
manager returned from streamed motion to planner mode by timeout.
A running full controller test and measured physical idle behavior remain open.
