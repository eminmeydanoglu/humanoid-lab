# ROS 2 / DDS

A ROS 2 Foxy side container is NOT added by default: the main container talks
DDS directly over `network_mode: host`. The real risk is custom message IDL
hashes, QoS, RMW vendor differences and Foxy ABI requirements — not the wire
protocol. Files here never write to the robot network.

## Stage A — direct DDS from the main container (robot safe, no motor commands)

1. Record ROS_DOMAIN_ID, RMW vendor, NIC/subnet, needed topics/services.
2. Join the robot NIC with the main container (host network).
3. Verify multicast/UDP + firewall; try discovery (`ros2 node/topic list`).
4. Echo a standard message read-only; build custom interfaces from the exact
   same source and decode read-only; match QoS.
5. Only then a safe test publisher under lab conditions.

If Stage A passes, no second container is added.

## Stage B — minimal Foxy service (only on real need)

Enabled only if: closed/binary packages compiled with Foxy, custom interfaces
that cannot be built on the main runtime, or reproducible direct-DDS failure.

Rules: digest-pinned Foxy image, no CUDA/Isaac/GR00T in it, host network,
bind-mounted workspace, started only via `./dev.sh foxy` (`profiles: [foxy]`).

## direct-dds-test.sh

Template that will run the Stage A steps once robot access parameters
(IP/NIC, ROS_DOMAIN_ID, RMW, custom interface repo) are known.