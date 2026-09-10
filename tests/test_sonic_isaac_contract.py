#!/usr/bin/env python3
"""Contract tests: name-based mapping, limits, DDS isolation and launch gates.

These load the shipped module from ``tools/`` exactly the way the runner does,
so a regression in the real mapping/safety code fails this suite.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sonic_isaac_contract import (  # noqa: E402
    BODY_JOINT_COUNT,
    SIM_DDS_DOMAIN_ID,
    SONIC_BODY_JOINT_NAMES,
    ContractError,
    DdsProfile,
    JointLimits,
    SessionClaim,
    build_joint_mapping,
    check_launch_safety,
    check_sim_dds_isolation,
    collect_refusals,
    load_dds_profile,
    mapping_evidence_summary,
    resolve_cyclonedds_uri,
    validate_limits,
)

# The two Inspire hands add 24 DOFs; they sit outside the SONIC body contract.
INSPIRE_HAND_JOINTS = (
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

LOOPBACK_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="lo" multicast="false" />
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
      <DontRoute>true</DontRoute>
    </General>
  </Domain>
</CycloneDDS>
"""


def limits_for(names=SONIC_BODY_JOINT_NAMES) -> list[JointLimits]:
    return [
        JointLimits(
            name=name,
            position_min=-2.5,
            position_max=2.5,
            effort_max=88.0,
            velocity_max=32.0,
        )
        for name in names
    ]


def safe_kwargs(**overrides):
    """A fully safe launcher configuration; override one field per refusal test."""
    base = dict(
        cyclonedds_uri="file:///opt/humanoid-lab/cyclonedds-sim.xml",
        profile=load_dds_profile(LOOPBACK_XML),
        domain_id=SIM_DDS_DOMAIN_ID,
        sonic_mode="sim",
        requested_interface="lo",
        argv=["g1_deploy_onnx_ref", "lo", "--input-type", "keyboard", "sim"],
        physical_interfaces=["lo", "enp129s0", "wlp130s0"],
        existing_session=None,
        own_pid=4242,
    )
    base.update(overrides)
    return base


class JointMappingTest(unittest.TestCase):
    def test_identity_order_round_trips(self) -> None:
        mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, SONIC_BODY_JOINT_NAMES)
        self.assertEqual(mapping.sonic_to_isaac, tuple(range(BODY_JOINT_COUNT)))
        self.assertEqual(mapping.isaac_to_sonic, tuple(range(BODY_JOINT_COUNT)))

    def test_maps_by_name_when_asset_order_differs(self) -> None:
        # Deliberately reordered asset: right leg before left leg, waist last.
        reordered = (
            tuple(name for name in SONIC_BODY_JOINT_NAMES if name.startswith("right_hip")
                  or name.startswith("right_knee") or name.startswith("right_ankle"))
            + tuple(name for name in SONIC_BODY_JOINT_NAMES if name.startswith("left_hip")
                    or name.startswith("left_knee") or name.startswith("left_ankle"))
            + tuple(name for name in SONIC_BODY_JOINT_NAMES if name.startswith("left_shoulder")
                    or name.startswith("left_elbow") or name.startswith("left_wrist"))
            + tuple(name for name in SONIC_BODY_JOINT_NAMES if name.startswith("right_shoulder")
                    or name.startswith("right_elbow") or name.startswith("right_wrist"))
            + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        )
        self.assertEqual(len(reordered), BODY_JOINT_COUNT)
        mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, reordered)

        # The index maps must follow the asset, not the DDS order.
        self.assertEqual(mapping.sonic_to_isaac[0], reordered.index("left_hip_pitch_joint"))
        self.assertEqual(mapping.sonic_to_isaac[12], reordered.index("waist_yaw_joint"))
        # And they must be mutual inverses.
        for sonic_index, isaac_index in enumerate(mapping.sonic_to_isaac):
            self.assertEqual(mapping.isaac_to_sonic[isaac_index], sonic_index)

    def test_extra_hand_dofs_keep_inverse_alignment(self) -> None:
        # Hands interleaved into the middle of the asset order, as a hands-bearing
        # asset would expose them.
        isaac = SONIC_BODY_JOINT_NAMES[:14] + INSPIRE_HAND_JOINTS + SONIC_BODY_JOINT_NAMES[14:]
        mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, isaac)
        self.assertEqual(len(mapping.isaac_names), BODY_JOINT_COUNT + len(INSPIRE_HAND_JOINTS))
        self.assertEqual(len(mapping.isaac_to_sonic), len(isaac))
        for sonic_index, isaac_index in enumerate(mapping.sonic_to_isaac):
            self.assertEqual(mapping.isaac_to_sonic[isaac_index], sonic_index)
        # Left leg maps into the first block untouched by the hands.
        self.assertEqual(mapping.sonic_to_isaac[0], 0)
        # Joints that follow the hands in the asset order are shifted by exactly
        # the number of hand DOFs that were inserted before them.
        self.assertEqual(
            mapping.sonic_to_isaac[14],
            isaac.index("waist_pitch_joint"),
        )
        self.assertEqual(mapping.sonic_to_isaac[14], 14 + len(INSPIRE_HAND_JOINTS))
        self.assertEqual(
            mapping.sonic_to_isaac[-1], isaac.index("right_wrist_yaw_joint")
        )

    def test_hand_dofs_are_not_sonic_controlled(self) -> None:
        isaac = SONIC_BODY_JOINT_NAMES + INSPIRE_HAND_JOINTS
        mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, isaac)
        hand_indices = set(range(BODY_JOINT_COUNT, len(isaac)))
        self.assertFalse(hand_indices & set(mapping.sonic_to_isaac))

    def test_missing_name_is_rejected(self) -> None:
        # A substituted name that is not part of the 29-joint contract leaves
        # right_wrist_yaw_joint unaccounted for.
        broken = SONIC_BODY_JOINT_NAMES[:-1] + ("left_wrist_middle_joint",)
        with self.assertRaises(ContractError) as ctx:
            build_joint_mapping(broken, SONIC_BODY_JOINT_NAMES)
        message = str(ctx.exception)
        self.assertIn("missing=['right_wrist_yaw_joint']", message)
        self.assertIn("unknown=['left_wrist_middle_joint']", message)

    def test_duplicate_name_is_rejected(self) -> None:
        broken = SONIC_BODY_JOINT_NAMES[:-1] + (SONIC_BODY_JOINT_NAMES[0],)
        with self.assertRaises(ContractError) as ctx:
            build_joint_mapping(broken, SONIC_BODY_JOINT_NAMES)
        self.assertIn("duplicate SONIC joint names", str(ctx.exception))

    def test_asset_missing_a_body_joint_is_rejected(self) -> None:
        isaac = tuple(name for name in SONIC_BODY_JOINT_NAMES if name != "waist_roll_joint")
        with self.assertRaises(ContractError) as ctx:
            build_joint_mapping(SONIC_BODY_JOINT_NAMES, isaac)
        self.assertIn("Isaac asset is missing body joints", str(ctx.exception))

    def test_wrong_joint_count_is_rejected(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            build_joint_mapping(SONIC_BODY_JOINT_NAMES[:28], SONIC_BODY_JOINT_NAMES)
        self.assertIn("expected 29 SONIC body joints", str(ctx.exception))


class LimitsTest(unittest.TestCase):
    def test_accepts_finite_positive_limits(self) -> None:
        validate_limits(limits_for())

    def test_rejects_non_finite_effort(self) -> None:
        limits = limits_for()
        limits[3] = JointLimits("left_knee_joint", -2.5, 2.5, float("inf"), 32.0)
        with self.assertRaises(ContractError) as ctx:
            validate_limits(limits)
        self.assertIn("not finite", str(ctx.exception))

    def test_rejects_non_positive_velocity(self) -> None:
        limits = limits_for()
        limits[0] = JointLimits("left_hip_pitch_joint", -2.5, 2.5, 88.0, 0.0)
        with self.assertRaises(ContractError) as ctx:
            validate_limits(limits)
        self.assertIn("velocity_max must be positive", str(ctx.exception))

    def test_rejects_non_positive_effort(self) -> None:
        limits = limits_for()
        limits[0] = JointLimits("left_hip_pitch_joint", -2.5, 2.5, -1.0, 32.0)
        with self.assertRaises(ContractError) as ctx:
            validate_limits(limits)
        self.assertIn("effort_max must be positive", str(ctx.exception))

    def test_rejects_unordered_position_bounds(self) -> None:
        limits = limits_for()
        limits[7] = JointLimits("right_hip_roll_joint", 1.0, -1.0, 88.0, 32.0)
        with self.assertRaises(ContractError) as ctx:
            validate_limits(limits)
        self.assertIn("position bounds are not ordered", str(ctx.exception))

    def test_evidence_summary_covers_every_body_joint(self) -> None:
        mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, SONIC_BODY_JOINT_NAMES)
        summary = mapping_evidence_summary(mapping, limits_for())
        self.assertEqual(summary["body_joint_count"], BODY_JOINT_COUNT)
        self.assertEqual(summary["matcher"], "name")
        self.assertFalse(summary["order_assumed"])
        self.assertEqual(set(summary["limits"]), set(SONIC_BODY_JOINT_NAMES))


class DdsIsolationTest(unittest.TestCase):
    def test_parses_the_shipped_loopback_profile(self) -> None:
        shipped = Path(__file__).resolve().parents[1] / "containers" / "cyclonedds-sim.xml"
        profile = load_dds_profile(shipped.read_text(), source=str(shipped))
        self.assertEqual(profile.interface_addresses, ("lo",))
        self.assertFalse(profile.allow_multicast)
        self.assertTrue(profile.dont_route)
        self.assertTrue(profile.is_loopback_only())

    def test_accepts_domain_42_on_loopback(self) -> None:
        check_sim_dds_isolation(load_dds_profile(LOOPBACK_XML), domain_id=SIM_DDS_DOMAIN_ID)

    def test_rejects_wrong_domain(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            check_sim_dds_isolation(load_dds_profile(LOOPBACK_XML), domain_id=0)
        self.assertIn("not the simulation domain", str(ctx.exception))

    def test_rejects_multicast_enabled(self) -> None:
        xml = LOOPBACK_XML.replace("<AllowMulticast>false</AllowMulticast>",
                                   "<AllowMulticast>true</AllowMulticast>")
        with self.assertRaises(ContractError) as ctx:
            check_sim_dds_isolation(load_dds_profile(xml), domain_id=SIM_DDS_DOMAIN_ID)
        self.assertIn("multicast must be disabled", str(ctx.exception))

    def test_rejects_physical_interface_binding(self) -> None:
        xml = LOOPBACK_XML.replace('address="lo"', 'address="enp129s0"')
        with self.assertRaises(ContractError) as ctx:
            check_sim_dds_isolation(load_dds_profile(xml), domain_id=SIM_DDS_DOMAIN_ID)
        self.assertIn("only 'lo' is allowed", str(ctx.exception))

    def test_rejects_config_without_interface(self) -> None:
        xml = "<CycloneDDS><Domain><General></General></Domain></CycloneDDS>"
        with self.assertRaises(ContractError) as ctx:
            check_sim_dds_isolation(load_dds_profile(xml), domain_id=SIM_DDS_DOMAIN_ID)
        self.assertIn("declares no NetworkInterface", str(ctx.exception))

    def test_uri_must_be_file_scheme(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            resolve_cyclonedds_uri("<CycloneDDS/>")
        self.assertIn("must use file://", str(ctx.exception))

    def test_uri_must_exist(self) -> None:
        with self.assertRaises(ContractError):
            resolve_cyclonedds_uri("file:///nonexistent/cyclonedds.xml")

    def test_uri_resolves_against_root(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "opt" / "humanoid-lab"
            target.mkdir(parents=True)
            config = target / "cyclonedds-sim.xml"
            config.write_text(LOOPBACK_XML)
            resolved = resolve_cyclonedds_uri(
                "file:///opt/humanoid-lab/cyclonedds-sim.xml", root=tmp
            )
            self.assertEqual(resolved, config)


class LauncherGateTest(unittest.TestCase):
    def test_safe_configuration_is_allowed(self) -> None:
        check_launch_safety(**safe_kwargs())
        self.assertEqual(collect_refusals(**safe_kwargs()), ())

    def test_refuses_inline_uri(self) -> None:
        reasons = collect_refusals(**safe_kwargs(cyclonedds_uri="<CycloneDDS/>"))
        self.assertIn("unsafe_cyclonedds_uri", reasons)

    def test_refuses_physical_interface_uri(self) -> None:
        xml = LOOPBACK_XML.replace('address="lo"', 'address="192.168.1.5"')
        reasons = collect_refusals(**safe_kwargs(profile=load_dds_profile(xml)))
        self.assertIn("unsafe_cyclonedds_uri", reasons)

    def test_refuses_wrong_domain(self) -> None:
        reasons = collect_refusals(**safe_kwargs(domain_id=0))
        self.assertIn("unsafe_dds_domain", reasons)

    def test_refuses_non_sim_sonic_mode(self) -> None:
        reasons = collect_refusals(**safe_kwargs(sonic_mode="real"))
        self.assertIn("unsafe_sonic_mode", reasons)

    def test_refuses_real_interface_argument(self) -> None:
        reasons = collect_refusals(**safe_kwargs(requested_interface="enp129s0"))
        self.assertIn("unsafe_physical_interface", reasons)

    def test_refuses_physical_interface_in_argv(self) -> None:
        reasons = collect_refusals(**safe_kwargs(argv=["g1_deploy_onnx_ref", "enp129s0"]))
        self.assertIn("unsafe_physical_interface", reasons)

    def test_refuses_robot_ip_in_argv(self) -> None:
        reasons = collect_refusals(**safe_kwargs(argv=["--zmq-host", "192.168.123.161"]))
        self.assertIn("unsafe_physical_interface", reasons)

    def test_allows_loopback_ip_in_argv(self) -> None:
        self.assertEqual(
            collect_refusals(**safe_kwargs(argv=["--zmq-host", "127.0.0.1"])), ()
        )

    def test_does_not_mistake_paths_for_addresses(self) -> None:
        self.assertEqual(
            collect_refusals(**safe_kwargs(argv=["file:///opt/humanoid-lab/x.xml"])), ()
        )
        self.assertEqual(
            collect_refusals(**safe_kwargs(argv=["--obs-config", "policy/release/o.yaml"])), ()
        )

    def test_refuses_concurrent_session(self) -> None:
        other = SessionClaim(pid=999, domain_id=SIM_DDS_DOMAIN_ID, interface="lo", owner="sonic-isaac")
        reasons = collect_refusals(**safe_kwargs(existing_session=other))
        self.assertIn("session_conflict", reasons)

    def test_reuses_own_recorded_session(self) -> None:
        own = SessionClaim(pid=4242, domain_id=SIM_DDS_DOMAIN_ID, interface="lo", owner="sonic-isaac")
        self.assertEqual(collect_refusals(**safe_kwargs(existing_session=own)), ())

    def test_unrelated_session_on_other_domain_is_ignored(self) -> None:
        other = SessionClaim(pid=999, domain_id=7, interface="lo", owner="other")
        self.assertEqual(collect_refusals(**safe_kwargs(existing_session=other)), ())

    def test_reports_every_reason_at_once(self) -> None:
        reasons = collect_refusals(
            **safe_kwargs(
                cyclonedds_uri=None,
                profile=None,
                domain_id=1,
                sonic_mode="real",
                requested_interface="enp129s0",
                existing_session=SessionClaim(pid=1, domain_id=1, interface="enp129s0", owner="x"),
            )
        )
        for expected in (
            "unsafe_cyclonedds_uri",
            "unsafe_dds_domain",
            "unsafe_sonic_mode",
            "unsafe_physical_interface",
            "session_conflict",
        ):
            self.assertIn(expected, reasons)


if __name__ == "__main__":
    unittest.main()
