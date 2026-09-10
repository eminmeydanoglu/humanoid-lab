import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import cloudwalk_sim_controller as controller


class CloudWalkSimControllerTests(unittest.TestCase):
    def test_status_reports_live_physics_and_actuation_state(self):
        bridge = controller.SimulatorControllerBridge.__new__(controller.SimulatorControllerBridge)
        bridge.status = controller.ControllerStatus()
        bridge.control = Mock()
        bridge.control.recv_json.return_value = {"op": "status"}
        bridge.zmq = Mock(NOBLOCK=1, Again=RuntimeError)
        actuation = Mock()
        actuation.mode.value = "passive"
        robot = Mock()
        robot.data.root_pos_w = MagicMock()
        robot.data.root_lin_vel_w = MagicMock()
        robot.data.root_pos_w.__getitem__.return_value.tolist.return_value = [-0.5, 0.0, 0.75]
        robot.data.root_lin_vel_w.__getitem__.return_value.tolist.return_value = [0.0, 0.0, -1.0]

        response = bridge.poll_control(actuation, robot, Mock(), True)

        self.assertEqual(response["actuation_mode"], "passive")
        self.assertEqual(response["root_pos_w"], [-0.5, 0.0, 0.75])
        self.assertEqual(response["root_lin_vel_w"], [0.0, 0.0, -1.0])
        bridge.control.send_json.assert_called_once_with(response)

    def test_policy_activation_restarts_watchdog_grace_period(self):
        bridge = controller.SimulatorControllerBridge.__new__(controller.SimulatorControllerBridge)
        bridge.status = controller.ControllerStatus(
            active=True,
            state="holding",
            started_monotonic=1.0,
            last_body_monotonic=2.0,
        )
        bridge.control = Mock()
        bridge.control.recv_json.return_value = {"op": "run"}
        bridge.zmq = Mock(NOBLOCK=1, Again=RuntimeError)

        with patch.object(controller.time, "monotonic", return_value=50.0):
            response = bridge.poll_control(Mock(), Mock(), Mock(), False)

        self.assertEqual(response, {"ok": True, "state": "controlled", "timeline_playing": False})
        self.assertEqual(bridge.status.started_monotonic, 50.0)
        self.assertIsNone(bridge.status.last_body_monotonic)
        bridge.control.send_json.assert_called_once_with(response)

    def test_watchdog_never_expires_holding_without_native_body_frames(self):
        bridge = controller.SimulatorControllerBridge.__new__(controller.SimulatorControllerBridge)
        bridge.status = controller.ControllerStatus(
            active=True,
            state="holding",
            started_monotonic=1.0,
            last_body_monotonic=None,
        )

        self.assertFalse(bridge._body_command_stale(1000.0))

    def test_watchdog_allows_first_body_round_trip_after_policy_activation(self):
        bridge = controller.SimulatorControllerBridge.__new__(controller.SimulatorControllerBridge)
        bridge.status = controller.ControllerStatus(
            active=True,
            state="controlled",
            started_monotonic=50.0,
            last_body_monotonic=None,
        )

        self.assertFalse(bridge._body_command_stale(50.0 + controller.STARTUP_GRACE_SECONDS))
        self.assertTrue(bridge._body_command_stale(50.0 + controller.STARTUP_GRACE_SECONDS + 0.001))

    def test_watchdog_rejects_expired_body_commands(self):
        bridge = controller.SimulatorControllerBridge.__new__(controller.SimulatorControllerBridge)
        bridge.status = controller.ControllerStatus(
            active=True,
            state="controlled",
            started_monotonic=10.0,
            last_body_monotonic=20.0,
        )

        self.assertFalse(bridge._body_command_stale(20.0 + controller.BODY_STALE_SECONDS))
        self.assertTrue(bridge._body_command_stale(20.0 + controller.BODY_STALE_SECONDS + 0.001))


if __name__ == "__main__":
    unittest.main()
