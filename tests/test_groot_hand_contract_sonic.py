from __future__ import annotations

import shutil
import subprocess
import textwrap
import unittest


class PinnedSonicHandContractTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("docker"), "docker is required for pinned SONIC integration")
    def test_real_robot_model_assignment_and_extraction(self) -> None:
        script = textwrap.dedent(
            """
            import numpy as np

            from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
            from gear_sonic.utils.inference.vla_utils import prepare_observation_for_eval
            from humanoid_lab.psi0_bridge.groot_hand_contract import left_hand_observation_to_robot_model_actuated

            model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

            def parse(body, left, right):
                q = model.get_configuration_from_actuated_joints(
                    body_actuated_joint_values=body,
                    left_hand_actuated_joint_values=left,
                    right_hand_actuated_joint_values=right,
                )
                return prepare_observation_for_eval(
                    model, {"q": q[None, None], "state": {}}
                )["state"]

            body = np.arange(100, 129, dtype=np.float64)
            left_live = np.arange(200, 207, dtype=np.float64)
            right_live = np.arange(300, 307, dtype=np.float64)
            state = parse(
                body,
                left_hand_observation_to_robot_model_actuated(left_live, "model-independent"),
                right_live,
            )
            expected_body = {
                "left_leg": np.arange(100, 106),
                "right_leg": np.arange(106, 112),
                "waist": np.arange(112, 115),
                "left_arm": np.arange(115, 122),
                "right_arm": np.arange(122, 129),
            }
            for name, expected in expected_body.items():
                np.testing.assert_array_equal(state[name].reshape(-1), expected)
            np.testing.assert_array_equal(
                state["left_hand"].reshape(-1), [205, 206, 203, 204, 200, 201, 202]
            )
            np.testing.assert_array_equal(
                state["right_hand"].reshape(-1), [303, 304, 305, 306, 300, 301, 302]
            )

            raw_row = np.array([
                -0.29605788, 0.77710658, 0.03198371,
                -0.22154425, -1.15354490,
                -0.29018801, -0.74398851,
            ], dtype=np.float64)
            row_state = parse(
                np.zeros(29),
                left_hand_observation_to_robot_model_actuated(raw_row, "model-independent"),
                np.zeros(7),
            )
            np.testing.assert_allclose(
                row_state["left_hand"].reshape(-1),
                [-0.29018801, -0.74398851, -0.22154425, -1.15354490,
                 -0.29605788, 0.77710658, 0.03198371],
                rtol=0,
                atol=1e-7,
            )
            """
        )
        result = subprocess.run(
            [
                "docker", "exec", "humanoid-lab-dev", "sh", "-lc",
                "cd /opt/src/sonic && PYTHONPATH=/workspace/humanoid-lab/src:/opt/src/sonic "
                "/opt/venvs/isaac-sonic/bin/python -",
            ],
            input=script,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
