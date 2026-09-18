"""Wire + contract boundaries: request encoding, response decoding, /info, 43D state."""

from __future__ import annotations

import base64
import json
import unittest

import numpy as np

from humanoid_lab.psi0_bridge.contracts import (
    ContractError,
    IMAGE_KEY,
    RAW_STATE_DIM,
    validate_info,
)
from humanoid_lab.psi0_bridge.psi0_client import (
    Psi0ClientError,
    decode_response,
    http_base_from_ws,
    numpy_serialize,
    serialize_request,
)
from humanoid_lab.psi0_bridge.prompt import CANONICAL_PROMPT
from humanoid_lab.psi0_bridge.state import StateContractError, build_raw_state


def reference_decode(value: dict) -> np.ndarray:
    """Independent decoder for the ``psi.deploy.helpers`` numpy envelope."""
    buffer = np.frombuffer(base64.b64decode(value["__numpy__"]), dtype=np.dtype(value["dtype"]))
    return buffer.reshape(tuple(value["shape"]))


def valid_info() -> dict:
    return {
        "policy": "psi0",
        "run_dir": "/outputs/psi0-unitree-dex3-sonic-v1/finetune/run.2609171455",
        "ckpt_step": 40000,
        "dataset_name": "psi0-unitree-dex3-sonic-v1",
        "transforms": [
            {"name": "resize", "size": [240, 320]},
            {"name": "center_crop", "size": [240, 320]},
        ],
        "expected_keys": {
            "image": {IMAGE_KEY: "HxWx3 uint8 image array"},
            "state": {"states": "1x45 unnormalized state vector"},
        },
        "observation": {"state_dim": 45, "normalize_state": True},
        "action": {"action_dim": 80, "action_chunk_size": 30, "action_exec_horizon": 30},
        "rtc_enabled": False,
    }


class RequestEncodingTest(unittest.TestCase):
    def test_request_matches_the_served_wire_format(self) -> None:
        frame = (np.arange(4 * 6 * 3) % 256).astype(np.uint8).reshape(4, 6, 3)
        state = np.arange(RAW_STATE_DIM, dtype=np.float32)
        payload = json.loads(serialize_request(
            image={IMAGE_KEY: frame},
            state=state,
            instruction=CANONICAL_PROMPT,
            dataset_name="psi0-unitree-dex3-sonic-v1",
            timestamp="123.0",
        ))

        self.assertEqual(
            sorted(payload),
            ["condition", "dataset_name", "gt_action", "history", "image", "instruction", "state", "timestamp"],
        )
        self.assertEqual(payload["history"], {})
        self.assertEqual(payload["condition"], {})
        self.assertEqual(payload["gt_action"], [])
        self.assertEqual(payload["instruction"], CANONICAL_PROMPT)
        np.testing.assert_array_equal(reference_decode(payload["image"][IMAGE_KEY]), frame)
        decoded_state = reference_decode(payload["state"]["states"])
        np.testing.assert_array_equal(decoded_state, state)
        self.assertEqual(decoded_state.dtype, np.float32)

    def test_batched_state_is_flattened(self) -> None:
        payload = json.loads(serialize_request(
            image={IMAGE_KEY: np.zeros((2, 3, 3), np.uint8)},
            state=np.arange(RAW_STATE_DIM, dtype=np.float32).reshape(1, -1),
            instruction=CANONICAL_PROMPT,
            dataset_name="d",
        ))
        self.assertEqual(reference_decode(payload["state"]["states"]).shape, (RAW_STATE_DIM,))

    def test_http_base_is_derived_from_the_ws_url(self) -> None:
        self.assertEqual(http_base_from_ws("ws://localhost:8014/ws"), "http://localhost:8014")
        self.assertEqual(http_base_from_ws("wss://host:9000/ws"), "https://host:9000")


class ResponseDecodingTest(unittest.TestCase):
    def test_response_reads_version_and_action(self) -> None:
        action = np.arange(80, dtype=np.float32).reshape(1, 80)
        message = json.dumps({
            "action": numpy_serialize(action),
            "err": 0.0,
            "traj_image": numpy_serialize(np.zeros((1, 1, 3), np.uint8)),
            "version": 7,
        })
        reply = decode_response(message)
        self.assertEqual(reply.version, 7)
        self.assertEqual(reply.action.shape, (80,))
        np.testing.assert_array_equal(reply.action, np.arange(80, dtype=np.float32))

    def test_response_rejects_a_missing_action_or_bad_json(self) -> None:
        with self.assertRaises(Psi0ClientError):
            decode_response(json.dumps({"err": 0.0, "version": 1}))
        with self.assertRaises(Psi0ClientError):
            decode_response("not json")


class InfoContractTest(unittest.TestCase):
    def test_valid_info_is_accepted(self) -> None:
        info = validate_info(valid_info())
        self.assertEqual(info.action_dim, 80)
        self.assertEqual(info.action_chunk_size, 30)
        self.assertEqual(info.action_exec_horizon, 30)
        self.assertEqual(info.state_dim, 45)
        self.assertEqual(info.online_state_dim, RAW_STATE_DIM)  # the bridge still sends 43 raw
        self.assertEqual(info.resize_size, (240, 320))
        self.assertEqual(info.image_key, IMAGE_KEY)

    def test_43d_server_state_is_accepted(self) -> None:
        raw = valid_info()
        raw["expected_keys"]["state"]["states"] = "1x43 unnormalized state vector"
        raw["observation"]["state_dim"] = 43
        self.assertEqual(validate_info(raw).state_dim, 43)

    def test_contract_violations_are_rejected(self) -> None:
        mutations = {
            "image_key": lambda raw: raw["expected_keys"]["image"].pop(IMAGE_KEY),
            "history": lambda raw: raw["expected_keys"]["state"].update({"states": "2x45 unnormalized state vector"}),
            "state_dim": lambda raw: raw["observation"].update({"state_dim": 46}),
            "normalize": lambda raw: raw["observation"].update({"normalize_state": False}),
            "action_dim": lambda raw: raw["action"].update({"action_dim": 79}),
            "chunk": lambda raw: raw["action"].update({"action_exec_horizon": 31}),
            "resize": lambda raw: raw["transforms"][0].update({"size": [128, 128]}),
            "crop_missing": lambda raw: raw["transforms"].pop(1),
            "run_dir_missing": lambda raw: raw.pop("run_dir"),
            "run_dir_empty": lambda raw: raw.update({"run_dir": "  "}),
            "run_dir_not_string": lambda raw: raw.update({"run_dir": 42}),
            "dataset_name_missing": lambda raw: raw.pop("dataset_name"),
            "dataset_name_empty": lambda raw: raw.update({"dataset_name": ""}),
            "extra_image_key": lambda raw: raw["expected_keys"]["image"].update(
                {"observation.images.wrist_left": "HxWx3 uint8 image array"}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                raw = valid_info()
                mutate(raw)
                with self.assertRaises(ContractError):
                    validate_info(raw)

    def test_extra_required_image_key_is_rejected_with_the_offending_key(self) -> None:
        raw = valid_info()
        raw["expected_keys"]["image"]["observation.images.wrist_left"] = "HxWx3 uint8 image array"
        with self.assertRaises(ContractError) as ctx:
            validate_info(raw)
        self.assertIn("observation.images.wrist_left", str(ctx.exception))

    def test_rtc_init_prev_mode_is_rejected_early(self) -> None:
        # PSI_RTC_INIT_PREV=1 makes the server expect state.init_prev_action from
        # the client; this bridge never produces it, so the run must not start.
        raw = valid_info()
        raw["rtc"] = {"mode": "test_time", "init_prev_enabled": True}
        with self.assertRaises(ContractError) as ctx:
            validate_info(raw)
        self.assertIn("init_prev_action", str(ctx.exception))

        raw["rtc"]["init_prev_enabled"] = False
        self.assertTrue(validate_info(raw).rtc_enabled is False)

        raw.pop("rtc")  # older servers may omit the block; that is not an error
        validate_info(raw)


class RawStateTest(unittest.TestCase):
    def test_state_is_body_then_left_then_right(self) -> None:
        payload = {
            "body_q": np.arange(29, dtype=np.float32),
            "left_hand_q": np.arange(7, dtype=np.float32) + 100,
            "right_hand_q": np.arange(7, dtype=np.float32) + 200,
        }
        state = build_raw_state(payload)
        self.assertEqual(state.shape, (RAW_STATE_DIM,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_array_equal(state[:29], np.arange(29, dtype=np.float32))
        np.testing.assert_array_equal(state[29:36], np.arange(7, dtype=np.float32) + 100)
        np.testing.assert_array_equal(state[36:], np.arange(7, dtype=np.float32) + 200)

    def test_plain_lists_are_accepted(self) -> None:
        state = build_raw_state({
            "body_q": list(range(29)),
            "left_hand_q": list(range(7)),
            "right_hand_q": list(range(7)),
        })
        self.assertEqual(state.shape, (RAW_STATE_DIM,))

    def test_incomplete_payloads_raise(self) -> None:
        good = {"body_q": np.zeros(29), "left_hand_q": np.zeros(7), "right_hand_q": np.zeros(7)}
        for name, payload in {
            "not_a_mapping": [],
            "missing_hand": {k: v for k, v in good.items() if k != "left_hand_q"},
            "short_body": {**good, "body_q": np.zeros(28)},
        }.items():
            with self.subTest(case=name):
                with self.assertRaises(StateContractError):
                    build_raw_state(payload)
