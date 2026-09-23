"""The applied-reset handshake: queue ack vs. the simulator's real boundary.

``reset_queued`` only says the simulation loop accepted the work; the loop runs
the reset at its next iteration.  These cases pin the difference: the legacy
``request_reset`` keeps its queue-acknowledged behaviour, and
``request_reset_applied`` returns only once the loop's ``episode_id`` has
advanced, fails closed when it never does, and refuses a malformed status reply
instead of reading it as "no reset yet".

The transport is faked at the single ``_request`` seam, so the decision rule is
exercised without ZMQ.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from humanoid_lab.psi0_bridge import model_controller
from humanoid_lab.psi0_bridge.reset_client import (
    RESET_REQUEST,
    STATUS_REQUEST,
    IsaacResetClient,
    ResetError,
    parse_episode_id,
)


def status_reply(episode_id: int) -> bytes:
    return json.dumps({"episode_id": episode_id, "physics_tick": 10, "timeline": "playing"}).encode()


class FakeEndpointResetClient(IsaacResetClient):
    """An endpoint that queues the reset and applies it on a later status poll.

    ``polls_before_apply`` is how many status reads the loop needs before it
    consumes the queued flag; the episode id is bumped exactly once, when it
    does, and the pre-reset read that ``request_reset_applied`` makes before
    sending the request must never count as an application.
    """

    def __init__(self, *, start_episode: int = 3, polls_before_apply: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.episode_id = int(start_episode)
        self.polls_before_apply = int(polls_before_apply)
        self.queued = False
        self.status_reads = 0
        self.requests: list[bytes] = []
        self.status_replies: list[bytes] | None = None

    def _request(self, request, expected, refused, what, *, timeout_ms=None) -> bytes:
        self.requests.append(request)
        if request == RESET_REQUEST:
            self.queued = True
            return b"reset_queued"
        if request == STATUS_REQUEST:
            if self.status_replies is not None:
                return self.status_replies.pop(0)
            self.status_reads += 1
            if self.queued and self.status_reads > self.polls_before_apply:
                self.episode_id += 1
                self.queued = False
            return status_reply(self.episode_id)
        raise AssertionError(f"unexpected request {request!r}")


class ParseEpisodeIdTest(unittest.TestCase):
    def test_reads_the_boundary_the_simulator_publishes(self) -> None:
        self.assertEqual(parse_episode_id(status_reply(7)), 7)

    def test_refuses_a_reply_that_is_not_a_json_object(self) -> None:
        for reply in (b"reset_queued", b"[]", b"not json", b'{"episode_id": "7"}'):
            with self.subTest(reply=reply):
                with self.assertRaises(ResetError):
                    parse_episode_id(reply)

    def test_refuses_a_boolean_episode_id(self) -> None:
        # ``isinstance(True, int)`` is True in Python; a bool is not a boundary.
        with self.assertRaises(ResetError):
            parse_episode_id(b'{"episode_id": true}')


class RequestResetAppliedTest(unittest.TestCase):
    def test_returns_only_after_the_episode_id_advances(self) -> None:
        client = FakeEndpointResetClient(start_episode=3, polls_before_apply=2)
        applied = client.request_reset_applied(applied_timeout_ms=2000)
        self.assertEqual(applied, 4)
        self.assertEqual(client.requests[0], STATUS_REQUEST)
        self.assertEqual(client.requests[1], RESET_REQUEST)
        self.assertGreaterEqual(client.status_reads, 3)

    def test_accepts_a_reset_the_loop_applied_before_the_first_post_request_poll(self) -> None:
        # The loop can consume the flag before the caller's first status read
        # *after* the request; the boundary is read before the request precisely
        # so that this case still shows an advance.
        client = FakeEndpointResetClient(start_episode=5, polls_before_apply=0)
        self.assertEqual(client.request_reset_applied(applied_timeout_ms=2000), 6)

    def test_fails_closed_when_the_loop_never_applies_the_reset(self) -> None:
        client = FakeEndpointResetClient(start_episode=9, polls_before_apply=10**9)
        with self.assertRaises(ResetError) as caught:
            client.request_reset_applied(applied_timeout_ms=30)
        self.assertIn("did not apply", str(caught.exception))

    def test_a_refused_reset_propagates(self) -> None:
        class Refusing(IsaacResetClient):
            def _request(self, request, expected, refused, what, *, timeout_ms=None) -> bytes:
                if request == RESET_REQUEST:
                    raise ResetError("refused")
                return status_reply(1)

        with self.assertRaises(ResetError):
            Refusing("tcp://localhost:5559").request_reset_applied(applied_timeout_ms=100)

    def test_legacy_request_reset_still_returns_on_the_queue_ack(self) -> None:
        client = FakeEndpointResetClient(start_episode=3, polls_before_apply=10**9)
        client.request_reset()
        # No status read at all: the legacy path is unchanged.
        self.assertEqual(client.status_reads, 0)
        self.assertEqual(client.requests, [RESET_REQUEST])


class ModelControllerResetGateTest(unittest.TestCase):
    """Only a session that owns the settle waits for the applied boundary.

    The controller must keep the legacy queue-acknowledged reset for every
    default run; the strict wait is what the opt-in settle commands need, because
    their stream is aimed at the controller's first post-reset decode.
    """

    def _controller(self, *, initial_pose=None, warmstart=None):
        from humanoid_lab.psi0_bridge.model_controller import ModelController

        controller = ModelController.__new__(ModelController)
        controller.reset_endpoint = "tcp://localhost:5559"
        controller.initial_pose = initial_pose
        controller.warmstart = warmstart
        controller._log = lambda message: None
        return controller

    def _patched_client(self, calls):
        class Recording(IsaacResetClient):
            def __init__(self, endpoint, *, timeout_ms=2000):
                super().__init__(endpoint, timeout_ms=timeout_ms)

            def request_reset(self) -> None:
                calls.append("queued")

            def request_reset_applied(self, **kwargs) -> int:
                calls.append("applied")
                return 42

        return Recording

    def test_the_default_session_keeps_the_queue_acknowledged_reset(self) -> None:
        calls: list[str] = []
        controller = self._controller()
        with mock.patch.object(model_controller, "IsaacResetClient",
                               self._patched_client(calls)):
            controller._request_reset()
        self.assertEqual(calls, ["queued"])

    def test_each_opt_in_settle_command_waits_for_the_applied_reset(self) -> None:
        for label, kwargs in (
            ("initial_pose", {"initial_pose": object()}),
            ("warmstart", {"warmstart": object()}),
        ):
            with self.subTest(settle=label):
                calls: list[str] = []
                controller = self._controller(**kwargs)
                with mock.patch.object(model_controller, "IsaacResetClient",
                                       self._patched_client(calls)):
                    controller._request_reset()
                self.assertEqual(calls, ["applied"])


if __name__ == "__main__":
    unittest.main()
