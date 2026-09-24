import time

import pytest

from humanoid_lab.robot_runtime.sonic_status import decode_status


def test_status_is_robot_reported_and_expires():
    status = decode_status(
        b'sonic_status {"mode":"STREAMED_MOTION","valid_token_age_ms":44,"sequence":7}',
        received_at=time.monotonic() - 0.7,
    )
    assert status.mode == "STREAMED_MOTION"
    assert status.valid_token_age_ms == 44
    assert not status.fresh()


@pytest.mark.parametrize("payload", [
    b'other {"mode":"PLANNER","sequence":1}',
    b'sonic_status {"mode":"IDLE","sequence":1}',
    b'sonic_status {"mode":"PLANNER","valid_token_age_ms":-1,"sequence":1}',
    b'sonic_status {"mode":"PLANNER","sequence":0}',
])
def test_invalid_status_is_rejected(payload):
    with pytest.raises(ValueError):
        decode_status(payload)
