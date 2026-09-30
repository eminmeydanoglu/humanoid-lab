import numpy as np
import pytest

from humanoid_lab.evaluation.policy_metrics import action_metrics, feedback_state, video_metrics


def test_partial_reference_metrics():
    prediction = np.ones((64, 28), dtype=np.float32)
    reference = np.zeros((32, 28), dtype=np.float32)
    result = action_metrics(prediction, reference, prediction * 2, reference)
    assert result['reference_steps'] == 32
    assert result['mae'] == result['rmse'] == 1
    assert result['normalized_mae'] == result['normalized_rmse'] == 2
    assert result['per_joint_mae'] == [1] * 28
    assert action_metrics(prediction, [], prediction, []) == {'reference_steps': 0}


def test_dream_feedback_is_last_physical_action_and_copy():
    actions = np.arange(32 * 28, dtype=np.float32).reshape(32, 28)
    state = feedback_state(actions)
    np.testing.assert_array_equal(state, actions[-1])
    state[:] = 0
    assert actions[-1, 0] != 0
    actions[-1, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        feedback_state(actions)


def test_video_metrics_before_compression():
    prediction = np.ones((32, 2, 2, 3), dtype=np.uint8)
    reference = np.zeros((10, 2, 2, 3), dtype=np.uint8)
    result = video_metrics(prediction, reference)
    assert result['reference_frames'] == 10
    assert result['mse'] == result['mae'] == 1
    assert video_metrics(reference, reference)['psnr_db'] is None
