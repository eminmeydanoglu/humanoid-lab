"""Unitree ordering and local command-disabled scheduler contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flux_dex3.executor import ChunkExecutor
from flux_dex3.mapping import JOINT_NAMES, measured_state, split_action


def _motors(count, start=0):
    return SimpleNamespace(motor_state=[SimpleNamespace(q=float(start + i)) for i in range(count)])


def test_manifest_matches_unitree_motor_order():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "outputs/dex3/index/manifest.json").read_text())
    assert tuple(manifest["state_names"]) == tuple(manifest["action_names"]) == JOINT_NAMES
    values = measured_state(_motors(35), _motors(7, 100), _motors(7, 200))
    assert values == tuple(map(float, range(15, 29))) + tuple(map(float, range(100, 107))) + tuple(map(float, range(200, 207)))
    arm, left, right = split_action(values)
    assert [arm[i] for i in range(15, 29)] == list(map(float, range(15, 29)))
    assert [left[i] for i in range(7)] == list(map(float, range(100, 107)))
    assert [right[i] for i in range(7)] == list(map(float, range(200, 207)))


def test_mapping_rejects_missing_motors_and_nonfinite_values():
    with pytest.raises(ValueError):
        measured_state(_motors(28), _motors(7), _motors(7))
    with pytest.raises(ValueError):
        measured_state(_motors(35), _motors(7), _motors(6))
    with pytest.raises(ValueError):
        split_action([float("nan")] + [0] * 27)


def test_chunk_schedules_30hz_and_holds_its_last_row_until_the_next_chunk():
    exe = ChunkExecutor()
    exe.start("episode-1")
    chunk = np.repeat(np.arange(32, dtype=np.float32)[:, None] / 10, 28, axis=1)
    exe.accept("episode-1", 0, 0.05, chunk, now=100.0)
    assert exe.tick(100.0)[0] == 0
    assert exe.tick(100.101)[0] == pytest.approx(0.3)
    assert exe.tick(100.0 + 31.2 / 30)[0] == pytest.approx(3.1)
    assert exe.tick(100.0 + 32 / 30)[0] == pytest.approx(3.1)  # last executed row, repeated
    assert exe.reason == "holding last target" and exe.session == "episode-1"
    assert exe.held_since == pytest.approx(100.0 + 32 / 30)
    assert exe.tick(100.0 + 40 / 30)[0] == pytest.approx(3.1)
    exe.accept("episode-1", 1, 0.4, np.full((32, 28), 2.0, np.float32), now=101.4)
    assert exe.held_since is None and exe.reason == "running"
    assert exe.tick(101.4)[0] == pytest.approx(2.0)  # the late chunk starts at its first row


def test_chunk_queues_next_and_rejects_stale_or_invalid():
    exe = ChunkExecutor(limits=[(-2, 2)] * 28)
    exe.start("episode")
    chunk = np.full((32, 28), 1, dtype=np.float32)
    exe.accept("episode", 0, 0.1, chunk, now=10)
    exe.accept("episode", 1, 0.2, chunk * 1.5, now=10.8)
    with pytest.raises(ValueError):
        exe.accept("episode", 2, 0.2, chunk, now=10.9)
    assert exe.tick(10 + 32 / 30)[0] == 1.5
    with pytest.raises(ValueError):
        exe.accept("episode", 1, 0.2, chunk, now=11.0)
    with pytest.raises(ValueError):
        exe.accept("wrong", 2, 0.2, chunk, now=11.0)
    with pytest.raises(ValueError):
        exe.accept("episode", 2, 0.2, chunk * 3, now=11.0)
    with pytest.raises(ValueError):
        exe.accept("episode", 2, 1.3, chunk, now=11.0)
    exe.stop()
    assert exe.tick(11.2) is None
