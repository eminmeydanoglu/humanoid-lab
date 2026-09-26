"""Local 30 Hz chunk scheduler. Publishing is owned by the ROS node.

A chunk that runs out before the next prediction lands keeps emitting its last row
until the new chunk arrives; the node's network timeout bounds that hold.
"""

import math
import time

import numpy as np


class ChunkExecutor:
    def __init__(self, *, rate_hz=30.0, max_chunk_age_s=1.2, limits=None):
        self.period = 1.0 / rate_hz
        self.max_chunk_age_s = max_chunk_age_s
        self.limits = limits
        self.stop()

    def stop(self):
        self.session = None
        self.current = None
        self.next_chunk = None
        self.last_seq = -1
        self.started = None
        self.index = 0
        self.held_since = None
        self.reason = "stopped"

    def start(self, session):
        self.stop()
        self.session = session
        self.reason = "waiting for first chunk"

    def accept(self, session, seq, observation_age_s, actions, now=None):
        now = time.monotonic() if now is None else now
        chunk = np.asarray(actions)
        if session != self.session or not isinstance(seq, int) or seq <= self.last_seq:
            raise ValueError("stale session or sequence")
        if not math.isfinite(observation_age_s) or observation_age_s < 0 or observation_age_s > self.max_chunk_age_s:
            raise ValueError("stale observation")
        if chunk.shape != (32, 28) or not np.issubdtype(chunk.dtype, np.number) or not np.isfinite(chunk).all():
            raise ValueError("invalid action chunk")
        if self.limits is not None:
            if len(self.limits) != 28 or any(len(pair) != 2 for pair in self.limits):
                raise ValueError("invalid hardware limit configuration")
            low, high = np.asarray(self.limits, np.float32).T
            if np.any(chunk < low) or np.any(chunk > high):
                raise ValueError("action exceeds joint limit")
        else:
            # Dry-run only: the dataset and installed Dex3 hardware may have different strokes.
            if np.any(np.abs(chunk) > 3.2):
                raise ValueError("action exceeds coarse dry-run bound")
        if self.current is not None and self.next_chunk is not None:
            raise ValueError("next chunk is already queued")
        chunk = np.array(chunk, dtype=np.float32, order="C", copy=True)
        if self.current is None or self.held_since is not None:
            # First chunk of the session, or the previous one ran out while this prediction
            # was in flight: either way the new chunk starts at its own first row now.
            self.current, self.next_chunk = chunk, None
            self.started, self.index, self.held_since = now, 0, None
        else:
            self.next_chunk = chunk
        self.last_seq = seq
        self.reason = "running"

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if self.session is None or self.current is None:
            return None
        due = int(max(0, (now - self.started + 1e-9) / self.period))
        if due >= 32:
            if self.next_chunk is None:
                if self.held_since is None:
                    self.held_since = now
                self.index = 31
                self.reason = "holding last target"
                return self.current[31].copy()
            self.current, self.next_chunk = self.next_chunk, None
            self.started, self.held_since, due = now, None, 0
        self.index = due
        return self.current[due].copy()
