from __future__ import annotations

import asyncio
import threading
import unittest

import numpy as np

from humanoid_lab.psi0_bridge.policy_clock import PolicyTime
from humanoid_lab.psi0_bridge.session import Session, SessionConfig, SessionError
from humanoid_lab.psi0_bridge.sim_clock import SimulationClockPacer


class FakePublisher:
    def halt(self): pass
    def resume(self): pass
    def close(self): pass


class FakeMonitor:
    def set_active(self, active): pass
    def status(self): return {}
    def frame(self, **kwargs): return None


class SequenceClock:
    mode = "simulation"

    def __init__(self, values):
        self.values = iter(values)
        self.last = None

    def now(self):
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        if self.last is None:
            raise RuntimeError("empty fake clock")
        return PolicyTime(*self.last)


class Psi0SessionSimulationClockTest(unittest.IsolatedAsyncioTestCase):
    def session(self, clock) -> Session:
        session = Session(
            SessionConfig(control_hz=30.0), monitor=FakeMonitor(),
            publisher=FakePublisher(), policy_clock=clock,
        )
        session._generation = 1
        return session

    async def test_fractional_30hz_schedule_on_quantized_50hz_clock(self) -> None:
        clock = SequenceClock([(0.0, 0), (0.02, 0), (0.04, 0), (0.06, 0), (0.08, 0)])
        session = self.session(clock)
        pacer = SimulationClockPacer(30.0)
        await session._wait_simulation_tick(1, pacer)
        await session._wait_simulation_tick(1, pacer)
        await session._wait_simulation_tick(1, pacer)
        self.assertAlmostEqual(clock.last[0], 0.08)

    async def test_running_session_fails_closed_on_clock_reset(self) -> None:
        clock = SequenceClock([(1.0, 0), (0.0, 1)])
        session = self.session(clock)
        pacer = SimulationClockPacer(30.0)
        await session._wait_simulation_tick(1, pacer)
        with self.assertRaisesRegex(SessionError, "reset"):
            await session._wait_simulation_tick(1, pacer)

    async def test_jittered_reader_returns_tick_debt_without_false_failure(self) -> None:
        clock = SequenceClock([(0.0, 0), (0.02, 0), (0.08, 0)])
        session = self.session(clock)
        pacer = SimulationClockPacer(30.0)
        self.assertEqual(await session._wait_simulation_tick(1, pacer), 1)
        self.assertEqual(await session._wait_simulation_tick(1, pacer), 2)


class Psi0RtcTickBasisTest(unittest.TestCase):
    def test_rtc_delay_ticks_advance_only_on_simulation_scheduled_requests(self) -> None:
        from psi.deploy import serve_psi0_sonic as upstream

        controller = upstream.RealTimeChunkController.__new__(upstream.RealTimeChunkController)
        controller.C = threading.Condition(threading.Lock())
        controller.t = 0
        controller.o_cur = None
        controller.A_cur = np.arange(30, dtype=np.float32)[:, None]

        pacer = SimulationClockPacer(30.0)
        returned = []
        for sim_s in (0.0, 0.0, 0.02, 0.04, 0.04, 0.06, 0.08):
            update = pacer.update(sim_s, 0)
            if update.rows_due:
                returned.append(controller.step({}).item())
        self.assertEqual(returned, [0.0, 1.0, 2.0])
        self.assertEqual(controller.t, 3)


if __name__ == "__main__":
    unittest.main()
