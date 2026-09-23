#!/usr/bin/env python3
"""Launch the pinned psi0 SONIC server with a functional RTC mode switch."""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch

from humanoid_lab.psi0_bridge.open_loop import OpenLoopChunkController
from humanoid_lab.psi0_bridge.policy_clock import PolicyClock
from humanoid_lab.psi0_bridge.sim_clock import SimulationClockPacer, clamp_rtc_delay
from psi.deploy import serve_psi0_sonic as upstream


class SimulationClockRtcController(upstream.RealTimeChunkController):
    """RTC with separate executed-row and elapsed-simulation delay counters."""

    def __init__(self, *args, **kwargs):
        self.policy_clock = PolicyClock.from_environment()
        self.delay_pacer = SimulationClockPacer(1.0 / upstream.CTRL_PERIOD_SEC)
        # `d` is not "how late the replan is"; it is the RTC condition the
        # checkpoint was trained with, and `max_delay` is its declared bound.  The
        # delay buffer `self.Q` shares that bound for the same reason (the control
        # loop pushes `self.t`, which the schedule keeps under it).
        self.max_delay = int(kwargs.get("max_delay", 8))
        super().__init__(*args, **kwargs)

    def _fatal(self, exc: BaseException) -> None:
        upstream.update_dashboard(
            status="server crashed", error=f"{type(exc).__name__}: {exc}"
        )
        os._exit(1)

    def _inference_loop(self):
        while not self._stop_event.is_set():
            with self.C:
                try:
                    while self.t < self.s_min and not self._stop_event.is_set():
                        self.C.wait()
                    if self._stop_event.is_set():
                        return
                    executed = self.t
                    observation = copy.deepcopy(self.o_cur)
                    delay = max(self.Q)
                    previous = upstream.shift_and_pad_action_chunk(self.A_cur, executed)
                    inference_start = self.policy_clock.now()
                    wall_start = time.perf_counter()
                    self.C.release()
                    try:
                        new_actions = self._predict_action_rtc(
                            observation, previous, delay, executed
                        )
                        inference_end = self.policy_clock.now()
                        infer_ms = (time.perf_counter() - wall_start) * 1000
                        elapsed_s = self.policy_clock.latency_seconds(
                            inference_start, inference_end
                        )
                        # Bound the simulated delay by the RTC horizon the
                        # checkpoint declares, not by the chunk length: a slow
                        # forward pass (seconds, for a 3B model) would otherwise
                        # hand `_predict_action_rtc` a delay of up to H-1 and
                        # freeze most of the new chunk against a stale prefix,
                        # while the trained path only ever saw delays up to
                        # `max_delay`.  The wall-clock path is unaffected.
                        elapsed_rows = clamp_rtc_delay(
                            self.delay_pacer.latency_rows(0.0, elapsed_s, self.H),
                            self.max_delay,
                        )
                    finally:
                        self.C.acquire()
                    self.A_cur = new_actions
                    self.t -= executed
                    self.Q.append(elapsed_rows)
                    self._max_infer_ms = max(self._max_infer_ms, infer_ms)
                    upstream.update_dashboard(
                        infer_ms=infer_ms,
                        max_infer_ms=self._max_infer_ms,
                        infer_executed=executed,
                        infer_delay=delay,
                        infer_ticks=elapsed_rows,
                        status="live",
                    )
                except Exception as exc:
                    self._fatal(exc)
                    return


def _simulation_clock_enabled() -> bool:
    return os.environ.get("HUMANOID_POLICY_CLOCK", "wall") == "simulation"


def _simulation_control_loop(server: upstream.Server, generation: int) -> None:
    clock = PolicyClock.from_environment()
    pacer = SimulationClockPacer(1.0 / upstream.CTRL_PERIOD_SEC)
    while not server._control_stop.is_set() and generation == server._conn_generation:
        policy_time = clock.now()
        update = pacer.update(policy_time.seconds, policy_time.generation)
        if update.reset:
            raise RuntimeError("Isaac simulation clock reset during a running psi0 server")
        if update.rows_due == 0:
            time.sleep(0.001)
            continue

        with server.obs_lock:
            observation = copy.deepcopy(server.latest_obs)
        if observation is None:
            break
        controller = server.controller
        if controller is None:
            break
        action = controller.step(observation)
        prediction = server._postprocess_action(action)
        with server.action_lock:
            server.latest_action = prediction
            server.action_version += 1
            upstream.update_dashboard(action_version=server.action_version)
        if server._action_ready_event is not None and generation == server._conn_generation:
            assert server._loop is not None
            server._loop.call_soon_threadsafe(server._action_ready_event.set)


def _control_loop(server: upstream.Server, generation: int) -> None:
    if not _simulation_clock_enabled():
        upstream._HUMANOID_ORIGINAL_CONTROL_LOOP(server, generation)
        return
    try:
        _simulation_control_loop(server, generation)
    except Exception as exc:
        upstream.update_dashboard(
            status="server crashed", error=f"{type(exc).__name__}: {exc}"
        )
        os._exit(1)


def _predict_unguided(server: upstream.Server, observation: dict) -> np.ndarray:
    return server.model.predict_action(
        observations=observation["imgs"],
        states=torch.from_numpy(observation["obs"]).to(server.device),
        traj2ds=None,
        instructions=observation["text_instructions"],
        num_inference_steps=8,
        pooled_projections=observation.get("pooled_projections"),
    )[0].float().detach().cpu().numpy()


def _serve(config) -> None:
    """Start the pinned SONIC transport in guided or independent mode."""
    upstream.overwatch.info("Server :: Initializing Policy")
    assert config.policy is not None, "which policy to serve?"
    assert type(config.ckpt_step) is int, "ckpt_step must be specified"
    upstream._check_port_free(config.host, config.port)
    server = upstream.Server(
        config.policy,
        upstream.Path(config.run_dir),
        config.ckpt_step,
        config.device,
        config.rtc,
        config.action_exec_horizon,
        dashboard=config.dashboard,
    )
    upstream.dprint("Server :: Spinning Up")
    server.run(config.host, config.port)


def _init_controller(server: upstream.Server, first_observation: dict):
    if server.enable_rtc:
        controller_type = (
            SimulationClockRtcController
            if _simulation_clock_enabled()
            else upstream.RealTimeChunkController
        )
        if _simulation_clock_enabled():
            print(
                "[policy-clock] psi0 controller cadence and RTC delay driven by simulation time",
                flush=True,
            )
        return controller_type(
            policy=server.model,
            o_first=first_observation,
            trained_rtc=server.trained_rtc,
            max_delay=server.rtc_max_delay,
            init_prev_action=server._pending_init_prev,
        )
    return OpenLoopChunkController(
        lambda observation: _predict_unguided(server, observation),
        first_observation,
        execution_horizon=server.Ta,
    )


def main() -> None:
    if not hasattr(upstream, "_HUMANOID_ORIGINAL_CONTROL_LOOP"):
        upstream._HUMANOID_ORIGINAL_CONTROL_LOOP = upstream.Server._control_loop
    upstream.Server._init_controller = _init_controller
    upstream.Server._control_loop = _control_loop
    upstream.serve = _serve
    upstream.main()


if __name__ == "__main__":
    main()
