"""
Tests for the tau=0 hook in src/core/online_schedule.run_episode: it feeds the
transition-stream detector, lets a prober replace the action, keeps probe
transitions out of learning, and survives the object leaving the grid.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.online_schedule import run_episode  # noqa: E402
from src.core.random_transition_matrices import parse_matrix_spec  # noqa: E402
from src.core.switching_grid_env import SwitchingTransitionGridEnv  # noqa: E402
from src.core.tabular_td import TabularTDAgent  # noqa: E402
from src.core.transition_detector import row_supports  # noqa: E402

CFG = {"time_limit": 1, "missing_state": 19, "max_ep_steps": 20,
       "max_sensors": 5}
SUPPORTS = row_supports("grid", 3, 3)


def _setup(exit_prob=0.0):
    _, mats = parse_matrix_spec("R0", "grid", 20261006, 1.0, 3, 3)
    env = SwitchingTransitionGridEnv(3, 3, 5, 5, 19, 1, mats, seed=0,
                                     no_moore_constraint=True,
                                     exit_prob=exit_prob)
    agent = TabularTDAgent(time_limit=1, max_sensors=5, missing_state=19,
                           seed=0, n_nodes=9, grid_width=9,
                           action_mode="local", grid_shape=(3, 3),
                           optimistic_init=5.0)
    return agent, env


class RecordingHook:
    def __init__(self, probe_all=False):
        self.probe_all = probe_all
        self.calls = []

    def probe(self, cell):
        return list(SUPPORTS[cell]) if self.probe_all else None

    def observe(self, cell, watched, found, probed):
        self.calls.append((cell, list(watched), found, probed))


def test_no_hook_matches_passive_hook():
    a1, e1 = _setup()
    r1 = run_episode(a1, e1, CFG)
    a2, e2 = _setup()
    hook = RecordingHook()
    r2 = run_episode(a2, e2, CFG, hook=hook)
    for key in ("accuracy", "sensors", "reward", "detections", "cells"):
        assert r1[key] == r2[key]
    assert r1["n_probes"] == 0 and r2["n_probes"] == 0
    assert all(probed is False for *_, probed in hook.calls)


def test_hook_calls_align_with_tau0_steps():
    agent, env = _setup()
    hook = RecordingHook()
    res = run_episode(agent, env, CFG, hook=hook)
    cells = res["cells"]
    expected = [cells[t - 1] for t in range(1, len(cells))
                if cells[t - 1] is not None]
    assert [c for c, *_ in hook.calls] == expected
    for cell, watched, found, _ in hook.calls:
        assert found is None or found in watched


def test_always_probe_full_support(monkeypatch):
    agent, env = _setup()
    n_observe = []
    real = agent.observe
    monkeypatch.setattr(agent, "observe",
                        lambda *a, **k: (n_observe.append(1), real(*a, **k)))
    hook = RecordingHook(probe_all=True)
    res = run_episode(agent, env, CFG, hook=hook)
    assert res["n_probes"] == 19
    assert res["accuracy"] == pytest.approx(19 / 20)
    assert len(n_observe) == 1          # only the missing-state step is learned
    assert all(probed for *_, probed in hook.calls)


def test_exited_object_ends_episode_without_samples():
    agent, env = _setup(exit_prob=1.0)
    hook = RecordingHook()
    res = run_episode(agent, env, CFG, hook=hook)
    assert hook.calls == []
    assert len(res["detections"]) < CFG["max_ep_steps"]
