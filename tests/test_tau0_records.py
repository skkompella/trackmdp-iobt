"""
Tests for the tau=0 observation record (grid_env_rect.last_tau0) and the
opt-in rollout-worker buffer (grid_environment_rect.drain_tau0) that feed the
transition-stream detector from PPO.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.grid_env_rect import learning_grid_sarsa_0      # noqa: E402
from src.core.grid_wrapper_rect import grid_environment_rect  # noqa: E402

CUM = [0.2817, 0.5633, 0.845, 0.995]


def make(record=False, budget="none", k=6, seed=1):
    q = learning_grid_sarsa_0(0, 5, 5, 4, CUM, k, k, 1, 1,
                              movement="original", seed=seed, sensor_budget=budget)
    cfg = {"qobj": q, "time_limit_schedule": [2000], "time_limit_max": 1,
           "seed": seed, "max_ep_steps": 0}
    if record:
        cfg["record_tau0"] = True
    return grid_environment_rect(cfg)


def drive(env, n=5000, seed=0):
    """Step with random actions; yield (pre_state, pre_delay, step result, record)."""
    rng = np.random.default_rng(seed)
    ge = env.qobj.grid_env
    for _ in range(n):
        pre_state, pre_delay = env.current_state, env.time_delay
        out = env.step(rng.integers(0, 2, size=env.action_space.shape[0]))
        yield pre_state, pre_delay, out, ge.last_tau0
        if out[2]:
            env.reset()


def chebyshev(a, b, ncols=5):
    return max(abs(a % ncols - b % ncols), abs(a // ncols - b // ncols))


def test_records_only_at_tau0_tracked_steps():
    env = make()
    seen = 0
    for pre_state, pre_delay, _, rec in drive(env):
        tracked0 = pre_state != env.missing_state and pre_delay == 0
        assert (rec is not None) == tracked0
        if rec is not None:
            assert rec[0] == pre_state // 2
            seen += 1
    assert seen > 500


def test_found_implies_in_active_and_detection():
    env = make()
    for _, _, out, rec in drive(env):
        if rec is None:
            continue
        cell, active, found = rec
        detected = env.time_delay == 0          # delay after the step
        assert (found is not None) == detected
        if found is not None:
            assert found in active
            assert env.current_state == found * 2


def test_active_cells_on_grid_in_block():
    env = make()
    corner, edge = {0, 4, 20, 24}, set()
    edge = {c for c in range(25) if (c % 5 in (0, 4) or c // 5 in (0, 4))} - corner
    for _, _, _, rec in drive(env):
        if rec is None:
            continue
        cell, active, _ = rec
        assert list(active) == sorted(set(active))
        assert all(0 <= j < 25 and chebyshev(cell, j) <= 1 for j in active)
        if cell in corner:
            assert len(active) <= 4
        elif cell in edge:
            assert len(active) <= 6


def test_budget_env_records_kept_cells():
    env = make(budget="env", k=2)
    n = 0
    for _, _, _, rec in drive(env):
        if rec is not None:
            assert len(rec[1]) <= 2
            n += 1
    assert n > 0


def test_behaviour_unchanged_by_recording():
    a = [out[:4] for _, _, out, _ in drive(make(record=False))]
    b = [out[:4] for _, _, out, _ in drive(make(record=True))]
    for x, y in zip(a, b):
        assert x[1:] == y[1:]
        assert x[0][0] == y[0][0] and x[0][1] == y[0][1]
        assert np.array_equal(x[0][2], y[0][2])


def test_wrapper_buffer_opt_in():
    off = make(record=False)
    list(drive(off, 500))
    assert off.drain_tau0() == []

    on = make(record=True)
    recs = [rec for *_, rec in drive(on, 2000) if rec is not None]
    on.reset()                                   # a reset keeps the buffer
    got = on.drain_tau0()
    assert got == recs and len(got) > 0
    assert on.drain_tau0() == []
