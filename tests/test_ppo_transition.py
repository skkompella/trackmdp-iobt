"""
Tests for src/core/ppo_transition.py — draining tau=0 records from RLlib
rollout workers into the transition-stream detector — and the Moore supports.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.core.ppo_transition import TransitionMonitor, drain_records  # noqa: E402
from src.core.transition_detector import row_supports                 # noqa: E402

CUM = [0.2817, 0.5633, 0.845, 0.995]


def test_supports_grid_moore():
    s = row_supports("grid_moore", 5, 5)
    assert s[0] == (0, 1, 5, 6)
    assert s[12] == (6, 7, 8, 11, 12, 13, 16, 17, 18)
    assert sum(len(v) for v in s.values()) == 169
    assert sum(len(v) for v in row_supports("grid_moore", 10, 10).values()) == 784


def test_moore_supports_cover_original_table():
    from src.core.grid_env_rect import grid_env_rect
    env = grid_env_rect(5, 5, 4, CUM, 6, 6, 51, 1, movement="original", seed=1)
    s = row_supports("grid_moore", 5, 5)
    for i, dests in env._orig_dests.items():
        assert set(dests) <= set(s[i])


class FakeGroup:
    def __init__(self, batches):
        self.batches = batches          # list over calls of list-per-worker

    def foreach_env(self, fn):
        return self.batches.pop(0)


class FakeAlgo:
    def __init__(self, batches):
        self.env_runner_group = FakeGroup(batches)


def test_drain_records_flattens_workers_in_order():
    algo = FakeAlgo([[[], [[(0, (0, 1), 1)], [(5, (5,), None)]], [[(6, (6,), 6)]]]])
    assert drain_records(algo) == [(0, (0, 1), 1), (5, (5,), None), (6, (6,), 6)]


def test_monitor_feeds_each_record_once():
    rec = (12, (7, 12, 13), 12)
    algo = FakeAlgo([[[[rec] * 10]], [[[rec] * 5]]])
    m = TransitionMonitor(row_supports("grid_moore", 5, 5))
    m.update(algo, iteration=1)
    m.update(algo, iteration=2)
    assert m.n_records == 15
    assert m.detector.counts(12)[12] == 15 and m.detector.counts(12)[7] == 15


def test_monitor_alarm_clears_detector_and_logs():
    rng = np.random.default_rng(0)
    m = TransitionMonitor(row_supports("grid_moore", 5, 5), save_streams=True)
    # j = 13 is taken with probability 0.28, then never again
    pre = [(12, (13,), 13 if rng.random() < 0.28 else None) for _ in range(1500)]
    post = [(12, (13,), None)] * 1500
    assert m.feed(pre, iteration=1, env_steps=4000) is None
    alarm = m.feed(post, iteration=2, env_steps=8000)
    assert alarm is not None and alarm["iteration"] == 2
    assert alarm["cell"] == 12 and alarm["next_cell"] == 13
    assert alarm["env_steps"] == 8000
    assert len(m.alarms) == 1
    assert sum(m.detector.counts(12).values()) == 0      # cleared
    # every record was saved, also those after the alarm in that iteration
    assert len(m.streams[(12, 13)]) == 3000


def test_monitor_save_round_trip(tmp_path):
    m = TransitionMonitor(row_supports("grid_moore", 5, 5), save_streams=True,
                          save_records=True)
    m.feed([(0, (0, 1), 1), (0, (0, 1, 6), None), (24, (24,), 24)], iteration=3)
    m.save(str(tmp_path))
    al = json.loads((tmp_path / "alarms.json").read_text())
    assert al["n_streams"] == 169 and al["n_records"] == 3 and al["alarms"] == []
    z = np.load(tmp_path / "streams.npz")
    assert list(z["0_1"]) == [1, 0] and list(z["0_6"]) == [0]
    assert z["0_1"].dtype == np.int8
    r = np.load(tmp_path / "records.npz")
    assert list(r["iteration"]) == [3, 3, 3]
    assert list(r["cell"]) == [0, 0, 24] and list(r["found"]) == [1, -1, 24]
    assert int(r["active"][1]) == (1 << 0) | (1 << 1) | (1 << 6)



def test_real_rllib_workers_are_drained():
    """One PPO iteration on 5x5 with two workers: records reach the driver."""
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from src.core.grid_env_rect import learning_grid_sarsa_0
    from src.core.grid_wrapper_rect import grid_environment_rect

    q = learning_grid_sarsa_0(0, 5, 5, 4, CUM, 6, 6, 1, 1,
                              movement="original", seed=1)
    cfg = {"qobj": q, "time_limit_schedule": [2000], "time_limit_max": 1,
           "seed": 1, "max_ep_steps": 0, "record_tau0": True}
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath(ROOT)}})
    try:
        pc = (PPOConfig().environment(grid_environment_rect, env_config=cfg)
              .framework("torch").env_runners(num_env_runners=2)
              .training(train_batch_size=400, minibatch_size=100, num_epochs=1)
              .api_stack(enable_rl_module_and_learner=False,
                         enable_env_runner_and_connector_v2=False))
        pc.normalize_actions = False
        algo = pc.build()
        m = TransitionMonitor(row_supports("grid_moore", 5, 5))
        algo.train()
        m.update(algo, iteration=1)
        n1 = m.n_records
        assert n1 > 0
        assert m.update(algo, iteration=1) is None and m.n_records == n1  # drained
        algo.stop()
    finally:
        ray.shutdown()
