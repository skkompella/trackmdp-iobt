"""
Tests for the pure metric helpers in examples/train_multiloop_online.py, plus
one end-to-end rollout check of the tabular agent against MultiLoopIoBTEnv.

Kept ray-free: only the metric functions and run_episode are imported, and
neither touches RLlib.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.core.multiloop_iobt_env import MultiLoopIoBTEnv  # noqa: E402
from src.core.tabular_td import TabularTDAgent            # noqa: E402


def _load_online_module():
    """Import the script by path — examples/ is not a package."""
    path = os.path.join(ROOT, "examples", "train_multiloop_online.py")
    spec = importlib.util.spec_from_file_location("_online", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


online = _load_online_module()


# ---------------------------------------------------------------------------
# adaptation_latency
# ---------------------------------------------------------------------------

def test_latency_zero_when_already_at_steady_state():
    assert online.adaptation_latency([0.9] * 30, epsilon=0.05) == 0


def test_latency_detects_a_dip_then_recovery():
    # 10 bad episodes, then 30 good ones
    series = [0.2] * 10 + [0.9] * 30
    lat = online.adaptation_latency(series, epsilon=0.05)
    assert lat is not None
    assert 8 <= lat <= 12


def test_latency_is_relative_to_where_the_segment_ENDS_not_to_an_optimum():
    """
    Steady state is the segment's own tail, so a segment that degrades scores
    latency 0 — it was 'already at steady state' from the first episode.

    This is a real limitation of the metric and the reason the summary also
    reports mean accuracy: a low latency alone does not mean the learner did
    well, only that it reached its own plateau quickly.
    """
    degrading = [0.9] * 5 + [0.1] * 35
    assert online.adaptation_latency(degrading, epsilon=0.01) == 0


def test_latency_none_for_too_short_a_segment():
    assert online.adaptation_latency([0.5, 0.5], epsilon=0.05) is None


def test_larger_epsilon_never_increases_latency():
    series = [0.1] * 12 + [0.9] * 28
    tight = online.adaptation_latency(series, epsilon=0.01)
    loose = online.adaptation_latency(series, epsilon=0.5)
    assert loose <= tight


# ---------------------------------------------------------------------------
# detector_scores
# ---------------------------------------------------------------------------

def test_perfect_detection():
    s = online.detector_scores(fired_episodes=[10, 20],
                               switch_episodes=[10, 20], tolerance=3)
    assert s["true_positive"] == 2
    assert s["false_alarms"] == 0
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0


def test_detection_within_tolerance_counts():
    s = online.detector_scores([12], [10], tolerance=3)
    assert s["true_positive"] == 1


def test_detection_outside_tolerance_is_a_false_alarm():
    s = online.detector_scores([20], [10], tolerance=3)
    assert s["true_positive"] == 0
    assert s["false_alarms"] == 1


def test_detection_before_the_switch_does_not_count():
    """Firing early is not clairvoyance, it is a false alarm."""
    s = online.detector_scores([8], [10], tolerance=3)
    assert s["true_positive"] == 0


def test_missed_switch_lowers_recall():
    s = online.detector_scores([10], [10, 20], tolerance=3)
    assert s["recall"] == pytest.approx(0.5)


def test_no_detections_reports_none_precision():
    s = online.detector_scores([], [10], tolerance=3)
    assert s["precision"] is None
    assert s["recall"] == 0.0


# ---------------------------------------------------------------------------
# run_episode integration
# ---------------------------------------------------------------------------

TL = 1
MISSING = 16 * (TL + 1) + 1
LOOPS = [[0, 4, 5, 3, 6, 2, 1], [7, 8, 9]]


def make_pair(max_sensors=2, **agent_kw):
    env = MultiLoopIoBTEnv(max_sensors, max_sensors, MISSING, TL, LOOPS,
                           seed=0, no_moore_constraint=True,
                           reward_preset="grid")
    agent = TabularTDAgent(time_limit=TL, max_sensors=max_sensors,
                           missing_state=MISSING, seed=0, **agent_kw)
    cfg = {"time_limit": TL, "missing_state": MISSING,
           "max_ep_steps": 50, "max_sensors": max_sensors}
    return agent, env, cfg


def test_run_episode_returns_sane_metrics():
    agent, env, cfg = make_pair()
    env.force_loop(0)
    res = online.run_episode(agent, env, cfg)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 < res["sensors"] <= cfg["max_sensors"]
    assert len(res["detections"]) == cfg["max_ep_steps"]


def test_accuracy_cannot_reach_one():
    """The first step is always in the missing state, so it can never count."""
    agent, env, cfg = make_pair()
    env.force_loop(0)
    for _ in range(20):
        res = online.run_episode(agent, env, cfg)
        assert res["accuracy"] <= (cfg["max_ep_steps"] - 1) / cfg["max_ep_steps"]


def test_sensors_never_exceed_the_budget():
    agent, env, cfg = make_pair(max_sensors=1)
    env.force_loop(0)
    for _ in range(10):
        assert online.run_episode(agent, env, cfg)["sensors"] <= 1.0


def test_learning_improves_accuracy_on_a_single_loop():
    """The core sanity check: a table must learn a single fixed loop."""
    agent, env, cfg = make_pair(max_sensors=2, alpha=0.2, epsilon=0.1,
                                optimistic_init=1.0)
    env.force_loop(0)
    early = np.mean([online.run_episode(agent, env, cfg)["accuracy"]
                     for _ in range(20)])
    for _ in range(300):
        online.run_episode(agent, env, cfg)
    late = np.mean([online.run_episode(agent, env, cfg, explore=False)["accuracy"]
                    for _ in range(20)])
    assert late > early + 0.2, f"early={early:.3f} late={late:.3f}"


def test_no_learning_when_learn_is_false():
    agent, env, cfg = make_pair()
    env.force_loop(0)
    before = agent.snapshot()
    online.run_episode(agent, env, cfg, learn=False)
    assert np.allclose(agent.q, before)
