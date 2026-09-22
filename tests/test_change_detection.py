"""
Tests for src/core/change_detection.py — Bernoulli-KL GLR detector and the
restart strategies it drives.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.change_detection import (  # noqa: E402
    GLRChangeDetector,
    RestartController,
    beta_threshold,
    kl_bernoulli,
)
from src.core.tabular_td import TabularTDAgent, state_index  # noqa: E402


def feed(det, values):
    """Push a sequence; return the index where it first fired, else None."""
    for i, v in enumerate(values):
        if det.update(v):
            return i
    return None


# ---------------------------------------------------------------------------
# Divergence and threshold
# ---------------------------------------------------------------------------

def test_kl_is_zero_for_identical_means():
    assert kl_bernoulli(0.4, 0.4) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_positive_and_symmetric_in_sign():
    assert kl_bernoulli(0.2, 0.8) > 0
    assert kl_bernoulli(0.8, 0.2) > 0


def test_kl_grows_with_separation():
    assert kl_bernoulli(0.5, 0.6) < kl_bernoulli(0.5, 0.9)


def test_kl_clamps_degenerate_inputs():
    """p or q at exactly 0 or 1 must not produce inf or nan."""
    for p in (0.0, 1.0):
        for q in (0.0, 0.5, 1.0):
            v = kl_bernoulli(p, q)
            assert np.isfinite(v)


def test_beta_grows_with_n_and_shrinks_with_delta():
    assert beta_threshold(100, 0.01) > beta_threshold(10, 0.01)
    assert beta_threshold(100, 0.001) > beta_threshold(100, 0.1)


def test_beta_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        beta_threshold(0, 0.01)
    with pytest.raises(ValueError):
        beta_threshold(10, 0.0)
    with pytest.raises(ValueError):
        beta_threshold(10, 1.0)


# ---------------------------------------------------------------------------
# Detector behaviour
# ---------------------------------------------------------------------------

def test_silent_on_a_stationary_stream():
    """A stationary Bernoulli stream must not trip the detector."""
    rng = np.random.default_rng(0)
    det = GLRChangeDetector(delta=0.01, min_samples=10)
    fired = feed(det, rng.random(400) < 0.8)
    assert fired is None


def test_fires_on_a_clear_mean_shift():
    rng = np.random.default_rng(1)
    det = GLRChangeDetector(delta=0.01, min_samples=10)
    stream = np.concatenate([rng.random(100) < 0.95, rng.random(100) < 0.15])
    fired = feed(det, stream)
    assert fired is not None
    assert fired >= 100, "must not fire before the change point"
    assert fired < 160, f"took {fired - 100} samples to react"


def test_latency_shrinks_as_the_shift_grows():
    rng = np.random.default_rng(2)

    def latency(post_rate):
        det = GLRChangeDetector(delta=0.01, min_samples=10)
        stream = np.concatenate([rng.random(80) < 0.95,
                                 rng.random(200) < post_rate])
        fired = feed(det, stream)
        return None if fired is None else fired - 80

    small, large = latency(0.70), latency(0.05)
    assert small is not None and large is not None
    assert large <= small


def test_does_not_fire_before_min_samples():
    det = GLRChangeDetector(delta=0.01, min_samples=25)
    # a blatant shift, but too few samples to be allowed to fire
    assert feed(det, [1] * 10 + [0] * 10) is None


def test_reset_clears_the_window():
    det = GLRChangeDetector(delta=0.01, min_samples=5)
    feed(det, [1] * 20)
    assert det.n_samples == 20
    det.reset()
    assert det.n_samples == 0


def test_detector_resets_itself_after_firing():
    """After a detection the next segment starts from a clean window."""
    rng = np.random.default_rng(3)
    det = GLRChangeDetector(delta=0.01, min_samples=10)
    stream = np.concatenate([rng.random(80) < 0.95, rng.random(80) < 0.1])
    fired = feed(det, stream)
    assert fired is not None
    assert det.n_samples < 5


def test_window_caps_memory():
    det = GLRChangeDetector(delta=0.01, min_samples=5, max_window=50)
    feed(det, [1] * 200)
    assert det.n_samples <= 50


def test_changepoints_are_recorded():
    rng = np.random.default_rng(4)
    det = GLRChangeDetector(delta=0.01, min_samples=10)
    feed(det, np.concatenate([rng.random(80) < 0.95, rng.random(80) < 0.1]))
    assert len(det.changepoints) == 1
    assert det.changepoints[0] >= 80


# ---------------------------------------------------------------------------
# Restart strategies
# ---------------------------------------------------------------------------

TL = 1


def make_agent():
    return TabularTDAgent(time_limit=TL, max_sensors=2, missing_state=33,
                          alpha=0.5, optimistic_init=0.0, seed=0)


def dirty(agent, value=7.0):
    agent.q[state_index(1, 0, TL), 3] = value
    return agent


def test_cold_restart_wipes_the_table():
    agent = dirty(make_agent())
    ctl = RestartController(agent, strategy="cold")
    ctl.restart(segment_reward=1.0)
    assert np.allclose(agent.q, agent.optimistic_init)


def test_warm_restart_restores_the_prior():
    agent = make_agent()
    prior = agent.snapshot()
    prior[0, 0] = 3.0
    ctl = RestartController(agent, strategy="warm", prior=prior)
    dirty(agent)
    ctl.restart(segment_reward=1.0)
    assert np.allclose(agent.q, prior)


def test_warm_restart_requires_a_prior():
    with pytest.raises(ValueError):
        RestartController(make_agent(), strategy="warm")


def test_library_archives_each_segment():
    agent = make_agent()
    ctl = RestartController(agent, strategy="library")
    dirty(agent, 5.0)
    ctl.restart(segment_reward=1.0)
    assert len(ctl.library) == 1


def test_library_first_restart_starts_fresh():
    agent = make_agent()
    ctl = RestartController(agent, strategy="library")
    dirty(agent, 5.0)
    ctl.restart(segment_reward=1.0)
    assert np.allclose(agent.q, agent.optimistic_init)


def test_library_reuses_the_best_scoring_archive():
    agent = make_agent()
    ctl = RestartController(agent, strategy="library")

    agent.q[0, 0] = 1.0
    ctl.restart(segment_reward=0.2)      # archive A, weak
    agent.q[0, 0] = 2.0
    ctl.restart(segment_reward=0.9)      # archive B, strong -> restored next
    agent.q[0, 0] = 3.0
    ctl.restart(segment_reward=0.1)      # archive C, weak

    assert len(ctl.library) == 3
    assert agent.q[0, 0] == pytest.approx(2.0)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        RestartController(make_agent(), strategy="teleport")


def test_restart_count_is_tracked():
    ctl = RestartController(make_agent(), strategy="cold")
    ctl.restart(segment_reward=1.0)
    ctl.restart(segment_reward=1.0)
    assert ctl.n_restarts == 2
