"""
Tests for src/core/detector_eval.py — step-resolution detector scoring.

Why this replaces detector_scores for calibration: that function matches at
EPISODE resolution with a tolerance chosen as episodes_per_segment // 5, which
in run 821 was 204 episodes. At that tolerance "precision 0.07, recall 1.0"
says almost nothing. Calibration needs detection delay in environment steps.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.detector_eval import (  # noqa: E402
    detection_delays,
    evaluate_config,
    operating_curve,
    replay,
)


def dense(values):
    return {"steps": list(range(len(values))), "values": list(values)}


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def test_replay_returns_env_steps_not_sample_indices():
    """A sparse stream's 5th sample may be env step 400. The distinction is the
    whole reason streams carry step indices."""
    stream = {"steps": [0, 100, 200, 300, 400, 500], "values": [1, 1, 1, 0, 0, 0]}
    fires = replay(stream, delta=0.5, min_samples=2)
    assert all(f in stream["steps"] for f in fires)


def test_replay_fires_on_a_clear_shift():
    rng = np.random.default_rng(0)
    s = dense(np.concatenate([rng.random(300) < 0.9, rng.random(300) < 0.1]))
    fires = replay(s, delta=0.01, min_samples=30)
    assert fires and fires[0] >= 300


def test_replay_silent_on_a_stationary_stream():
    rng = np.random.default_rng(1)
    s = dense((rng.random(3000) < 0.7).astype(float))
    assert replay(s, delta=1e-9, min_samples=30) == []


def test_replay_more_alarms_at_larger_delta():
    rng = np.random.default_rng(2)
    s = dense((rng.random(3000) < 0.7).astype(float))
    assert len(replay(s, delta=0.4, min_samples=30)) >= \
           len(replay(s, delta=1e-6, min_samples=30))


def test_replay_handles_an_empty_stream():
    assert replay({"steps": [], "values": []}, delta=0.01) == []


# ---------------------------------------------------------------------------
# detection_delays
# ---------------------------------------------------------------------------

def test_delay_is_measured_from_the_switch():
    d = detection_delays([120], switch_steps=[100], horizon=1000)
    assert d == [20]


def test_a_switch_with_no_following_alarm_is_missed():
    assert detection_delays([50], switch_steps=[100], horizon=1000) == [None]


def test_alarms_before_a_switch_do_not_count_for_it():
    """Firing early is a false alarm, not clairvoyance."""
    assert detection_delays([90], switch_steps=[100], horizon=1000) == [None]


def test_each_switch_takes_the_first_alarm_after_it():
    d = detection_delays([150, 250], switch_steps=[100, 200], horizon=1000)
    assert d == [50, 50]


def test_an_alarm_is_not_reused_across_switches():
    """One alarm cannot be credited to two different switches."""
    d = detection_delays([150], switch_steps=[100, 200], horizon=1000)
    assert d == [50, None]


def test_delay_is_capped_at_the_next_switch():
    """An alarm arriving after the NEXT switch is too late to count."""
    d = detection_delays([260], switch_steps=[100, 200], horizon=1000)
    assert d[0] is None


# ---------------------------------------------------------------------------
# evaluate_config
# ---------------------------------------------------------------------------

def make_pair(seed=0, shift=True):
    rng = np.random.default_rng(seed)
    stationary = dense((rng.random(6000) < 0.7).astype(float))
    if shift:
        seg = [rng.random(1000) < 0.9, rng.random(1000) < 0.3,
               rng.random(1000) < 0.9]
    else:
        seg = [rng.random(1000) < 0.7 for _ in range(3)]
    switching = dense(np.concatenate(seg))
    return switching, stationary


def test_evaluate_config_reports_arl0_and_delay():
    sw, st = make_pair()
    r = evaluate_config(sw, st, switch_steps=[1000, 2000],
                        delta=0.01, min_samples=30)
    assert "arl0" in r and "mean_delay" in r and "missed" in r
    assert r["n_switches"] == 2


def test_evaluate_config_detects_a_planted_shift():
    sw, st = make_pair()
    r = evaluate_config(sw, st, switch_steps=[1000, 2000],
                        delta=0.05, min_samples=30)
    assert r["missed"] < 2, r


def test_evaluate_config_finds_nothing_when_there_is_nothing():
    sw, st = make_pair(shift=False)
    r = evaluate_config(sw, st, switch_steps=[1000, 2000],
                        delta=1e-9, min_samples=30)
    assert r["missed"] == 2
    assert r["arl0"] == float("inf")


# ---------------------------------------------------------------------------
# operating_curve — the deliverable
# ---------------------------------------------------------------------------

def test_operating_curve_spans_the_delta_axis():
    sw, st = make_pair()
    curve = operating_curve(sw, st, switch_steps=[1000, 2000],
                            deltas=np.logspace(-8, -0.5, 6), min_samples=30)
    assert len(curve) == 6
    assert all("arl0" in row and "delta" in row for row in curve)


def test_arl0_decreases_as_delta_grows_along_the_curve():
    sw, st = make_pair()
    curve = operating_curve(sw, st, switch_steps=[1000, 2000],
                            deltas=np.logspace(-8, -0.5, 6), min_samples=30)
    arls = [r["arl0"] for r in curve]
    finite = [a for a in arls if np.isfinite(a)]
    assert finite == sorted(finite, reverse=True) or len(finite) < 2
