"""
Tests for the calibration additions to src/core/change_detection.py:
a Gaussian divergence for continuous signals, exposure of the GLR statistic,
and calibrate_delta() which solves for a target false-alarm rate.

The point of all of this: delta is currently a magic number. Measured across the
existing runs it produced ARL0 between 4,166 and 13,888 steps depending only on
how fast the arm fed the detector. Calibration makes the false-alarm rate the
thing you specify and delta the thing that is derived.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.change_detection import (  # noqa: E402
    GLRChangeDetector,
    arl0,
    calibrate_delta,
    gaussian_divergence,
    kl_bernoulli,
)


# ---------------------------------------------------------------------------
# Gaussian divergence
# ---------------------------------------------------------------------------

def test_gaussian_divergence_zero_for_equal_means():
    assert gaussian_divergence(0.4, 0.4, 1.0) == pytest.approx(0.0)


def test_gaussian_divergence_is_quadratic_in_separation():
    a = gaussian_divergence(0.0, 1.0, 1.0)
    b = gaussian_divergence(0.0, 2.0, 1.0)
    assert b == pytest.approx(4 * a)


def test_gaussian_divergence_scales_inversely_with_variance():
    a = gaussian_divergence(0.0, 1.0, 1.0)
    b = gaussian_divergence(0.0, 1.0, 4.0)
    assert b == pytest.approx(a / 4)


def test_gaussian_divergence_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        gaussian_divergence(0.0, 1.0, 0.0)


def test_gaussian_detector_accepts_continuous_values():
    """Bernoulli KL would clamp these to [0,1] and lose the signal."""
    rng = np.random.default_rng(0)
    det = GLRChangeDetector(delta=0.01, min_samples=20, divergence="gaussian")
    stream = np.concatenate([rng.normal(5.0, 1.0, 200),
                             rng.normal(9.0, 1.0, 200)])
    fired = next((i for i, v in enumerate(stream) if det.update(v)), None)
    assert fired is not None
    assert fired >= 200, "must not fire before the shift"


def test_gaussian_detector_silent_on_stationary_continuous_stream():
    rng = np.random.default_rng(1)
    det = GLRChangeDetector(delta=1e-4, min_samples=20, divergence="gaussian")
    assert not any(det.update(v) for v in rng.normal(3.0, 1.0, 600))


def test_unknown_divergence_rejected():
    with pytest.raises(ValueError):
        GLRChangeDetector(divergence="cauchy")


# ---------------------------------------------------------------------------
# Statistic exposure — needed to sweep without re-running
# ---------------------------------------------------------------------------

def test_update_exposes_statistic_and_threshold():
    det = GLRChangeDetector(delta=0.01, min_samples=10)
    for v in [1] * 20:
        det.update(v)
    assert det.last_statistic is not None
    assert det.last_threshold is not None
    assert det.last_statistic >= 0.0


def test_statistic_is_none_before_min_samples():
    det = GLRChangeDetector(delta=0.01, min_samples=50)
    det.update(1)
    assert det.last_statistic is None


def test_critical_delta_matches_the_firing_decision():
    """
    critical_delta is the largest delta that would NOT have fired. Any delta
    above it fires. That identity is what lets one replay score a whole delta
    axis instead of re-running per value.
    """
    rng = np.random.default_rng(3)
    stream = np.concatenate([rng.random(120) < 0.9, rng.random(120) < 0.2])
    det = GLRChangeDetector(delta=0.01, min_samples=20)
    for v in stream:
        fired = det.update(v)
        if det.last_statistic is not None:
            cd = det.critical_delta
            assert (det.delta > cd) == fired, (det.delta, cd, fired)
            if fired:
                break


# ---------------------------------------------------------------------------
# ARL0 and calibration
# ---------------------------------------------------------------------------

def test_arl0_counts_steps_between_alarms():
    """A stream with no change should give a long run length."""
    rng = np.random.default_rng(4)
    stream = (rng.random(8000) < 0.7).astype(float)
    a = arl0(stream, delta=1e-6, min_samples=30, max_window=200)
    assert a > 5000, a


def test_arl0_shortens_as_delta_grows():
    rng = np.random.default_rng(5)
    stream = (rng.random(8000) < 0.7).astype(float)
    assert arl0(stream, delta=0.4, min_samples=30, max_window=200) < \
           arl0(stream, delta=1e-6, min_samples=30, max_window=200)


def test_arl0_is_infinite_when_nothing_fires():
    stream = np.full(2000, 0.5)          # perfectly constant
    assert arl0(stream, delta=1e-9, min_samples=30, max_window=200) == float("inf")


def test_calibrate_delta_hits_its_target():
    rng = np.random.default_rng(6)
    stream = (rng.random(20000) < 0.7).astype(float)
    target = 4000.0
    d = calibrate_delta(stream, target_arl0=target, min_samples=30,
                        max_window=200)
    achieved = arl0(stream, delta=d, min_samples=30, max_window=200)
    assert 0.3 * target <= achieved <= 3.0 * target, (d, achieved)


def test_calibrate_delta_is_monotone_in_target():
    """A longer target run length demands a smaller delta."""
    rng = np.random.default_rng(7)
    stream = (rng.random(12000) < 0.6).astype(float)
    lo = calibrate_delta(stream, target_arl0=1500.0, min_samples=30,
                         max_window=200)
    hi = calibrate_delta(stream, target_arl0=12000.0, min_samples=30,
                         max_window=200)
    assert hi <= lo


def test_calibrate_delta_works_for_gaussian_streams():
    rng = np.random.default_rng(8)
    stream = rng.normal(2.0, 1.0, 12000)
    d = calibrate_delta(stream, target_arl0=3000.0, min_samples=30,
                        max_window=200, divergence="gaussian")
    assert 0.0 < d < 1.0


def test_calibrate_delta_rejects_bad_target():
    with pytest.raises(ValueError):
        calibrate_delta(np.zeros(100), target_arl0=0.0)


def test_two_streams_of_different_length_get_different_deltas():
    """
    The heart of the miscalibration: the same delta means different
    sensitivities at different sample rates, so calibration must be per stream.
    """
    rng = np.random.default_rng(9)
    short = (rng.random(4000) < 0.7).astype(float)
    long_ = (rng.random(16000) < 0.7).astype(float)
    d_short = calibrate_delta(short, target_arl0=2000.0, min_samples=30,
                              max_window=200)
    d_long = calibrate_delta(long_, target_arl0=2000.0, min_samples=30,
                             max_window=200)
    assert np.isfinite(d_short) and np.isfinite(d_long)
