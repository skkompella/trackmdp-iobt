"""
Tests for src/core/detection_signals.py.

The load-bearing property: every signal must be constructible from what the
agent legitimately observes. The agent learns the object's cell on each
successful detection (it is encoded into next_state and is already the agent's
own Q-table row), but it learns NOTHING about the object's position on a miss.
Any signal that needs the latter would be cheating.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.detection_signals import (  # noqa: E402
    SignalRecorder,
    TransitionSurprise,
)

N_CELLS = 25


# ---------------------------------------------------------------------------
# TransitionSurprise
# ---------------------------------------------------------------------------

def test_no_surprise_until_two_consecutive_observations():
    ts = TransitionSurprise(N_CELLS)
    assert ts.observe(3) is None            # first ever
    assert ts.observe(4) is not None        # now a pair exists


def test_a_gap_breaks_the_pair():
    ts = TransitionSurprise(N_CELLS)
    ts.observe(3)
    ts.observe(None)                        # missed: position unknown
    assert ts.observe(7) is None, "a pair may not span a miss"


def test_surprise_falls_for_a_repeated_transition():
    ts = TransitionSurprise(N_CELLS)
    first = None
    for _ in range(40):
        ts.observe(3)
        s = ts.observe(4)
        if first is None:
            first = s
        ts.observe(None)
    assert s < first, "a transition seen repeatedly must stop being surprising"


def test_surprise_rises_when_the_regime_flips():
    ts = TransitionSurprise(N_CELLS)
    for _ in range(60):                     # learn 3 -> 4
        ts.observe(3); ts.observe(4); ts.observe(None)
    ts.observe(3)
    familiar = ts.observe(4)
    ts.observe(None)
    ts.observe(3)
    novel = ts.observe(19)                  # never seen from 3
    assert novel > familiar


def test_surprise_is_finite_for_an_unseen_transition():
    """The Dirichlet prior is what keeps this from being -log(0)."""
    ts = TransitionSurprise(N_CELLS)
    ts.observe(0)
    assert np.isfinite(ts.observe(24))


def test_decay_lets_the_estimate_forget():
    fast = TransitionSurprise(N_CELLS, decay=0.5)
    slow = TransitionSurprise(N_CELLS, decay=1.0)
    for ts in (fast, slow):
        for _ in range(50):
            ts.observe(3); ts.observe(4); ts.observe(None)
    for ts in (fast, slow):
        for _ in range(10):
            ts.observe(3); ts.observe(9); ts.observe(None)
    ts_f, ts_s = fast, slow
    ts_f.observe(3); s_fast = ts_f.observe(4)
    ts_s.observe(3); s_slow = ts_s.observe(4)
    # the forgetful one should find the OLD transition less familiar now
    assert s_fast > s_slow


def test_reset_clears_the_estimate():
    ts = TransitionSurprise(N_CELLS)
    for _ in range(30):
        ts.observe(3); ts.observe(4); ts.observe(None)
    ts.reset()
    ts.observe(3)
    after = ts.observe(4)
    fresh = TransitionSurprise(N_CELLS)
    fresh.observe(3)
    assert after == pytest.approx(fresh.observe(4))


def test_rejects_out_of_range_cell():
    with pytest.raises(ValueError):
        TransitionSurprise(N_CELLS).observe(99)


# ---------------------------------------------------------------------------
# SignalRecorder
# ---------------------------------------------------------------------------

def make_recorder():
    return SignalRecorder(n_cells=N_CELLS)


def test_records_every_declared_signal():
    r = make_recorder()
    r.step(hit=1, cell=3, delay=0, reward=0.8, sensors=2)
    r.step(hit=0, cell=None, delay=1, reward=-0.3, sensors=2)
    s = r.streams()
    assert set(s) >= {"hit", "staleness", "reward", "miss_run",
                      "transition_surprise"}


def test_dense_signals_have_one_value_per_step():
    r = make_recorder()
    for i in range(10):
        r.step(hit=i % 2, cell=i if i % 2 else None, delay=0, reward=0.1,
               sensors=1)
    for name in ("hit", "staleness", "reward", "miss_run"):
        assert len(r.streams()[name]["values"]) == 10, name


def test_transition_surprise_is_sparse_and_carries_step_indices():
    """
    It only updates on consecutive observations, so it has a different sample
    rate from the per-step signals. That is exactly why every stream carries
    its own step indices -- ARL0 must be measured in ENV STEPS, not samples.
    """
    r = make_recorder()
    r.step(hit=1, cell=1, delay=0, reward=1.0, sensors=1)
    r.step(hit=1, cell=2, delay=0, reward=1.0, sensors=1)
    r.step(hit=0, cell=None, delay=1, reward=0.0, sensors=1)
    r.step(hit=1, cell=8, delay=0, reward=1.0, sensors=1)
    ts = r.streams()["transition_surprise"]
    assert len(ts["values"]) == 1          # only the 1->2 pair
    assert ts["steps"] == [1]
    assert len(r.streams()["hit"]["values"]) == 4


def test_miss_run_counts_consecutive_misses():
    r = make_recorder()
    for hit in (1, 0, 0, 0, 1, 0):
        r.step(hit=hit, cell=0 if hit else None, delay=0, reward=0.0, sensors=1)
    assert r.streams()["miss_run"]["values"] == [0, 1, 2, 3, 0, 1]


def test_switch_marks_are_recorded_separately_from_signals():
    """Ground truth must never be fed to a detector - only used for scoring."""
    r = make_recorder()
    r.step(hit=1, cell=0, delay=0, reward=0.0, sensors=1)
    r.mark_switch()
    r.step(hit=1, cell=1, delay=0, reward=0.0, sensors=1)
    assert r.switch_steps == [1]
    assert "switch" not in r.streams()


def test_total_steps_tracks_the_env_clock():
    r = make_recorder()
    for _ in range(7):
        r.step(hit=0, cell=None, delay=0, reward=0.0, sensors=1)
    assert r.total_steps == 7


def test_roundtrip_through_npz(tmp_path):
    r = make_recorder()
    for i in range(20):
        r.step(hit=i % 3 == 0, cell=i % N_CELLS if i % 3 == 0 else None,
               delay=i % 2, reward=0.1 * i, sensors=2)
    r.mark_switch()
    p = tmp_path / "s.npz"
    r.save(p, meta={"env": "test"})
    loaded = SignalRecorder.load(p)
    assert loaded["meta"]["env"] == "test"
    assert loaded["switch_steps"] == r.switch_steps
    assert loaded["signals"]["hit"]["values"] == \
           r.streams()["hit"]["values"]
