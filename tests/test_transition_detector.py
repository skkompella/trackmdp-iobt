"""
Tests for src/core/transition_detector.py — one Bernoulli GLR stream per
(cell, next cell), fed only from tau = 0 observations.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.transition_detector import (  # noqa: E402
    TransitionStreamDetector,
    row_supports,
)


def _draw(rng, support, probs):
    return int(support[rng.choice(len(support), p=probs)])


def test_supports_grid_3x3():
    s = row_supports("grid", 3, 3)
    assert s[4] == (1, 3, 4, 5, 7)
    assert s[0] == (0, 1, 3)


def test_supports_iobt():
    s = row_supports("iobt")
    assert s[5] == (3, 4, 5)
    assert len(s[4]) == 6


def test_supports_rejects_unknown_kind():
    with pytest.raises(ValueError):
        row_supports("torus", 3, 3)


def test_stream_count_and_delta():
    d = TransitionStreamDetector(row_supports("grid", 3, 3), delta_F=0.01)
    assert d.n_streams == 33
    assert d.delta_stream == pytest.approx(0.01 / 33)


def test_observe_hit_and_miss():
    d = TransitionStreamDetector(row_supports("grid", 3, 3))
    d.observe(4, watched=[1, 4, 5], found=5)
    assert d.counts(4) == {1: 1, 3: 0, 4: 1, 5: 1, 7: 0}


def test_off_support_ignored():
    d = TransitionStreamDetector(row_supports("grid", 3, 3))
    d.observe(4, watched=[4, 8], found=None)        # 8 is not in S(4)
    assert d.counts(4)[4] == 1 and 8 not in d.counts(4)
    d.observe(4, watched=[4, 8], found=8)           # impossible move: drop it
    assert d.counts(4)[4] == 1
    assert d.n_off_support == 1


def test_silent_on_iid_rows_with_adaptive_watching():
    """Past-dependent choice of which cells to watch must not create alarms."""
    supports = row_supports("grid", 3, 3)
    rng = np.random.default_rng(0)
    rows = {i: rng.dirichlet(np.ones(len(s))) for i, s in supports.items()}
    freq = {i: {j: 0 for j in s} for i, s in supports.items()}
    d = TransitionStreamDetector(supports, delta_F=0.01)
    for _ in range(60_000):
        i = int(rng.integers(9))
        s = supports[i]
        top = sorted(s, key=lambda j: (-freq[i][j], j))[:2]
        watched = set(top) | {int(rng.choice(s))}
        nxt = _draw(rng, s, rows[i])
        freq[i][nxt] += 1
        d.observe(i, watched, nxt if nxt in watched else None)
    assert not d.alarmed


def _feed_row_change(d, rng, n_before=5_000, n_after=1_500):
    s = (1, 3, 4, 5, 7)
    before = [.1, .1, .6, .1, .1]
    after = [.1, .1, .1, .6, .1]
    for _ in range(n_before):
        d.observe(4, s, _draw(rng, s, before))
    assert not d.alarmed
    for k in range(n_after):
        if d.observe(4, s, _draw(rng, s, after)):
            return k
    return None


def test_fires_after_row_change():
    d = TransitionStreamDetector(row_supports("grid", 3, 3))
    k = _feed_row_change(d, np.random.default_rng(1))
    assert k is not None
    assert d.last_alarm["cell"] == 4


def test_latches_after_alarm_until_reset():
    d = TransitionStreamDetector(row_supports("grid", 3, 3))
    assert _feed_row_change(d, np.random.default_rng(1)) is not None
    frozen = d.counts(4)
    assert d.observe(4, (1, 3, 4, 5, 7), 4) is False
    assert d.counts(4) == frozen
    d.reset()
    assert not d.alarmed and d.last_alarm is None
    assert d.counts(4) == {1: 0, 3: 0, 4: 0, 5: 0, 7: 0}


def test_last_alarm_fields():
    d = TransitionStreamDetector(row_supports("grid", 3, 3))
    _feed_row_change(d, np.random.default_rng(1))
    assert set(d.last_alarm) == {"cell", "next_cell", "statistic",
                                 "threshold", "n"}
    assert d.last_alarm["statistic"] > d.last_alarm["threshold"]
