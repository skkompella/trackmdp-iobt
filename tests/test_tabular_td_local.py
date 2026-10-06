"""Tests for TabularTDAgent's "local" action mode (subsets of reachable cells)."""
from math import comb

import numpy as np
import pytest

from src.core.tabular_td import TabularTDAgent, reachable_cells

TL = 1
N = 25
MISS = N * (TL + 1) + 1


def make(k=4, **kw):
    return TabularTDAgent(time_limit=TL, max_sensors=k, missing_state=MISS,
                          n_nodes=N, grid_width=N, action_mode="local",
                          grid_shape=(5, 5), seed=0, **kw)


def test_reachable_cells_interior_and_corner():
    assert sorted(reachable_cells(12, 1, 5, 5)) == [7, 11, 12, 13, 17]
    assert len(reachable_cells(12, 2, 5, 5)) == 13
    assert sorted(reachable_cells(0, 1, 5, 5)) == [0, 1, 5]


def test_action_counts_per_state():
    ag = make(k=4)
    s0 = ag.obs_to_state((12, 0, None))
    s1 = ag.obs_to_state((12, 1, None))
    assert ag._n_valid[s0] == sum(comb(5, i) for i in range(5))      # 31
    assert ag._n_valid[s1] == sum(comb(13, i) for i in range(5))     # 1093
    assert ag._n_valid[ag.obs_to_state((N, 0, None))] == 1           # missing


def test_actions_stay_in_reachable_set_and_budget():
    ag = make(k=4, epsilon=1.0)
    for cell in range(N):
        for d in (0, 1):
            s = ag.obs_to_state((cell, d, None))
            ok = set(reachable_cells(cell, d + 1, 5, 5))
            for _ in range(20):
                v = ag.action_vector(ag.select_action(s, explore=True))
                assert set(np.flatnonzero(v)) <= ok
                assert v.sum() <= 4


def test_padding_is_never_selected_or_updated():
    ag = make(k=4)
    s = ag.obs_to_state((0, 0, None))          # corner: only 3 cells, 8 subsets
    n = ag._n_valid[s]
    assert np.all(np.isneginf(ag.q[s, n:]))
    for _ in range(50):
        a = ag.select_action(s, explore=True)
        ag.observe(s, a, 1.0, s, a)
    assert np.all(np.isneginf(ag.q[s, n:]))
    ag.reset()
    assert np.all(np.isneginf(ag.q[s, n:]))


def test_learns_to_watch_the_rewarded_cell():
    """Rewarding only the subset containing cell 13 makes it the greedy choice."""
    ag = make(k=1, alpha=0.5, lam=0.0, epsilon=1.0)
    s = ag.obs_to_state((12, 0, None))
    for _ in range(400):
        a = ag.select_action(s, explore=True)
        r = 1.0 if ag.action_vector(a)[13] == 1 else 0.0
        ag.observe(s, a, r, s, a, done=True)
    assert ag.action_vector(ag.select_action(s, explore=False))[13] == 1


def test_snapshot_restore_round_trip():
    ag = make(k=3)
    snap = ag.snapshot()
    ag.q[ag.q > -np.inf] = 7.0
    ag.restore(snap)
    np.testing.assert_array_equal(ag.q, snap)


def test_requires_matching_grid_shape():
    with pytest.raises(ValueError):
        TabularTDAgent(time_limit=TL, max_sensors=3, missing_state=MISS,
                       n_nodes=N, grid_width=N, action_mode="local")
