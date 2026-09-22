"""
Tests for the FACTORED action mode of TabularTDAgent.

Factored mode exists because max_sensors=6 over 25 cells is 245,505 subsets —
a 12.5M-entry table that would never converge. Scoring cells independently
turns that into 51 x 25 = 1,275 weights.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.tabular_td import TabularTDAgent, state_index  # noqa: E402

CELLS = 25          # 5x5 grid
TL = 1
MISSING = CELLS * (TL + 1) + 1


def make(max_sensors=6, **kw):
    kw.setdefault("seed", 0)
    return TabularTDAgent(time_limit=TL, max_sensors=max_sensors,
                          missing_state=MISSING, n_nodes=CELLS,
                          grid_width=CELLS, action_mode="factored", **kw)


# ---------------------------------------------------------------------------
# Shape and configuration
# ---------------------------------------------------------------------------

def test_weight_table_is_one_column_per_cell():
    agent = make()
    assert agent.q.shape == (CELLS * (TL + 1) + 1, CELLS + 1)   # +1 bias
    assert agent.q.size == 1326


def test_factored_is_vastly_smaller_than_subset_would_be():
    from math import comb
    subset = sum(comb(CELLS, k) for k in range(1, 7))
    assert subset > 240_000
    assert make().q.size < subset / 100


def test_grid_width_must_cover_the_cells():
    with pytest.raises(ValueError, match="grid_width"):
        TabularTDAgent(time_limit=TL, max_sensors=2, missing_state=MISSING,
                       n_nodes=CELLS, grid_width=16, action_mode="factored")


def test_unknown_action_mode_rejected():
    with pytest.raises(ValueError):
        TabularTDAgent(time_limit=TL, max_sensors=2, missing_state=MISSING,
                       action_mode="quantum")


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

def test_greedy_picks_the_highest_weighted_cells():
    agent = make(max_sensors=3)
    s = state_index(4, 0, TL, CELLS)
    agent.q[s] = -1.0
    agent.q[s, [7, 11, 19]] = [5.0, 4.0, 3.0]
    assert set(agent.select_action(s, explore=False)) == {7, 11, 19}


def test_greedy_respects_the_budget():
    agent = make(max_sensors=2)
    s = state_index(4, 0, TL, CELLS)
    agent.q[s] = 9.0                     # every cell looks great
    assert len(agent.select_action(s, explore=False)) == 2


def test_greedy_declines_to_activate_when_all_weights_negative():
    """With a per-sensor cost, activating nothing is a legitimate action."""
    agent = make()
    s = state_index(4, 0, TL, CELLS)
    agent.q[s] = -0.5
    assert agent.select_action(s, explore=False) == ()


def test_greedy_activates_only_the_positive_cells():
    agent = make(max_sensors=4)
    s = state_index(4, 0, TL, CELLS)
    agent.q[s] = -1.0
    agent.q[s, [2, 5]] = [3.0, 1.0]
    assert set(agent.select_action(s, explore=False)) == {2, 5}


def test_optimistic_init_makes_the_agent_activate_initially():
    agent = make(max_sensors=3, optimistic_init=1.0)
    s = state_index(4, 0, TL, CELLS)
    assert len(agent.select_action(s, explore=False)) == 3


# ---------------------------------------------------------------------------
# Action vectors and values
# ---------------------------------------------------------------------------

def test_action_vector_marks_the_chosen_cells():
    agent = make()
    vec = agent.action_vector((1, 4, 9))
    assert vec.shape == (CELLS,)
    assert set(np.flatnonzero(vec)) == {1, 4, 9}


def test_empty_action_vector_is_all_zero():
    assert make().action_vector(()).sum() == 0


def test_action_value_sums_the_weights():
    agent = make()
    s = state_index(3, 0, TL, CELLS)
    agent.q[s, [1, 2]] = [0.5, 1.5]
    assert agent.action_value(s, (1, 2)) == pytest.approx(2.0)


def test_empty_action_has_zero_value():
    assert make().action_value(0, ()) == 0.0


def test_compute_single_action_returns_a_grid_vector():
    agent = make(max_sensors=3, optimistic_init=1.0)
    a = np.asarray(agent.compute_single_action((4, 0, None), explore=False))
    assert a.shape == (CELLS,)
    assert set(np.unique(a)).issubset({0, 1})
    assert a.sum() <= 3


def test_missing_state_observation_maps_to_the_missing_row():
    agent = make()
    assert agent.obs_to_state((CELLS, 0, None)) == state_index(None, 0, TL, CELLS)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def test_update_moves_every_activated_cell():
    agent = make(alpha=0.5, lam=0.0)
    s, a = state_index(1, 0, TL, CELLS), (3, 8)
    before = agent.q[s, list(a)].copy()
    agent.observe(s, a, reward=1.0, next_state=s, next_action=a, done=False)
    assert (agent.q[s, list(a)] > before).all()


def test_update_leaves_unactivated_cells_alone():
    agent = make(alpha=0.5, lam=0.0)
    s, a = state_index(1, 0, TL, CELLS), (3,)
    before = agent.q[s, 8]
    agent.observe(s, a, 1.0, s, a, done=False)
    assert agent.q[s, 8] == pytest.approx(before)


def test_negative_reward_moves_weights_down():
    agent = make(alpha=0.5, lam=0.0)
    s, a = state_index(1, 0, TL, CELLS), (3, 8)
    before = agent.q[s, list(a)].copy()
    agent.observe(s, a, -1.0, s, a, done=False)
    assert (agent.q[s, list(a)] < before).all()


def test_traces_decay_by_gamma_lambda():
    agent = make(alpha=0.1, gamma=0.9, lam=0.5)
    s, a = state_index(1, 0, TL, CELLS), (2,)
    agent.observe(s, a, 0.0, s, a, done=False)
    assert agent.traces[s, 2] == pytest.approx(0.45)


def test_eligibility_propagates_credit_backwards():
    agent = make(alpha=0.5, gamma=1.0, lam=1.0)
    s1, s2, a = (state_index(1, 0, TL, CELLS),
                 state_index(2, 0, TL, CELLS), (4,))
    agent.observe(s1, a, 0.0, s2, a, done=False)
    before = agent.q[s1, 4]
    agent.observe(s2, a, 1.0, s2, a, done=False)
    assert agent.q[s1, 4] > before


def test_empty_action_still_learns_from_the_next_state():
    """An episode of doing nothing must not crash the update."""
    agent = make(alpha=0.5, lam=0.0)
    s = state_index(1, 0, TL, CELLS)
    agent.observe(s, (), reward=-0.5, next_state=s, next_action=(), done=False)
    assert np.isfinite(agent.q).all()


def test_learning_separates_good_cells_from_bad():
    """Reward only cell 7 and it must end up on top for that state."""
    agent = make(max_sensors=2, alpha=0.2, lam=0.0, optimistic_init=0.0)
    s = state_index(5, 0, TL, CELLS)
    for _ in range(200):
        agent.observe(s, (7,), reward=1.0, next_state=s, next_action=(7,))
        agent.observe(s, (12,), reward=-1.0, next_state=s, next_action=(12,))
    assert agent.q[s, 7] > agent.q[s, 12]
    assert 7 in agent.select_action(s, explore=False)


# ---------------------------------------------------------------------------
# Restart support
# ---------------------------------------------------------------------------

def test_reset_restores_optimistic_init():
    agent = make(optimistic_init=1.0, alpha=0.5)
    s = state_index(1, 0, TL, CELLS)
    agent.observe(s, (3,), 5.0, s, (3,))
    assert not np.allclose(agent.q, 1.0)
    agent.reset()
    assert np.allclose(agent.q, 1.0)
    assert not agent.traces.any()


def test_snapshot_restore_roundtrip():
    agent = make(alpha=0.5)
    s = state_index(1, 0, TL, CELLS)
    agent.observe(s, (3,), 5.0, s, (3,))
    snap = agent.snapshot()
    agent.reset()
    agent.restore(snap)
    assert np.allclose(agent.q, snap)


def test_restore_rejects_a_subset_mode_table():
    """A table from the enumerated mode must not silently load here."""
    with pytest.raises(ValueError):
        make().restore(np.zeros((51, 245505)))


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def test_epsilon_one_explores_and_stays_in_budget():
    agent = make(max_sensors=3, epsilon=1.0)
    s = state_index(4, 0, TL, CELLS)
    picks = set()
    for _ in range(200):
        a = agent.select_action(s, explore=True)
        assert 1 <= len(a) <= 3
        assert len(set(a)) == len(a)          # no duplicate cells
        assert all(0 <= c < CELLS for c in a)
        picks.add(a)
    assert len(picks) > 5


def test_epsilon_zero_is_deterministic():
    agent = make(max_sensors=2, epsilon=0.0, optimistic_init=1.0)
    s = state_index(4, 0, TL, CELLS)
    agent.q[s, [6, 13]] = [4.0, 3.0]
    assert all(agent.select_action(s, explore=True) == (6, 13)
               for _ in range(30))
