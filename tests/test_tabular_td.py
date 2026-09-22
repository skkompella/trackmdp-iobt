"""
Tests for src/core/tabular_td.py — tabular SARSA(lambda) over the 10-node IoBT graph.

Ray-free by construction: the agent is plain numpy, which is the whole point of
using a table here (a few thousand floats instead of a 65,536-way joint action
space).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.tabular_td import (  # noqa: E402
    N_GRID,
    TabularTDAgent,
    action_index_size,
    state_index,
)

TL = 1
MISSING = N_GRID * (TL + 1) + 1   # 33, the MultiLoopIoBTEnv sentinel


def make_agent(**kw):
    kw.setdefault("time_limit", TL)
    kw.setdefault("max_sensors", 2)
    kw.setdefault("missing_state", MISSING)
    kw.setdefault("seed", 0)
    return TabularTDAgent(**kw)


# ---------------------------------------------------------------------------
# Action enumeration
# ---------------------------------------------------------------------------

def test_action_counts_match_combinatorics():
    # sum of C(10,k) for k=1..max_sensors, matching build_action_map's ordering
    assert action_index_size(1) == 10
    assert action_index_size(2) == 55
    assert action_index_size(3) == 175
    assert action_index_size(6) == 847


def test_every_action_respects_the_budget():
    agent = make_agent(max_sensors=3)
    for idx in range(agent.n_actions):
        vec = agent.action_vector(idx)
        assert vec.shape == (N_GRID,)
        assert 1 <= int(vec.sum()) <= 3


def test_actions_only_activate_real_nodes():
    """The backing grid has 16 cells but only 10 are real nodes."""
    agent = make_agent(max_sensors=3)
    for idx in range(agent.n_actions):
        assert not agent.action_vector(idx)[10:].any()


def test_action_index_is_a_bijection():
    agent = make_agent(max_sensors=2)
    seen = set()
    for idx in range(agent.n_actions):
        key = tuple(np.flatnonzero(agent.action_vector(idx)))
        assert key not in seen
        seen.add(key)
    assert len(seen) == agent.n_actions


# ---------------------------------------------------------------------------
# State indexing
# ---------------------------------------------------------------------------

def test_state_index_packs_node_and_delay():
    assert state_index(0, 0, TL) == 0
    assert state_index(0, 1, TL) == 1
    assert state_index(1, 0, TL) == 2
    assert state_index(9, TL, TL) == 9 * (TL + 1) + TL


def test_missing_state_gets_its_own_row():
    missing_row = state_index(None, 0, TL)
    assert missing_row == 10 * (TL + 1)
    # and it must not collide with any real (node, delay) pair
    reals = {state_index(n, d, TL) for n in range(10) for d in range(TL + 1)}
    assert missing_row not in reals


def test_q_table_shape_is_compact():
    agent = make_agent(max_sensors=2)
    # 21 rows at tl=1 (10 nodes x 2 delays + missing), 55 actions
    assert agent.q.shape == (21, 55)


def test_q_table_shape_scales_with_time_limit():
    agent = make_agent(time_limit=3, max_sensors=2,
                       missing_state=N_GRID * 4 + 1)
    assert agent.q.shape == (10 * 4 + 1, 55)


# ---------------------------------------------------------------------------
# Observation handling / policy interface
# ---------------------------------------------------------------------------

def obs_for(node, delay):
    """Mimic what evaluate_policy builds: (state_pos, state_time, history)."""
    return (node, delay, np.ones(64, dtype=np.int64))


def test_compute_single_action_returns_a_grid_vector():
    agent = make_agent()
    action = agent.compute_single_action(obs_for(3, 0), explore=False)
    action = np.asarray(action)
    assert action.shape == (N_GRID,)
    assert set(np.unique(action)).issubset({0, 1})
    assert 1 <= int(action.sum()) <= 2


def test_greedy_is_deterministic_when_not_exploring():
    agent = make_agent()
    agent.q[state_index(3, 0, TL), 7] = 5.0
    a = agent.compute_single_action(obs_for(3, 0), explore=False)
    b = agent.compute_single_action(obs_for(3, 0), explore=False)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(a), agent.action_vector(7))


def test_epsilon_zero_never_explores():
    agent = make_agent(epsilon=0.0)
    agent.q[state_index(2, 0, TL), 3] = 9.0
    for _ in range(50):
        idx = agent.select_action(state_index(2, 0, TL), explore=True)
        assert idx == 3


def test_epsilon_one_always_explores():
    agent = make_agent(epsilon=1.0)
    s = state_index(2, 0, TL)
    agent.q[s, 3] = 9.0
    picks = {agent.select_action(s, explore=True) for _ in range(200)}
    assert len(picks) > 1


def test_missing_state_observation_maps_to_missing_row():
    """evaluate_policy signals the missing state with state_pos == n_cells."""
    agent = make_agent()
    assert agent.obs_to_state(obs_for(N_GRID, 0)) == state_index(None, 0, TL)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def test_update_moves_the_visited_entry_toward_the_target():
    agent = make_agent(alpha=0.5, gamma=1.0, lam=0.0)
    s, a = state_index(1, 0, TL), 4
    before = agent.q[s, a]
    agent.observe(s, a, reward=1.0, next_state=s, next_action=a, done=False)
    assert agent.q[s, a] > before


def test_negative_reward_moves_it_down():
    agent = make_agent(alpha=0.5, gamma=1.0, lam=0.0)
    s, a = state_index(1, 0, TL), 4
    before = agent.q[s, a]
    agent.observe(s, a, reward=-1.0, next_state=s, next_action=a, done=False)
    assert agent.q[s, a] < before


def test_traces_decay_by_gamma_lambda():
    agent = make_agent(alpha=0.1, gamma=0.9, lam=0.5)
    s, a = state_index(1, 0, TL), 4
    agent.observe(s, a, 0.0, s, a, done=False)
    assert agent.traces[s, a] == pytest.approx(0.45)   # 1.0 * gamma * lam
    s2 = state_index(2, 0, TL)
    agent.observe(s2, a, 0.0, s2, a, done=False)
    assert agent.traces[s, a] == pytest.approx(0.2025)  # decayed again


def test_traces_reset_at_episode_end():
    agent = make_agent(alpha=0.1, lam=0.9)
    s, a = state_index(1, 0, TL), 4
    agent.observe(s, a, 1.0, s, a, done=False)
    assert agent.traces.any()
    agent.end_episode()
    assert not agent.traces.any()


def test_eligibility_propagates_credit_backwards():
    """With lambda>0 an earlier state must also be updated by a later reward."""
    agent = make_agent(alpha=0.5, gamma=1.0, lam=1.0)
    s1, s2, a = state_index(1, 0, TL), state_index(2, 0, TL), 4
    agent.observe(s1, a, 0.0, s2, a, done=False)
    before = agent.q[s1, a]
    agent.observe(s2, a, 1.0, s2, a, done=False)
    assert agent.q[s1, a] > before


def test_lambda_zero_does_not_propagate_backwards():
    agent = make_agent(alpha=0.5, gamma=1.0, lam=0.0)
    s1, s2, a = state_index(1, 0, TL), state_index(2, 0, TL), 4
    agent.observe(s1, a, 0.0, s2, a, done=False)
    before = agent.q[s1, a]
    agent.observe(s2, a, 1.0, s2, a, done=False)
    assert agent.q[s1, a] == pytest.approx(before)


def test_optimistic_initialisation_is_applied():
    agent = make_agent(optimistic_init=2.5)
    assert np.allclose(agent.q, 2.5)


# ---------------------------------------------------------------------------
# Reset (used by change-detection restarts)
# ---------------------------------------------------------------------------

def test_reset_restores_optimistic_init_and_clears_traces():
    agent = make_agent(optimistic_init=1.0, alpha=0.5, lam=0.9)
    s, a = state_index(1, 0, TL), 4
    agent.observe(s, a, 5.0, s, a, done=False)
    assert not np.allclose(agent.q, 1.0)
    agent.reset()
    assert np.allclose(agent.q, 1.0)
    assert not agent.traces.any()


def test_snapshot_and_restore_roundtrip():
    """Warm restarts need to reload a saved table."""
    agent = make_agent(alpha=0.5)
    s, a = state_index(1, 0, TL), 4
    agent.observe(s, a, 5.0, s, a, done=False)
    snap = agent.snapshot()
    agent.reset()
    assert not np.allclose(agent.q, snap)
    agent.restore(snap)
    assert np.allclose(agent.q, snap)


def test_restore_rejects_wrong_shape():
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.restore(np.zeros((3, 3)))


def test_seeded_agents_behave_identically():
    a1, a2 = make_agent(seed=99, epsilon=0.5), make_agent(seed=99, epsilon=0.5)
    s = state_index(4, 0, TL)
    for _ in range(50):
        assert a1.select_action(s, explore=True) == a2.select_action(s, explore=True)
