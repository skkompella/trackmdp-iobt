"""
Tests for src/core/multiloop_iobt_env.py — the synthetic multi-loop IoBT
environment.

The env walks one of N loops per episode, picking the loop uniformly at reset.
Detection is ideal binary: activating the node the object is on always detects
it.  These tests pin down loop selection, movement, and the tracking state
machine (hit / miss / overflow into missing state / re-acquisition).
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.multiloop_iobt_env import (  # noqa: E402
    MultiLoopIoBTEnv,
    MultiLoopIoBTLearner,
)

TIME_LIMIT = 3
N_GRID = 16                                   # 4x4 backing grid
MISSING = N_GRID * (TIME_LIMIT + 1) + 1       # matches the learner's formula

LOOPS = [
    [0, 4, 3, 6, 2, 1],
    [7, 8, 9],
    [0, 1, 2, 3, 5, 4],
]


def make_env(loops=None, seed=0, time_limit=TIME_LIMIT, **kw):
    return MultiLoopIoBTEnv(
        max_sensors=6, max_sensors_null=6,
        missing_state=MISSING, time_limit=time_limit,
        loops=loops if loops is not None else LOOPS,
        seed=seed, **kw,
    )


def all_off():
    return np.zeros(N_GRID, dtype=int)


def only(node):
    a = np.zeros(N_GRID, dtype=int)
    a[node] = 1
    return a


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_builds_and_starts_on_a_loop():
    env = make_env()
    assert env.object_pos in env.current_loop


def test_rejects_invalid_loop():
    with pytest.raises(ValueError, match="not adjacent"):
        make_env(loops=[[0, 5, 4]])


def test_rejects_empty_loop_set():
    with pytest.raises(ValueError):
        make_env(loops=[])


def test_exposes_loops_readonly_copy():
    env = make_env()
    env.loops[0][0] = 999
    assert env.loops[0][0] != 999, "loops property must not expose internal state"


# ---------------------------------------------------------------------------
# Loop selection
# ---------------------------------------------------------------------------

def test_reset_selects_a_valid_loop_and_position():
    env = make_env()
    for _ in range(50):
        env.reset_object_state()
        assert 0 <= env.current_loop_idx < len(LOOPS)
        assert env.object_pos in LOOPS[env.current_loop_idx]


def test_loop_selection_is_seeded_and_reproducible():
    a = make_env(seed=42)
    b = make_env(seed=42)
    for _ in range(25):
        a.reset_object_state()
        b.reset_object_state()
        assert a.current_loop_idx == b.current_loop_idx
        assert a.object_pos == b.object_pos


def test_different_seeds_diverge():
    a, b = make_env(seed=1), make_env(seed=2)
    picks_a, picks_b = [], []
    for _ in range(40):
        a.reset_object_state(); picks_a.append(a.current_loop_idx)
        b.reset_object_state(); picks_b.append(b.current_loop_idx)
    assert picks_a != picks_b


def test_loop_selection_is_roughly_uniform():
    env = make_env(seed=0)
    counts = Counter()
    for _ in range(3000):
        env.reset_object_state()
        counts[env.current_loop_idx] += 1
    assert set(counts) == set(range(len(LOOPS)))
    for idx in range(len(LOOPS)):
        # expect ~1000 each; generous band, this only catches gross skew
        assert 800 < counts[idx] < 1200, counts


def test_start_position_varies_within_a_loop():
    env = make_env(seed=0)
    env.force_loop(0)
    seen = set()
    for _ in range(200):
        env.reset_object_state()
        seen.add(env.object_pos)
    assert seen == set(LOOPS[0]), "every position on the loop should be reachable"


# ---------------------------------------------------------------------------
# force_loop
# ---------------------------------------------------------------------------

def test_force_loop_pins_selection():
    env = make_env()
    env.force_loop(1)
    for _ in range(30):
        env.reset_object_state()
        assert env.current_loop_idx == 1
        assert env.object_pos in LOOPS[1]


def test_force_loop_none_restores_random_selection():
    env = make_env(seed=0)
    env.force_loop(1)
    env.force_loop(None)
    picks = set()
    for _ in range(100):
        env.reset_object_state()
        picks.add(env.current_loop_idx)
    assert len(picks) > 1


def test_force_loop_rejects_bad_index():
    env = make_env()
    with pytest.raises(IndexError):
        env.force_loop(len(LOOPS))


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def test_object_move_follows_the_loop_in_order():
    env = make_env()
    env.force_loop(0)
    env.reset_object_state()
    loop = LOOPS[0]
    start = loop.index(env.object_pos)
    for step in range(1, 2 * len(loop) + 1):
        env.object_move()
        assert env.object_pos == loop[(start + step) % len(loop)]


def test_object_move_wraps_forever():
    env = make_env()
    env.force_loop(1)              # length-3 loop
    env.reset_object_state()
    positions = [env.object_pos]
    for _ in range(30):
        env.object_move()
        positions.append(env.object_pos)
    assert set(positions) == set(LOOPS[1])


def test_move_stays_on_selected_loop_after_reset():
    env = make_env(seed=3)
    for _ in range(20):
        env.reset_object_state()
        loop = env.current_loop
        for _ in range(10):
            env.object_move()
            assert env.object_pos in loop


# ---------------------------------------------------------------------------
# Tracking state machine
# ---------------------------------------------------------------------------

def test_hit_resets_time_delay_and_rebases_state():
    env = make_env()
    env.force_loop(0)
    env.reset_object_state()
    pos = env.object_pos
    state = pos * (TIME_LIMIT + 1)

    _, next_state, terminal, delay = env.get_reward_next_state(state, only(pos), 0)

    assert delay == 0
    assert next_state == pos * (TIME_LIMIT + 1)
    assert terminal == 0


def test_miss_increments_time_delay():
    env = make_env()
    env.force_loop(0)
    env.reset_object_state()
    pos = env.object_pos
    state = pos * (TIME_LIMIT + 1)

    _, next_state, _, delay = env.get_reward_next_state(state, all_off(), 0)

    assert delay == 1
    assert next_state == state + 1


def test_repeated_misses_overflow_into_missing_state():
    env = make_env()
    env.force_loop(0)
    env.reset_object_state()
    state = env.object_pos * (TIME_LIMIT + 1)
    delay = 0
    for _ in range(TIME_LIMIT + 1):
        _, state, _, delay = env.get_reward_next_state(state, all_off(), delay)
    assert state == MISSING


def test_missing_state_always_reacquires():
    """The full rescan in missing state must always succeed, or time_delay
    would run away unbounded."""
    env = make_env()
    env.force_loop(0)
    env.reset_object_state()
    pos = env.object_pos
    # Pass an all-off action: the env must override it with a full scan.
    _, next_state, _, delay = env.get_reward_next_state(MISSING, all_off(), 0)
    assert delay == 0
    assert next_state == pos * (TIME_LIMIT + 1)


def test_terminal_flag_is_always_zero():
    """Loops never end — the object circles forever."""
    env = make_env()
    env.reset_object_state()
    state = env.object_pos * (TIME_LIMIT + 1)
    delay = 0
    for _ in range(50):
        _, state, terminal, delay = env.get_reward_next_state(state, all_off(), delay)
        assert terminal == 0


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def test_hit_beats_miss():
    env = make_env()
    env.force_loop(0)
    env.reset_object_state()
    pos = env.object_pos
    state = pos * (TIME_LIMIT + 1)

    hit, *_ = env.get_reward_next_state(state, only(pos), 0)
    env.force_loop(0); env.reset_object_state()
    pos2 = env.object_pos
    miss, *_ = env.get_reward_next_state(pos2 * (TIME_LIMIT + 1), all_off(), 0)

    assert hit > miss


def test_extra_sensors_cost_reward():
    env = make_env(sensor_rew=-0.25)
    env.force_loop(0)
    env.reset_object_state()
    pos = env.object_pos
    state = pos * (TIME_LIMIT + 1)

    lean, *_ = env.get_reward_next_state(state, only(pos), 0)

    env.force_loop(0); env.reset_object_state()
    pos2 = env.object_pos
    wide = np.ones(N_GRID, dtype=int)
    fat, *_ = env.get_reward_next_state(pos2 * (TIME_LIMIT + 1), wide, 0)

    assert lean > fat, "activating every node must cost more than a single hit"


def test_sensor_rew_is_configurable():
    free = make_env(sensor_rew=0.0)
    free.force_loop(0); free.reset_object_state()
    pos = free.object_pos
    r_free, *_ = free.get_reward_next_state(pos * (TIME_LIMIT + 1), only(pos), 0)

    costly = make_env(sensor_rew=-1.0)
    costly.force_loop(0); costly.reset_object_state()
    pos2 = costly.object_pos
    r_costly, *_ = costly.get_reward_next_state(pos2 * (TIME_LIMIT + 1), only(pos2), 0)

    assert r_free > r_costly


# ---------------------------------------------------------------------------
# Learner adapter
# ---------------------------------------------------------------------------

def test_learner_exposes_wrapper_interface():
    lrn = MultiLoopIoBTLearner(
        run_number=800, num_trans=6, max_sensors=6, max_sensors_null=6,
        time_limit=TIME_LIMIT, time_limit_max=TIME_LIMIT, loops=LOOPS, seed=0,
    )
    assert lrn.N == 4
    assert lrn.n_cells == 10
    assert lrn.missing_state == MISSING
    assert isinstance(lrn.grid_env, MultiLoopIoBTEnv)
    assert lrn.time_limit == TIME_LIMIT
    # attributes the gym wrapper reads
    for attr in ("total_actions", "total_actions_null", "current_state",
                 "time_delay", "max_sensors"):
        assert hasattr(lrn, attr)
