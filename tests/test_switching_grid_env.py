"""
Tests for src/core/switching_grid_env.py — a 5x5 grid whose object moves under
one of several transition matrices, switched on a schedule.

Three defects in the existing TransitionMatrixGridEnv are specifically guarded
against here: it ignores no_moore_constraint, it uses the global numpy RNG (so
runs are irreproducible), and grid_env_rect hardcodes sensor_rew.
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.grid_transition_matrices import (  # noqa: E402
    make_matrix_A, make_matrix_B, make_matrix_C, neighbours, xy_to_cell,
)
from src.core.switching_grid_env import (  # noqa: E402
    SwitchingGridLearner, SwitchingTransitionGridEnv,
)

ROWS = COLS = 5
N_CELLS = ROWS * COLS
TL = 1
MISSING = N_CELLS * (TL + 1) + 1
MATRICES = [make_matrix_A(ROWS, COLS), make_matrix_B(ROWS, COLS)]


def make_env(matrices=None, seed=0, max_sensors=6, **kw):
    return SwitchingTransitionGridEnv(
        nrows=ROWS, ncols=COLS,
        max_sensors=max_sensors, max_sensors_null=max_sensors,
        missing_state=MISSING, time_limit=TL,
        matrices=matrices if matrices is not None else MATRICES,
        seed=seed, no_moore_constraint=True, **kw)


def all_off():
    return np.zeros(N_CELLS, dtype=int)


def only(cell):
    a = np.zeros(N_CELLS, dtype=int)
    a[cell] = 1
    return a


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_builds_with_expected_geometry():
    env = make_env()
    assert env.n_cells == 25
    assert env.nrows == 5 and env.ncols == 5
    assert env.no_moore_constraint is True


def test_rejects_empty_matrix_list():
    with pytest.raises(ValueError):
        make_env(matrices=[])


def test_rejects_wrong_shaped_matrix():
    with pytest.raises(ValueError):
        make_env(matrices=[np.eye(9)])


def test_rejects_non_stochastic_matrix():
    bad = make_matrix_A(ROWS, COLS).copy()
    bad[0] *= 3.0
    with pytest.raises(ValueError):
        make_env(matrices=[bad])


def test_sensor_rew_is_overridable():
    """grid_env_rect hardcodes -0.16; the switching env must expose it."""
    assert make_env().sensor_rew == pytest.approx(-0.16)
    assert make_env(sensor_rew=-0.5).sensor_rew == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Matrix selection
# ---------------------------------------------------------------------------

def test_force_matrix_pins_the_regime():
    env = make_env()
    env.force_matrix(1)
    assert env.current_matrix_idx == 1
    for _ in range(20):
        env.reset_object_state()
        assert env.current_matrix_idx == 1


def test_force_matrix_rejects_bad_index():
    env = make_env()
    with pytest.raises(IndexError):
        env.force_matrix(len(MATRICES))


def test_force_matrix_none_restores_random_choice():
    env = make_env(seed=3)
    env.force_matrix(0)
    env.force_matrix(None)
    picks = set()
    for _ in range(100):
        env.reset_object_state()
        picks.add(env.current_matrix_idx)
    assert len(picks) > 1


# ---------------------------------------------------------------------------
# Movement follows the ACTIVE matrix
# ---------------------------------------------------------------------------

def test_object_only_moves_to_reachable_cells():
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    for _ in range(300):
        before = env.object_pos
        env.object_move()
        allowed = set(neighbours(before, ROWS, COLS)) | {before}
        assert env.object_pos in allowed


def test_matrix_A_drifts_north_east_empirically():
    env = make_env(seed=1)
    env.force_matrix(0)
    env.object_pos = xy_to_cell(2, 2, COLS)
    xs, ys = [], []
    for _ in range(2000):
        env.object_pos = xy_to_cell(2, 2, COLS)
        env.object_move()
        x, y = env.object_pos % COLS, env.object_pos // COLS
        xs.append(x - 2); ys.append(y - 2)
    assert np.mean(xs) > 0.2, np.mean(xs)
    assert np.mean(ys) > 0.2, np.mean(ys)


def test_matrix_B_drifts_south_west_empirically():
    env = make_env(seed=1)
    env.force_matrix(1)
    xs, ys = [], []
    for _ in range(2000):
        env.object_pos = xy_to_cell(2, 2, COLS)
        env.object_move()
        x, y = env.object_pos % COLS, env.object_pos // COLS
        xs.append(x - 2); ys.append(y - 2)
    assert np.mean(xs) < -0.2, np.mean(xs)
    assert np.mean(ys) < -0.2, np.mean(ys)


def test_switching_matrix_changes_behaviour():
    """The whole experiment depends on this actually differing."""
    env = make_env(seed=2)
    means = []
    for idx in (0, 1):
        env.force_matrix(idx)
        xs = []
        for _ in range(1500):
            env.object_pos = xy_to_cell(2, 2, COLS)
            env.object_move()
            xs.append(env.object_pos % COLS - 2)
        means.append(np.mean(xs))
    assert means[0] > 0 > means[1]


# ---------------------------------------------------------------------------
# Reproducibility — the global-RNG defect
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_trajectories():
    def traj(seed):
        env = make_env(seed=seed)
        env.force_matrix(0)
        env.reset_object_state()
        return [(env.object_move(), env.object_pos)[1] for _ in range(50)]
    assert traj(11) == traj(11)


def test_different_seeds_diverge():
    def traj(seed):
        env = make_env(seed=seed)
        env.force_matrix(0)
        env.reset_object_state()
        return [(env.object_move(), env.object_pos)[1] for _ in range(50)]
    assert traj(1) != traj(2)


def test_global_numpy_rng_does_not_affect_the_env():
    """TransitionMatrixGridEnv used np.random.choice; this must not."""
    env = make_env(seed=5)
    env.force_matrix(0)
    env.reset_object_state()
    np.random.seed(999)
    a = [(env.object_move(), env.object_pos)[1] for _ in range(30)]

    env2 = make_env(seed=5)
    env2.force_matrix(0)
    env2.reset_object_state()
    np.random.seed(1)
    b = [(env2.object_move(), env2.object_pos)[1] for _ in range(30)]
    assert a == b


# ---------------------------------------------------------------------------
# Tracking state machine
# ---------------------------------------------------------------------------

def test_hit_resets_delay_and_rebases():
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    pos = env.object_pos
    _, nxt, term, delay = env.get_reward_next_state(pos * (TL + 1), only(pos), 0)
    assert delay == 0
    assert nxt == pos * (TL + 1)
    assert term == 0


def test_miss_increments_delay():
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    state = env.object_pos * (TL + 1)
    _, nxt, _, delay = env.get_reward_next_state(state, all_off(), 0)
    assert delay == 1
    assert nxt == state + 1


def test_overflow_into_missing_state():
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    state, delay = env.object_pos * (TL + 1), 0
    for _ in range(TL + 1):
        _, state, _, delay = env.get_reward_next_state(state, all_off(), delay)
    assert state == MISSING


def test_missing_state_always_reacquires():
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    pos = env.object_pos
    _, nxt, _, delay = env.get_reward_next_state(MISSING, all_off(), 0)
    assert delay == 0
    assert nxt == pos * (TL + 1)


def test_terminal_is_always_zero():
    """The matrices have no terminal state — the object never leaves."""
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    state, delay = env.object_pos * (TL + 1), 0
    for _ in range(200):
        _, state, term, delay = env.get_reward_next_state(state, all_off(), delay)
        assert term == 0
    assert env.object_pos != env.n_cells


def test_hit_beats_miss_on_reward():
    env = make_env()
    env.force_matrix(0)
    env.reset_object_state()
    pos = env.object_pos
    hit, *_ = env.get_reward_next_state(pos * (TL + 1), only(pos), 0)
    env.reset_object_state()
    pos2 = env.object_pos
    miss, *_ = env.get_reward_next_state(pos2 * (TL + 1), all_off(), 0)
    assert hit > miss


# ---------------------------------------------------------------------------
# Learner adapter
# ---------------------------------------------------------------------------

def test_learner_exposes_wrapper_interface():
    lrn = SwitchingGridLearner(
        run_number=820, nrows=ROWS, ncols=COLS, max_sensors=6,
        max_sensors_null=6, time_limit=TL, time_limit_max=TL,
        matrices=MATRICES, seed=0)
    assert lrn.n_cells == 25
    assert lrn.missing_state == MISSING
    assert isinstance(lrn.grid_env, SwitchingTransitionGridEnv)
    for attr in ("total_actions", "current_state", "time_delay", "max_sensors"):
        assert hasattr(lrn, attr)
