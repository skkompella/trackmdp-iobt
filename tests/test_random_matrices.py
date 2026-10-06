"""Tests for random local transition matrices and the matrix-driven IoBT env."""
import numpy as np
import pytest

from src.core.grid_transition_matrices import validate_transition_matrix
from src.core.iobt_loops import IOBT_MAP
from src.core.matrix_iobt_env import MatrixIoBTEnv, validate_iobt_matrix
from src.core.random_transition_matrices import parse_matrix_spec

MISS = 16 * 2 + 1


def test_grid_random_matrices_are_valid_and_seeded():
    names, mats = parse_matrix_spec("R0,R1,C", "grid", 123)
    assert names == ["R0", "R1", "C"]
    for T in mats:
        ok, msg = validate_transition_matrix(T, 5, 5)
        assert ok, msg
    _, again = parse_matrix_spec("R0", "grid", 123)
    np.testing.assert_array_equal(mats[0], again[0])
    assert not np.allclose(mats[0], mats[1])


def test_iobt_random_matrices_respect_the_graph():
    _, mats = parse_matrix_spec("R0,R3,C", "iobt", 7)
    for T in mats:
        ok, msg = validate_iobt_matrix(T)
        assert ok, msg
    C = mats[2]
    assert np.isclose(C[4, 4], 1 / (len(IOBT_MAP[4]) + 1))


def test_unknown_token_rejected():
    with pytest.raises(ValueError):
        parse_matrix_spec("A", "iobt", 0)


def test_matrix_env_samples_from_the_active_matrix():
    _, mats = parse_matrix_spec("R0,R1", "iobt", 11)
    env = MatrixIoBTEnv(4, 4, MISS, 1, mats, seed=0, reward_preset="grid")
    for idx in (0, 1):
        env.force_matrix(idx)
        env.reset_object_state()
        assert env.current_matrix_idx == idx
        counts = np.zeros(10)
        for _ in range(4000):
            env.object_pos = 4
            env.object_move()
            counts[env.object_pos] += 1
        np.testing.assert_allclose(counts / counts.sum(), mats[idx][4], atol=0.03)


def test_force_loop_alias_and_rejects_bad_matrix():
    _, mats = parse_matrix_spec("R0,R1", "iobt", 11)
    env = MatrixIoBTEnv(4, 4, MISS, 1, mats, seed=0, reward_preset="grid")
    env.force_loop(1)
    env.reset_object_state()
    assert env.current_matrix_idx == 1
    bad = np.full((10, 10), 0.1)
    with pytest.raises(ValueError):
        MatrixIoBTEnv(4, 4, MISS, 1, [bad], seed=0)
