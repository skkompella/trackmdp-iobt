"""
Tests for src/core/grid_transition_matrices.py.

The whole switching experiment rests on matrices A and B genuinely conflicting.
If they don't, a "change" is not worth detecting and the experiment is void, so
the opposed-drift checks here are load-bearing, not decoration.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.grid_transition_matrices import (  # noqa: E402
    build_drift_matrix,
    cell_to_xy,
    expected_displacement,
    make_matrix_A,
    make_matrix_B,
    make_matrix_C,
    neighbours,
    validate_transition_matrix,
    xy_to_cell,
)

ROWS = COLS = 5
N = ROWS * COLS


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def test_cell_xy_roundtrip():
    for cell in range(N):
        x, y = cell_to_xy(cell, COLS)
        assert xy_to_cell(x, y, COLS) == cell


def test_xy_convention_matches_the_env():
    """cell = y*ncols + x, with y increasing 'north' (environment.py's b_u)."""
    assert cell_to_xy(0, COLS) == (0, 0)
    assert cell_to_xy(1, COLS) == (1, 0)      # +1 east
    assert cell_to_xy(COLS, COLS) == (0, 1)   # +1 north


def test_neighbours_interior_has_four():
    assert len(neighbours(xy_to_cell(2, 2, COLS), ROWS, COLS)) == 4


def test_neighbours_corner_has_two():
    assert len(neighbours(xy_to_cell(0, 0, COLS), ROWS, COLS)) == 2
    assert len(neighbours(xy_to_cell(4, 4, COLS), ROWS, COLS)) == 2


# ---------------------------------------------------------------------------
# Matrix validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker", [make_matrix_A, make_matrix_B, make_matrix_C])
def test_shape_and_row_stochastic(maker):
    T = maker(ROWS, COLS)
    assert T.shape == (N, N)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert (T >= 0).all()


@pytest.mark.parametrize("maker", [make_matrix_A, make_matrix_B, make_matrix_C])
def test_mass_stays_in_the_neighbourhood(maker):
    """No teleporting — unlike make_random_transition_matrix's Dirichlet rows."""
    T = maker(ROWS, COLS)
    for cell in range(N):
        allowed = set(neighbours(cell, ROWS, COLS)) | {cell}
        for dest in range(N):
            if dest not in allowed:
                assert T[cell, dest] == 0.0, f"{cell}->{dest} leaks"


@pytest.mark.parametrize("maker", [make_matrix_A, make_matrix_B, make_matrix_C])
def test_validate_accepts(maker):
    ok, msg = validate_transition_matrix(maker(ROWS, COLS), ROWS, COLS)
    assert ok, msg


def test_validate_rejects_non_stochastic():
    T = make_matrix_C(ROWS, COLS)
    T[0] *= 2.0
    ok, _ = validate_transition_matrix(T, ROWS, COLS)
    assert not ok


def test_validate_rejects_off_grid_mass():
    T = make_matrix_C(ROWS, COLS)
    T[0] = 0.0
    T[0, N - 1] = 1.0      # corner-to-corner teleport
    ok, msg = validate_transition_matrix(T, ROWS, COLS)
    assert not ok
    assert "neighbourhood" in msg


# ---------------------------------------------------------------------------
# The load-bearing property: A and B must conflict
# ---------------------------------------------------------------------------

def test_A_drifts_north_east():
    dx, dy = expected_displacement(make_matrix_A(ROWS, COLS), ROWS, COLS)
    assert dx > 0.1, dx
    assert dy > 0.1, dy


def test_B_drifts_south_west():
    dx, dy = expected_displacement(make_matrix_B(ROWS, COLS), ROWS, COLS)
    assert dx < -0.1, dx
    assert dy < -0.1, dy


def test_A_and_B_are_opposed():
    ax, ay = expected_displacement(make_matrix_A(ROWS, COLS), ROWS, COLS)
    bx, by = expected_displacement(make_matrix_B(ROWS, COLS), ROWS, COLS)
    assert ax == pytest.approx(-bx, abs=1e-9)
    assert ay == pytest.approx(-by, abs=1e-9)


def test_C_has_no_net_drift():
    dx, dy = expected_displacement(make_matrix_C(ROWS, COLS), ROWS, COLS)
    assert abs(dx) < 1e-9
    assert abs(dy) < 1e-9


def test_A_and_B_differ_substantially_row_by_row():
    """Per-row total variation distance — the thing a detector must notice."""
    A, B = make_matrix_A(ROWS, COLS), make_matrix_B(ROWS, COLS)
    tv = 0.5 * np.abs(A - B).sum(axis=1)
    assert tv.min() > 0.3, f"weakest row differs by only {tv.min():.3f}"


def test_C_sits_between_A_and_B():
    """The neutral prior should not be closer to one regime than the other."""
    A, B, C = (make_matrix_A(ROWS, COLS), make_matrix_B(ROWS, COLS),
               make_matrix_C(ROWS, COLS))
    assert np.abs(A - C).sum() == pytest.approx(np.abs(B - C).sum(), rel=1e-9)


# ---------------------------------------------------------------------------
# Equal-weight matrix C
# ---------------------------------------------------------------------------

def test_C_is_uniform_over_stay_plus_neighbours():
    T = make_matrix_C(ROWS, COLS)
    for cell in range(N):
        opts = list(neighbours(cell, ROWS, COLS)) + [cell]
        p = 1.0 / len(opts)
        for o in opts:
            assert T[cell, o] == pytest.approx(p)


def test_C_corner_and_interior_differ_in_spread():
    """A corner has fewer options, so each carries more mass."""
    T = make_matrix_C(ROWS, COLS)
    corner, interior = xy_to_cell(0, 0, COLS), xy_to_cell(2, 2, COLS)
    assert T[corner, corner] > T[interior, interior]


# ---------------------------------------------------------------------------
# Boundaries and generality
# ---------------------------------------------------------------------------

def test_boundary_rows_renormalise():
    """Drift pointing off-grid must be redistributed, not lost."""
    T = make_matrix_A(ROWS, COLS)
    top_right = xy_to_cell(COLS - 1, ROWS - 1, COLS)
    assert T[top_right].sum() == pytest.approx(1.0)


def test_determinism():
    assert np.array_equal(make_matrix_A(ROWS, COLS), make_matrix_A(ROWS, COLS))


def test_works_on_a_non_square_grid():
    T = make_matrix_A(3, 4)
    assert T.shape == (12, 12)
    assert np.allclose(T.sum(axis=1), 1.0)


def test_build_drift_matrix_custom_weights():
    """Pure east drift: every row's mass sits on the eastern neighbour."""
    T = build_drift_matrix(ROWS, COLS,
                           {"E": 1.0, "W": 0.0, "N": 0.0, "S": 0.0, "stay": 0.0})
    dx, dy = expected_displacement(T, ROWS, COLS)
    assert dx > 0.7
    assert abs(dy) < 1e-9


def test_build_drift_matrix_rejects_all_zero_weights():
    with pytest.raises(ValueError):
        build_drift_matrix(ROWS, COLS,
                           {"E": 0.0, "W": 0.0, "N": 0.0, "S": 0.0, "stay": 0.0})
