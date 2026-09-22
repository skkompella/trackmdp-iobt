"""
grid_transition_matrices.py — geometric transition matrices for a rectangular grid.

The switching experiment needs two movement regimes whose optimal tracking
policies genuinely conflict.  `make_random_transition_matrix` in
examples/finetune_deterministic.py cannot provide that: its rows are Dirichlet
draws over ALL cells, so the object teleports across the grid and two
independent draws are statistically near-identical from a tracker's point of
view — which is exactly the null case the multi-loop experiment already ran.

So these matrices are geometric: mass sits only on the 4-connected neighbours
plus the cell itself, and drift is imposed by weighting directions.

    A  north-east drift   N and E heavy
    B  south-west drift   the mirror of A
    C  equal weight       stay and every available neighbour equally likely

A and B are deliberately opposed.  C commits to nothing, which is what makes it
a sensible generic prior for warm restarts — the same role the topology random
walk played for run 241 on the IoBT graph.

Coordinates follow the env's convention (src/core/environment.py): a cell index
is ``y * ncols + x`` with x increasing east and y increasing north, matching the
``b_u``/``b_d`` (up/down) bounds used in get_valid_q_indices.
"""
from __future__ import annotations

import numpy as np

# Direction -> (dx, dy), with y increasing north.
DIRECTIONS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}

# Drift strength for the opposed regimes.  Raising 0.40 sharpens the conflict.
DRIFT_STRONG = 0.40
DRIFT_WEAK   = 0.05
DRIFT_STAY   = 0.10

A_WEIGHTS = {"N": DRIFT_STRONG, "E": DRIFT_STRONG,
             "S": DRIFT_WEAK,   "W": DRIFT_WEAK, "stay": DRIFT_STAY}
B_WEIGHTS = {"S": DRIFT_STRONG, "W": DRIFT_STRONG,
             "N": DRIFT_WEAK,   "E": DRIFT_WEAK, "stay": DRIFT_STAY}


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def cell_to_xy(cell: int, ncols: int) -> tuple[int, int]:
    return int(cell) % ncols, int(cell) // ncols


def xy_to_cell(x: int, y: int, ncols: int) -> int:
    return int(y) * ncols + int(x)


def neighbours(cell: int, nrows: int, ncols: int) -> list[int]:
    """The 4-connected in-grid neighbours of ``cell``."""
    x, y = cell_to_xy(cell, ncols)
    out = []
    for dx, dy in DIRECTIONS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < ncols and 0 <= ny < nrows:
            out.append(xy_to_cell(nx, ny, ncols))
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_drift_matrix(nrows: int, ncols: int, weights: dict) -> np.ndarray:
    """
    Row-stochastic (n_cells, n_cells) matrix from per-direction weights.

    ``weights`` maps 'N'/'S'/'E'/'W'/'stay' to non-negative numbers; they need
    not sum to 1.  Directions leaving the grid are dropped and each row is
    renormalised over what remains, so probability is redistributed rather than
    lost at the boundary.
    """
    if all(float(weights.get(k, 0.0)) <= 0.0
           for k in list(DIRECTIONS) + ["stay"]):
        raise ValueError("at least one direction weight must be positive")

    n = nrows * ncols
    T = np.zeros((n, n), dtype=np.float64)

    for cell in range(n):
        x, y = cell_to_xy(cell, ncols)
        T[cell, cell] += float(weights.get("stay", 0.0))
        for name, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < ncols and 0 <= ny < nrows:
                T[cell, xy_to_cell(nx, ny, ncols)] += float(weights.get(name, 0.0))

        total = T[cell].sum()
        if total <= 0.0:
            # Every weighted direction left the grid; fall back to uniform over
            # whatever is reachable so the row stays a distribution.
            opts = neighbours(cell, nrows, ncols) + [cell]
            T[cell, opts] = 1.0 / len(opts)
        else:
            T[cell] /= total

    return T


def make_matrix_A(nrows: int = 5, ncols: int = 5) -> np.ndarray:
    """North-east drift."""
    return build_drift_matrix(nrows, ncols, A_WEIGHTS)


def make_matrix_B(nrows: int = 5, ncols: int = 5) -> np.ndarray:
    """South-west drift — the mirror of A."""
    return build_drift_matrix(nrows, ncols, B_WEIGHTS)


def make_matrix_C(nrows: int = 5, ncols: int = 5) -> np.ndarray:
    """
    Equal weight: the target is as likely to stay put as to move to any one
    adjacent cell.  The neutral regime the warm-restart agents train on.
    """
    n = nrows * ncols
    T = np.zeros((n, n), dtype=np.float64)
    for cell in range(n):
        opts = neighbours(cell, nrows, ncols) + [cell]
        T[cell, opts] = 1.0 / len(opts)
    return T


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def expected_displacement(T: np.ndarray, nrows: int, ncols: int
                          ) -> tuple[float, float]:
    """Mean (dx, dy) of one step, averaged uniformly over start cells."""
    n = nrows * ncols
    dxs = dys = 0.0
    for cell in range(n):
        x, y = cell_to_xy(cell, ncols)
        for dest in range(n):
            p = T[cell, dest]
            if p:
                dx_, dy_ = cell_to_xy(dest, ncols)
                dxs += p * (dx_ - x)
                dys += p * (dy_ - y)
    return dxs / n, dys / n


def validate_transition_matrix(T, nrows: int, ncols: int):
    """Check shape, row-stochasticity, and that no mass leaves the neighbourhood."""
    T = np.asarray(T, dtype=np.float64)
    n = nrows * ncols
    if T.shape != (n, n):
        return False, f"shape {T.shape}, expected {(n, n)}"
    if (T < 0).any():
        return False, "negative probabilities"
    if not np.allclose(T.sum(axis=1), 1.0):
        bad = int(np.argmax(np.abs(T.sum(axis=1) - 1.0)))
        return False, f"row {bad} sums to {T[bad].sum():.6f}, not 1"
    for cell in range(n):
        allowed = set(neighbours(cell, nrows, ncols)) | {cell}
        for dest in np.flatnonzero(T[cell]):
            if int(dest) not in allowed:
                return False, (f"row {cell} puts mass on {dest}, outside its "
                               f"4-connected neighbourhood")
    return True, "OK"


def describe_matrix(T: np.ndarray, nrows: int, ncols: int, name: str = "") -> str:
    dx, dy = expected_displacement(T, nrows, ncols)
    return (f"{name or 'matrix'}: {nrows}x{ncols}, "
            f"expected step (dx={dx:+.3f}, dy={dy:+.3f}), "
            f"mean self-loop {np.mean(np.diag(T)):.3f}")


__all__ = ["DIRECTIONS", "A_WEIGHTS", "B_WEIGHTS",
           "cell_to_xy", "xy_to_cell", "neighbours", "build_drift_matrix",
           "make_matrix_A", "make_matrix_B", "make_matrix_C",
           "expected_displacement", "validate_transition_matrix",
           "describe_matrix"]
