"""
random_transition_matrices.py — random local movement models for both envs.

Each row puts probability only on the cell itself and its neighbours, drawn
from a symmetric Dirichlet.  That keeps the object moving one step at a time
(as in every other experiment here) while making each environment's drift
pattern arbitrary rather than hand-designed.

The repo's older ``make_random_transition_matrix`` (examples/finetune_
deterministic.py) draws rows over ALL cells, so the object teleports and two
draws look statistically identical to a tracker.  These do not.

Neighbourhoods:
  grid  4-connected in-grid neighbours (grid_transition_matrices.neighbours)
  iobt  IOBT_MAP's out-edges, as used by EqualProbIoBTEnv

Matrix specs used by the training scripts (``parse_matrix_spec``):
  "C"      equal probability over stay + every neighbour
  "R<i>"   random matrix i, drawn with seed ``matrix_seed + i``
  "A","B"  the hand-designed grid drift matrices (grid only)
  "O<i>"   original Track-MDP movement, drawn with seed ``matrix_seed + i``
           (grid only; 3x3 Moore neighbourhood, so diagonal moves too)
"""
from __future__ import annotations

import numpy as np

from .grid_transition_matrices import (
    make_matrix_A, make_matrix_B, neighbours as grid_neighbours_of,
)
from .iobt_loops import IOBT_MAP, IOBT_NUM_NODES

DEFAULT_CONCENTRATION = 1.0


def grid_neighbourhoods(nrows: int, ncols: int) -> dict[int, list[int]]:
    return {c: grid_neighbours_of(c, nrows, ncols) for c in range(nrows * ncols)}


def iobt_neighbourhoods() -> dict[int, list[int]]:
    return {n: list(IOBT_MAP[n]) for n in range(IOBT_NUM_NODES)}


def equal_matrix(neigh: dict[int, list[int]]) -> np.ndarray:
    """Stay or move to any neighbour, all equally likely."""
    n = len(neigh)
    T = np.zeros((n, n))
    for c, nb in neigh.items():
        opts = [c] + list(nb)
        T[c, opts] = 1.0 / len(opts)
    return T


def random_matrix(neigh: dict[int, list[int]], seed: int,
                  concentration: float = DEFAULT_CONCENTRATION) -> np.ndarray:
    """Each row ~ Dirichlet(concentration) over stay + neighbours."""
    rng = np.random.default_rng(seed)
    n = len(neigh)
    T = np.zeros((n, n))
    for c in range(n):
        opts = [c] + list(neigh[c])
        T[c, opts] = rng.dirichlet(np.full(len(opts), float(concentration)))
    return T


TRACKMDP_CUM_PROB = (0.2817, 0.5633, 0.845, 0.995)   # num_trans=4, stay-run 0.15, exit 0.005


def original_matrix(nrows: int, ncols: int, seed: int,
                    cum_prob=TRACKMDP_CUM_PROB) -> np.ndarray:
    """
    The original Track-MDP movement model (environment.object_transition_
    matrix_random_3x3) as a transition matrix: each cell draws len(cum_prob)
    destinations, with replacement, from its in-grid 3x3 neighbourhood (stay
    included); slot i gets the gap cum_prob[i] - cum_prob[i-1]. The exit mass
    (1 - cum_prob[-1]) is left to the environment, so rows are renormalised.
    """
    from .grid_transition_matrices import moore_neighbours
    rng = np.random.default_rng(seed)
    gaps = np.diff(np.concatenate([[0.0], cum_prob]))
    n = nrows * ncols
    T = np.zeros((n, n))
    for cell in range(n):
        nbhd = [cell] + moore_neighbours(cell, nrows, ncols)
        for g, j in zip(gaps, rng.integers(len(nbhd), size=len(gaps))):
            T[cell, nbhd[j]] += g
    return T / T.sum(axis=1, keepdims=True)


def parse_matrix_spec(spec: str, env: str, matrix_seed: int,
                      concentration: float = DEFAULT_CONCENTRATION,
                      nrows: int = 5, ncols: int = 5):
    """Turn "R0,R1,C" into (names, matrices) for ``env`` in {"grid", "iobt"}."""
    neigh = (grid_neighbourhoods(nrows, ncols) if env == "grid"
             else iobt_neighbourhoods())
    names, mats = [], []
    for tok in (t.strip().upper() for t in spec.split(",") if t.strip()):
        if tok == "C":
            mats.append(equal_matrix(neigh))
        elif tok.startswith("R") and tok[1:].isdigit():
            mats.append(random_matrix(neigh, matrix_seed + int(tok[1:]),
                                      concentration))
        elif env == "grid" and tok == "A":
            mats.append(make_matrix_A(nrows, ncols))
        elif env == "grid" and tok == "B":
            mats.append(make_matrix_B(nrows, ncols))
        elif env == "grid" and tok.startswith("O") and tok[1:].isdigit():
            mats.append(original_matrix(nrows, ncols, matrix_seed + int(tok[1:])))
        else:
            raise ValueError(f"unknown matrix token {tok!r} for env {env!r}")
        names.append(tok)
    return names, mats


def describe_random(T: np.ndarray, name: str = "") -> str:
    """One line: how peaked the rows are (mean max-row probability)."""
    nz = [row[row > 0] for row in T]
    peak = np.mean([r.max() for r in nz])
    stay = np.mean(np.diag(T))
    return f"{name:>4}: mean top move prob {peak:.2f}, mean stay prob {stay:.2f}"


__all__ = ["grid_neighbourhoods", "iobt_neighbourhoods", "equal_matrix",
           "random_matrix", "original_matrix", "parse_matrix_spec", "describe_random",
           "DEFAULT_CONCENTRATION"]
