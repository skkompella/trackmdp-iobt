"""
transition_detector.py — DARLING-style change detection on transition streams.

One Bernoulli stream H(i, j) per cell i and each cell j the object can reach
from i in one step (the support S(i)).  Only steps whose state is (i, tau=0)
feed it: the object was just seen at i, so its next cell is drawn from row
M[i, .] whatever the learner does.  For every switched-on j in S(i) the
stream receives 1{object found at j}.  Which cells get watched depends only on
the past, never on the current outcome, so each H(i, j) is i.i.d.
Bernoulli(M[i, j]) between changes (optional skipping).  That is what makes the
false-alarm budget delta_F mean what it says -- unlike the pooled hit stream,
whose misses cluster.

Each stream runs its own GLR test (change_detection.GLRChangeDetector) at
delta_F / n_streams, so delta_F bounds the false-alarm probability of the whole
detector by the union bound.  After the first alarm the detector latches:
further samples are ignored until reset(), so one episode yields at most one
alarm and the caller restarts at the episode end, as DARLING does.
"""
from __future__ import annotations

from typing import Iterable

from .change_detection import GLRChangeDetector
from .random_transition_matrices import grid_neighbourhoods, iobt_neighbourhoods


def row_supports(kind: str, nrows: int | None = None,
                 ncols: int | None = None) -> dict[int, tuple[int, ...]]:
    """S(i) = {i} plus the topological neighbours of i, as sorted tuples."""
    if kind == "grid":
        if nrows is None or ncols is None:
            raise ValueError("grid supports need nrows and ncols")
        neigh = grid_neighbourhoods(nrows, ncols)
    elif kind == "iobt":
        neigh = iobt_neighbourhoods()
    else:
        raise ValueError(f"kind must be 'grid' or 'iobt', got {kind!r}")
    return {i: tuple(sorted({i, *nb})) for i, nb in neigh.items()}


class TransitionStreamDetector:
    """Per-(cell, next cell) Bernoulli GLR streams with a union-bound threshold."""

    def __init__(self, supports: dict[int, tuple[int, ...]],
                 delta_F: float = 0.01, min_samples: int = 30,
                 max_window: int | None = 2000, exponent: float = 1.5):
        if not (0.0 < delta_F < 1.0):
            raise ValueError("delta_F must lie in (0, 1)")
        self.supports = {int(i): tuple(int(j) for j in s)
                         for i, s in supports.items()}
        self._support_sets = {i: frozenset(s) for i, s in self.supports.items()}
        self.n_streams = sum(len(s) for s in self.supports.values())
        self.delta_F = float(delta_F)
        self.delta_stream = self.delta_F / self.n_streams
        self._streams = {
            (i, j): GLRChangeDetector(delta=self.delta_stream,
                                      min_samples=min_samples,
                                      exponent=exponent, max_window=max_window)
            for i, s in self.supports.items() for j in s}
        self._counts = {key: 0 for key in self._streams}
        self.alarmed = False
        self.last_alarm: dict | None = None
        self.n_off_support = 0

    def observe(self, cell: int, watched: Iterable[int],
                found: int | None) -> bool:
        """Record one tau=0 step; True when this call raised the alarm."""
        if self.alarmed:
            return False
        support = self._support_sets[int(cell)]
        if found is not None and int(found) not in support:
            self.n_off_support += 1
            return False
        fired = None
        for j in sorted(support.intersection(int(w) for w in watched)):
            key = (int(cell), j)
            self._counts[key] += 1
            stream = self._streams[key]
            n = stream.n_samples + 1
            if stream.update(1.0 if j == found else 0.0) and fired is None:
                fired = (j, stream.last_statistic, stream.last_threshold, n)
        if fired is None:
            return False
        self.alarmed = True
        j, stat, thr, n = fired
        self.last_alarm = {"cell": int(cell), "next_cell": j,
                           "statistic": float(stat), "threshold": float(thr),
                           "n": int(n)}
        return True

    def counts(self, cell: int) -> dict[int, int]:
        return {j: self._counts[(int(cell), j)] for j in self.supports[int(cell)]}

    def reset(self) -> None:
        for stream in self._streams.values():
            stream.reset()
        for key in self._counts:
            self._counts[key] = 0
        self.alarmed = False
        self.last_alarm = None


__all__ = ["row_supports", "TransitionStreamDetector"]
