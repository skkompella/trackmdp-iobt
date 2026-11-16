"""
probing.py — coverage probes that keep every transition stream sampled.

The learner's own tau=0 steps sample only the cells it chooses to watch, so a
stream H(i, j) for a cell the policy ignores gets no data and a change confined
to it would never be seen.  CoverageProber guarantees a floor: since the last
reset, every stream must hold at least floor(rho * v(i)) samples, where v(i) is
the number of (i, tau=0) visits.  When some cell of S(i) falls below that floor
the next (i, 0) step becomes a probe.

A probe switches on all of S(i) when it fits the sensor budget (it then always
finds the object, so it costs energy but not tracking), otherwise the k
least-sampled cells of S(i).
"""
from __future__ import annotations

import math
from typing import Iterable


class CoverageProber:
    def __init__(self, supports: dict[int, tuple[int, ...]], max_sensors: int,
                 rho: float):
        if rho < 0.0:
            raise ValueError("rho must be non-negative")
        if max_sensors < 1:
            raise ValueError("max_sensors must be positive")
        self.supports = {int(i): tuple(int(j) for j in s)
                         for i, s in supports.items()}
        self.max_sensors = int(max_sensors)
        self.rho = float(rho)
        self.reset()

    def reset(self) -> None:
        self._visits = {i: 0 for i in self.supports}
        self._counts = {i: {j: 0 for j in s} for i, s in self.supports.items()}
        self.n_probes = 0

    def due(self, cell: int) -> bool:
        if self.rho <= 0.0:
            return False
        floor = math.floor(self.rho * self._visits[int(cell)])
        return any(n < floor for n in self._counts[int(cell)].values())

    def probe_cells(self, cell: int) -> list[int]:
        self.n_probes += 1
        support = self.supports[int(cell)]
        if len(support) <= self.max_sensors:
            return list(support)
        counts = self._counts[int(cell)]
        ranked = sorted(support, key=lambda j: (counts[j], j))
        return sorted(ranked[:self.max_sensors])

    def record(self, cell: int, watched: Iterable[int]) -> None:
        cell = int(cell)
        self._visits[cell] += 1
        counts = self._counts[cell]
        for j in set(int(w) for w in watched):
            if j in counts:
                counts[j] += 1


__all__ = ["CoverageProber"]
