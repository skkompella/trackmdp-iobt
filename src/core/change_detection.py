"""
change_detection.py — GLR change detection and learner restarts.

The tracker's per-step outcome is Bernoulli: 1 when the object was tracked and
found, 0 otherwise.  When the object's trajectory switches to a different loop,
a policy tuned to the old loop starts missing, and that shows up as a drop in
the success rate.  A generalised-likelihood-ratio detector on that stream turns
"the trajectory changed" into a decision that needs only a handful of samples,
which is the mechanism that lets a learner adapt in a few steps rather than a
few thousand.

Method: for every split of the observation window, compare a two-segment fit
(each half at its own mean) against the pooled single-mean fit, using the
Bernoulli KL divergence, and declare a change when the statistic exceeds
beta(n, delta) = log(n**exponent / delta).

This is our own implementation of the standard GLR procedure.  The design was
motivated by the DAL reference implementation (Gerogiannis et al., "DAL: A
Practical Prior-Free Black-Box Framework for Piecewise Stationary Bandits"),
which applies the same test to a bandit's cumulative-reward sequence and resets
the base learner on detection; no code from that project is reproduced here.
"""
from __future__ import annotations

from collections import deque

import numpy as np

_EPS = 1e-12


def kl_bernoulli(p: float, q: float) -> float:
    """KL(p || q) for Bernoulli means, clamped away from 0 and 1."""
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    q = min(max(float(q), _EPS), 1.0 - _EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def beta_threshold(n: int, delta: float, exponent: float = 1.5) -> float:
    """Confidence threshold the GLR statistic must exceed to declare a change."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    return float(np.log((n ** exponent) / delta))


class GLRChangeDetector:
    """
    Bernoulli GLR detector over a stream of 0/1 outcomes.

    After firing it resets itself, so the next segment is judged on its own
    evidence rather than against a window that straddles the change.
    """

    def __init__(self, delta: float = 0.01, min_samples: int = 10,
                 exponent: float = 1.5, max_window: int | None = 500):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must lie in (0, 1)")
        self.delta       = float(delta)
        self.min_samples = int(min_samples)
        self.exponent    = float(exponent)
        self.max_window  = max_window

        self._window: deque[float] = deque(maxlen=max_window)
        self._t          = 0            # total observations ever seen
        self.changepoints: list[int] = []

    # ── state ──────────────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self._window)

    def reset(self) -> None:
        self._window.clear()

    # ── the test ───────────────────────────────────────────────────────────

    def update(self, observation: float) -> bool:
        """Push one outcome; return True if a change is declared at this step."""
        self._t += 1
        self._window.append(float(observation))

        n = len(self._window)
        if n < self.min_samples:
            return False

        x = np.fromiter(self._window, dtype=np.float64, count=n)
        # Vectorised over every split point: prefix means vs suffix means.
        csum    = np.cumsum(x)
        total   = csum[-1]
        splits  = np.arange(1, n)
        mu1     = csum[:-1] / splits
        mu2     = (total - csum[:-1]) / (n - splits)
        mu      = total / n

        kl1  = np.array([kl_bernoulli(m, mu) for m in mu1])
        kl2  = np.array([kl_bernoulli(m, mu) for m in mu2])
        stat = splits * kl1 + (n - splits) * kl2

        if stat.max() > beta_threshold(n, self.delta, self.exponent):
            self.changepoints.append(self._t)
            self.reset()
            return True
        return False


class RestartController:
    """
    Applies a restart policy to a tabular learner when a change is detected.

    Strategies
    ----------
    cold     Wipe the table to its optimistic initialisation.  The honest
             measure of adaptation speed, and what DAL's base learner does.
    warm     Restore a prior table trained across all loops.  Recovers much
             faster and is the realistic deployment story, but it partly
             pre-solves the problem being measured.
    library  Archive the finished segment's table and restore the archived
             table that scored best so far, or start fresh if the archive is
             empty.

    Note on ``library``: selecting by best-recorded-score is a heuristic, not a
    matched-expert rule — it reuses the strongest policy learned so far rather
    than the one belonging to the current loop.  Identifying the right expert
    would need a probation phase that scores candidates on fresh data; that is
    deliberately out of scope here and the results should be read accordingly.
    """

    STRATEGIES = ("cold", "warm", "library")

    def __init__(self, agent, strategy: str = "cold", prior=None):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"unknown restart strategy {strategy!r}; "
                f"choose one of {list(self.STRATEGIES)}"
            )
        if strategy == "warm" and prior is None:
            raise ValueError("strategy 'warm' requires a prior Q-table")

        self.agent      = agent
        self.strategy   = strategy
        self.prior      = None if prior is None else np.asarray(prior).copy()
        self.library: list[tuple[float, np.ndarray]] = []
        self.n_restarts = 0

    def restart(self, segment_reward: float = 0.0) -> None:
        """Called on a detected change. ``segment_reward`` scores the segment
        just finished, which the library strategy uses to rank archives."""
        self.n_restarts += 1

        if self.strategy == "cold":
            self.agent.reset()
            return

        if self.strategy == "warm":
            self.agent.restore(self.prior)
            return

        # library
        best_before = max(self.library, key=lambda kv: kv[0], default=None)
        self.library.append((float(segment_reward), self.agent.snapshot()))
        if best_before is None:
            self.agent.reset()
        else:
            self.agent.restore(best_before[1])


__all__ = ["kl_bernoulli", "beta_threshold", "GLRChangeDetector",
           "RestartController"]
