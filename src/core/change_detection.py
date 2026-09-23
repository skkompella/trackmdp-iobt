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


def _kl_bernoulli_vec(p, q):
    """Array form of kl_bernoulli, used on the detector's hot path.

    The scalar version called once per split point cost ~1.3 ms per update,
    which dominated the online runs; this makes the scan a single numpy op.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    q = np.clip(np.asarray(q, dtype=np.float64), _EPS, 1.0 - _EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def gaussian_divergence(p: float, q: float, variance: float = 1.0) -> float:
    """
    Quadratic (Gaussian) divergence between means, for CONTINUOUS signals.

    The Bernoulli KL clamps its arguments into [0, 1], so feeding it a
    continuous signal such as a surprise or a reward silently destroys the
    information.  This is the variant DAL used for its linear-bandit case.
    """
    if variance is None or variance <= 0.0:
        raise ValueError("variance must be positive")
    return (float(p) - float(q)) ** 2 / (2.0 * float(variance))


def _gaussian_divergence_vec(p, q, variance):
    return (np.asarray(p, dtype=np.float64) - q) ** 2 / (2.0 * variance)


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

    DIVERGENCES = ("bernoulli", "gaussian")

    def __init__(self, delta: float = 0.01, min_samples: int = 10,
                 exponent: float = 1.5, max_window: int | None = 500,
                 divergence: str = "bernoulli", variance: float | None = None):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must lie in (0, 1)")
        if divergence not in self.DIVERGENCES:
            raise ValueError(
                f"divergence must be one of {self.DIVERGENCES}, got {divergence!r}")
        self.delta       = float(delta)
        self.min_samples = int(min_samples)
        self.exponent    = float(exponent)
        self.max_window  = max_window
        self.divergence  = divergence
        self.variance    = variance

        # Exposed after each update so a sweep can score without re-running.
        self.last_statistic: float | None = None
        self.last_threshold: float | None = None

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
            self.last_statistic = None
            self.last_threshold = None
            return False

        x = np.fromiter(self._window, dtype=np.float64, count=n)
        # Vectorised over every split point: prefix means vs suffix means.
        csum    = np.cumsum(x)
        total   = csum[-1]
        splits  = np.arange(1, n)
        mu1     = csum[:-1] / splits
        mu2     = (total - csum[:-1]) / (n - splits)
        mu      = total / n

        if self.divergence == "bernoulli":
            d1 = _kl_bernoulli_vec(mu1, mu)
            d2 = _kl_bernoulli_vec(mu2, mu)
        else:
            var = self.variance if self.variance else max(float(x.var()), 1e-9)
            d1 = _gaussian_divergence_vec(mu1, mu, var)
            d2 = _gaussian_divergence_vec(mu2, mu, var)
        stat = splits * d1 + (n - splits) * d2

        self.last_statistic = float(stat.max())
        self.last_threshold = beta_threshold(n, self.delta, self.exponent)

        if self.last_statistic > self.last_threshold:
            self.changepoints.append(self._t)
            self.reset()
            return True
        return False


    @property
    def critical_delta(self) -> float | None:
        """
        The largest delta that would NOT have fired on the latest update.

        Firing is ``stat > log(n**exponent / delta)``, i.e.
        ``delta > n**exponent * exp(-stat)``.  Inverting the threshold this way
        turns "would this configuration have fired?" into a comparison, which is
        what makes an offline delta sweep tractable.
        """
        if self.last_statistic is None:
            return None
        n = len(self._window)
        return float(np.clip(n ** self.exponent * np.exp(-self.last_statistic),
                             1e-300, 1.0 - 1e-12))


def arl0(stream, delta: float, min_samples: int = 30, exponent: float = 1.5,
         max_window: int | None = 500, divergence: str = "bernoulli",
         variance: float | None = None) -> float:
    """
    Average Run Length under the null: mean steps between alarms on a stream
    that contains NO change.  Every alarm on such a stream is false, so this is
    the false-alarm axis a detector should be specified on.

    Returns inf when nothing fires.
    """
    det = GLRChangeDetector(delta=delta, min_samples=min_samples,
                            exponent=exponent, max_window=max_window,
                            divergence=divergence, variance=variance)
    fires = [i for i, v in enumerate(stream) if det.update(v)]
    if not fires:
        return float("inf")
    return float(len(stream)) / len(fires)


def calibrate_delta(stream, target_arl0: float, min_samples: int = 30,
                    exponent: float = 1.5, max_window: int | None = 500,
                    divergence: str = "bernoulli", variance: float | None = None,
                    lo: float = 1e-12, hi: float = 0.5,
                    iterations: int = 16) -> float:
    """
    Solve for the delta achieving ``target_arl0`` on a change-free reference
    stream, by bisection on log(delta).

    This is the actual fix for "the detector is not calibrated".  delta stops
    being a number someone picked and becomes a derived quantity, so the same
    specification ("one false alarm per N steps") yields different deltas for
    streams fed at different rates -- which is exactly the mismatch that made
    the tabular arm fire at a 1.0pp shift and PPO at 2.1pp.

    ARL0 increases as delta shrinks, so the search is monotone.
    """
    if target_arl0 <= 0:
        raise ValueError("target_arl0 must be positive")

    def achieved(d):
        return arl0(stream, d, min_samples, exponent, max_window,
                    divergence, variance)

    lo_d, hi_d = float(lo), float(hi)
    for _ in range(iterations):
        mid = float(np.sqrt(lo_d * hi_d))          # geometric midpoint
        if achieved(mid) < target_arl0:
            hi_d = mid                             # too many alarms -> shrink
        else:
            lo_d = mid
    return float(np.sqrt(lo_d * hi_d))


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


__all__ = ["kl_bernoulli", "gaussian_divergence", "beta_threshold",
           "GLRChangeDetector", "RestartController", "arl0", "calibrate_delta"]
