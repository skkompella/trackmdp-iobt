"""
detection_signals.py — candidate signals for change detection, and a recorder.

The detector has so far watched ONE signal: per-step tracking success. Measured
on the switching grid, that signal is a poor place to look for a regime change:

  - Two of five switches produced ~0pp change in it. A change absent from the
    stream cannot be found at any threshold.
  - Its signal-to-confound ratio is about 2x. The learner's own improvement
    moves tracking success by 5-12pp while a switch moves it 5-39pp, and both
    are genuine distribution shifts, so the GLR cannot tell them apart.

So this module offers alternatives, and the calibration experiment compares
them at matched false-alarm rate.

OBSERVABILITY — the constraint every signal here respects. On a successful
detection the environment encodes the object's true cell into next_state
(grid_env_rect.py:576, multiloop_iobt_env.py:186), and the agent already
decodes it as its own state. Reading it is not cheating. On a MISS the object's
position is genuinely unknown, and no signal here uses it. The active
matrix/loop index is never used either -- it is ground truth for scoring only,
which is why SignalRecorder keeps switch marks apart from the signal streams.

SAMPLE RATES DIFFER, which is the whole reason streams carry step indices.
`hit` updates every step; `transition_surprise` only when two consecutive
observations exist. Average run lengths must therefore be computed in
ENVIRONMENT STEPS, not in samples -- comparing at matched delta across streams
of different rates is exactly what miscalibrated the detector in the first place.
"""
from __future__ import annotations

import json

import numpy as np

DENSE_SIGNALS = ("hit", "staleness", "reward", "miss_run", "sensors")
SPARSE_SIGNALS = ("transition_surprise",)


class TransitionSurprise:
    """
    Running estimate of P(next cell | current cell), reported as -log p.

    The transition matrix is what actually changes when the regime switches, so
    unlike tracking success this signal is a property of the environment rather
    than of how well the agent happens to be doing.

    KNOWN BIAS, to be measured rather than assumed: the agent only observes
    transitions out of cells it chose to watch, so a policy specialised on one
    regime samples that regime's transitions and misses the other's. The pairs
    it does get are correct; the ones it misses are missing not-at-random.
    Partly offsetting this, the missing-state rescan always succeeds, so a
    struggling policy gets MORE forced re-acquisitions and therefore more
    uncensored observations.
    """

    def __init__(self, n_cells: int, alpha: float = 0.5, decay: float = 1.0):
        if n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must lie in (0, 1]")
        self.n_cells = int(n_cells)
        self.alpha   = float(alpha)
        self.decay   = float(decay)
        self.counts  = np.full((self.n_cells, self.n_cells), self.alpha,
                               dtype=np.float64)
        self._prev: int | None = None

    def reset(self) -> None:
        self.counts.fill(self.alpha)
        self._prev = None

    def observe(self, cell: int | None) -> float | None:
        """
        Feed the observed cell, or None when the object was not detected.

        Returns the surprise of the (previous -> current) transition when a
        consecutive pair is available, else None. A miss breaks the chain: a
        pair may not span an unobserved step, because the intervening position
        is genuinely unknown.
        """
        if cell is None:
            self._prev = None
            return None
        cell = int(cell)
        if not (0 <= cell < self.n_cells):
            raise ValueError(f"cell {cell} out of range 0-{self.n_cells - 1}")

        prev = self._prev
        self._prev = cell
        if prev is None:
            return None

        row = self.counts[prev]
        p = row[cell] / row.sum()
        surprise = float(-np.log(p))       # finite: the alpha prior floors p

        if self.decay < 1.0:
            self.counts[prev] *= self.decay
        self.counts[prev, cell] += 1.0
        return surprise


class SignalRecorder:
    """
    Collects every candidate signal from one run, plus ground-truth switch
    marks, for offline detector calibration.

    Ground truth is deliberately stored apart from the signals: it is used to
    score detections and must never be fed to a detector.
    """

    def __init__(self, n_cells: int, alpha: float = 0.5, decay: float = 1.0):
        self.n_cells = int(n_cells)
        self._ts = TransitionSurprise(n_cells, alpha=alpha, decay=decay)
        self.total_steps = 0
        self.switch_steps: list[int] = []
        self._dense: dict[str, list[float]] = {k: [] for k in DENSE_SIGNALS}
        self._sparse_steps: list[int] = []
        self._sparse_vals: list[float] = []
        self._miss_run = 0

    # ── recording ──────────────────────────────────────────────────────────

    def step(self, hit, cell, delay, reward, sensors) -> None:
        """One environment step. ``cell`` is None when the object was missed."""
        hit = int(bool(hit))
        self._miss_run = 0 if hit else self._miss_run + 1

        self._dense["hit"].append(hit)
        self._dense["staleness"].append(float(delay))
        self._dense["reward"].append(float(reward))
        self._dense["miss_run"].append(float(self._miss_run))
        self._dense["sensors"].append(float(sensors))

        s = self._ts.observe(cell)
        if s is not None:
            self._sparse_steps.append(self.total_steps)
            self._sparse_vals.append(s)

        self.total_steps += 1

    def mark_switch(self) -> None:
        """Record that the regime changes before the NEXT step."""
        self.switch_steps.append(self.total_steps)

    # ── access ─────────────────────────────────────────────────────────────

    def streams(self) -> dict:
        out = {name: {"steps": list(range(len(vals))), "values": list(vals)}
               for name, vals in self._dense.items()}
        out["transition_surprise"] = {"steps": list(self._sparse_steps),
                                      "values": list(self._sparse_vals)}
        return out

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path, meta: dict | None = None) -> None:
        payload = {"__meta__": json.dumps(meta or {}),
                   "__switch_steps__": np.asarray(self.switch_steps, dtype=np.int64),
                   "__total_steps__": np.asarray([self.total_steps], dtype=np.int64)}
        for name, st in self.streams().items():
            payload[f"{name}__steps"] = np.asarray(st["steps"], dtype=np.int64)
            payload[f"{name}__values"] = np.asarray(st["values"], dtype=np.float64)
        np.savez_compressed(path, **payload)

    @staticmethod
    def load(path) -> dict:
        z = np.load(path, allow_pickle=False)
        signals: dict[str, dict] = {}
        for key in z.files:
            if key.startswith("__"):
                continue
            name, kind = key.rsplit("__", 1)
            signals.setdefault(name, {})[kind] = z[key].tolist()
        return {"meta": json.loads(str(z["__meta__"])),
                "switch_steps": z["__switch_steps__"].tolist(),
                "total_steps": int(z["__total_steps__"][0]),
                "signals": signals}


__all__ = ["TransitionSurprise", "SignalRecorder",
           "DENSE_SIGNALS", "SPARSE_SIGNALS"]
