"""
ppo_transition.py — feed the transition-stream detector from PPO rollout workers.

Each rollout env (src/core/grid_wrapper_rect.py with env_config["record_tau0"])
buffers one record per tau=0 step: (cell, watched cells, found cell or None),
built by grid_env_rect.get_reward_next_state.  After every algo.train() the
driver drains all workers (drain_records) and TransitionMonitor feeds the
records to one TransitionStreamDetector.

On an alarm the monitor logs it and clears the detector; what to do with the
learner (nothing in monitor mode, a restart otherwise) is the caller's choice.
The detector latches inside a batch, so one iteration yields at most one alarm.

Streams: optional per-(i, j) int8 sample lists (streams.npz, the format
examples/replay_streams.py reads).  Records: optional ordered log of every
record (records.npz: iteration, cell, watched cells as a bitmask over the grid,
found cell or -1), enough to replay the detector offline at any delta_F.
"""
from __future__ import annotations

import json
import os
from array import array

import numpy as np

from .transition_detector import TransitionStreamDetector


def drain_records(algo) -> list:
    """All tau=0 records buffered in the rollout envs, in worker order."""
    per_worker = algo.env_runner_group.foreach_env(
        lambda e: e.drain_tau0() if hasattr(e, "drain_tau0") else [])
    return [rec for worker in per_worker for env_recs in worker for rec in env_recs]


class TransitionMonitor:
    def __init__(self, supports: dict[int, tuple[int, ...]], delta_F: float = 0.01,
                 min_samples: int = 30, max_window: int | None = 2000,
                 save_streams: bool = False, save_records: bool = False):
        self.detector = TransitionStreamDetector(
            supports, delta_F=delta_F, min_samples=min_samples, max_window=max_window)
        self.settings = {"delta_F": delta_F, "min_samples": min_samples,
                         "max_window": max_window}
        self.alarms: list[dict] = []
        self.n_records = 0
        self.streams = ({key: array("b") for key in self.detector._streams}
                        if save_streams else None)
        self._log = ({k: array(t) for k, t in
                      (("iteration", "l"), ("cell", "h"), ("active", "q"), ("found", "h"))}
                     if save_records else None)

    def feed(self, records, iteration: int, env_steps: int | None = None) -> dict | None:
        """Feed one iteration's records; return the alarm it raised, if any."""
        det = self.detector
        for cell, active, found in records:
            self.n_records += 1
            det.observe(cell, active, found)
            if self.streams is not None:
                for j in active:
                    s = self.streams.get((cell, j))
                    if s is not None:
                        s.append(1 if j == found else 0)
            if self._log is not None:
                self._log["iteration"].append(iteration)
                self._log["cell"].append(cell)
                self._log["active"].append(sum(1 << j for j in active))
                self._log["found"].append(-1 if found is None else found)
        if not det.alarmed:
            return None
        alarm = {**det.last_alarm, "iteration": int(iteration),
                 "env_steps": None if env_steps is None else int(env_steps)}
        self.alarms.append(alarm)
        det.reset()
        return alarm

    def update(self, algo, iteration: int, env_steps: int | None = None) -> dict | None:
        return self.feed(drain_records(algo), iteration, env_steps)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "alarms.json"), "w") as fh:
            json.dump({"alarms": self.alarms, "n_alarms": len(self.alarms),
                       "n_streams": self.detector.n_streams,
                       "n_records": self.n_records,
                       "n_off_support": self.detector.n_off_support,
                       **self.settings}, fh, indent=1)
        if self.streams is not None:
            np.savez_compressed(os.path.join(out_dir, "streams.npz"),
                                **{f"{i}_{j}": np.frombuffer(s, dtype=np.int8)
                                   for (i, j), s in self.streams.items()})
        if self._log is not None:
            np.savez_compressed(os.path.join(out_dir, "records.npz"),
                                **{k: np.array(v) for k, v in self._log.items()})


__all__ = ["drain_records", "TransitionMonitor"]
