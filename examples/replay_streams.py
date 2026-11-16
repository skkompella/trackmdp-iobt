#!/usr/bin/env python
"""
replay_streams.py — ordered vs shuffled replay of recorded transition streams.

The null check for the transition-stream detector: if every stream really is
i.i.d. under no change, shuffling each stream's samples cannot change how often
the GLR fires.  If the ordered replay fires much more often than the shuffled
ones, the streams carry serial structure (as the old pooled hit stream did) and
delta_F does not mean a false-alarm rate.

glr_alarm_steps is a prefix-sum implementation of GLRChangeDetector's exact
procedure (same window cap, minimum samples, threshold and reset-on-fire); a
test pins the two to identical alarm steps.  It avoids the reference
implementation's per-sample deque conversion, which makes replaying a full run
take minutes instead of hours.

Usage:
    ./track_mdp_env/bin/python examples/replay_streams.py \\
        --streams RUN/streams.npz --delta-F 0.01,0.1,0.5 --n-shuffles 3 \\
        --out RUN/replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.change_detection import _kl_bernoulli_vec  # noqa: E402


def glr_alarm_steps(x, delta: float, min_samples: int = 30,
                    max_window: int | None = 2000,
                    exponent: float = 1.5) -> list[int]:
    """Indices at which GLRChangeDetector(delta, ...) would fire on x."""
    x = np.asarray(x, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    fires, start = [], 0
    for t in range(1, len(x) + 1):
        lo = start if max_window is None else max(start, t - max_window)
        n = t - lo
        if n < min_samples:
            continue
        total = csum[t] - csum[lo]
        if total == 0.0 or total == n:          # constant window: statistic 0
            continue
        prefix = csum[lo + 1:t] - csum[lo]
        splits = np.arange(1, n)
        mu = total / n
        stat = (splits * _kl_bernoulli_vec(prefix / splits, mu)
                + (n - splits) * _kl_bernoulli_vec((total - prefix) / (n - splits), mu))
        if stat.max() > np.log((n ** exponent) / delta):
            fires.append(t - 1)
            start = t
    return fires


def replay(streams: dict[str, np.ndarray], delta_F: float,
           shuffle_seed: int | None, min_samples: int = 30,
           max_window: int | None = 2000) -> int:
    """Total alarms over all streams at delta_F / n_streams per stream."""
    delta = delta_F / len(streams)
    rng = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    total = 0
    for key in sorted(streams):
        x = np.asarray(streams[key])
        if rng is not None:
            x = rng.permutation(x)
        total += len(glr_alarm_steps(x, delta, min_samples, max_window))
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", required=True)
    p.add_argument("--delta-F", type=str, default="0.01,0.1,0.5")
    p.add_argument("--n-shuffles", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z = np.load(args.streams)
    streams = {k: z[k] for k in z.files}
    result = {}
    for d in (float(v) for v in args.delta_F.split(",")):
        result[str(d)] = {
            "ordered": replay(streams, d, None),
            "shuffled": [replay(streams, d, seed) for seed in range(args.n_shuffles)],
        }
        print(f"  delta_F {d}: ordered {result[str(d)]['ordered']}, "
              f"shuffled {result[str(d)]['shuffled']}", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"streams": args.streams, "n_streams": len(streams),
                   "n_samples": int(sum(len(v) for v in streams.values())),
                   "replay": result}, fh, indent=2)


if __name__ == "__main__":
    main()
