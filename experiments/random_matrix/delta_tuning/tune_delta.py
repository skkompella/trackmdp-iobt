#!/usr/bin/env python
"""
tune_delta.py — pick delta = T**-p for PPO + DAL offline, from recorded streams.

Replays a recorded per-iteration eval hit stream (train_switching_grid_ppo.py
--save-stream) through GLRChangeDetector with exactly the in-loop rules of
train_switching_grid_ppo.py: warm-up and cooldown iterations, at most one alarm
per iteration, detector reset while quiet, and the same switch-matching
tolerance (iterations_per_segment // 5).  Each delta costs milliseconds instead
of a 1-2 hour training run.

T is the number of training episodes in a run:
    15 segments x 200 iterations x 512 steps / 100 steps per episode = 15,360.

Streams
-------
  norestart  no restarts: every switch hits the OLD policy, as in a real DAL run
             before the alarm.  Scores detection (caught / missed / delay).
  oracle     warm restart at the true switches: contains the re-learning climb
             after a restart, which is what made earlier offline calibration
             over-fire in the loop.  Scores false alarms after restarts.

Usage
-----
    ./track_mdp_env/bin/python experiments/random_matrix/delta_tuning/tune_delta.py \\
        --streams experiments/random_matrix/delta_tuning/stream_norestart_k5_c0.04.npz \\
                  experiments/random_matrix/delta_tuning/stream_oracle_k5_c0.04.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from src.core.change_detection import GLRChangeDetector, beta_threshold  # noqa: E402
from src.core.online_schedule import detector_scores                      # noqa: E402


def replay(stream, delta, warmup=10, cooldown=10, min_samples=30, max_window=500):
    hits, lens = stream["hits"], stream["iter_len"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    det = GLRChangeDetector(delta=delta, min_samples=min_samples,
                            max_window=max_window)
    fired, suppress_until = [], 0
    for it, (s, n) in enumerate(zip(starts, lens)):
        if it < warmup or it < suppress_until:
            det.reset()
            continue
        for d in hits[s:s + n]:
            if det.update(d):
                fired.append(it)
                suppress_until = it + 1 + cooldown
                break
    switches = [int(x) for x in stream["switch_iters"]]
    tol = max(2, int(stream["iterations_per_segment"]) // 5)
    sc = detector_scores(fired, switches, tolerance=tol)
    delays = []
    for sw in switches:
        after = [f - sw for f in fired if 0 <= f - sw <= tol]
        if after:
            delays.append(min(after))
    sc["mean_delay_iters"] = float(np.mean(delays)) if delays else None
    sc["n_iterations"] = int(len(lens))
    return sc


def _replay_job(job):
    return replay(*job)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--T", type=float, default=15_360)
    ap.add_argument("--p-grid", default="0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    streams = {os.path.basename(p).split("_")[1]: dict(np.load(p))
               for p in args.streams}
    ps = [float(x) for x in args.p_grid.split(",")]
    rows = []
    hdr = f"{'delta':>12} {'p':>5} {'beta@500':>8}"
    for name in streams:
        hdr += f" | {name:>9}: alarms caught miss  FA delay"
    print(hdr, flush=True)
    grid = [("0.01", 0.01)] + [(f"T^-{p:g}", args.T ** -p) for p in ps]
    jobs = [(st, d) for _, d in grid for st in streams.values()]
    with ProcessPoolExecutor(args.workers) as ex:
        res = iter(list(ex.map(_replay_job, jobs)))
    for label, delta in grid:
        row = {"label": label, "delta": delta,
               "beta_full_window": beta_threshold(500, delta)}
        line = f"{label:>12} {'' if label == '0.01' else label[3:]:>5} {row['beta_full_window']:8.1f}"
        for name, st in streams.items():
            sc = next(res)
            row[name] = sc
            miss = sc["n_switches"] - sc["true_positive"]
            dl = "-" if sc["mean_delay_iters"] is None else f"{sc['mean_delay_iters']:.1f}"
            line += (f" | {'':>9}  {sc['n_detections']:>6} {sc['true_positive']:>6} "
                     f"{miss:>4} {sc['false_alarms']:>3} {dl:>5}")
        rows.append(row)
        print(line)

    # Best: fewest total errors (missed switches on the no-restart stream + false
    # alarms on every stream), ties broken by the shorter detection delay.
    def cost(r):
        miss = sum(r[n]["n_switches"] - r[n]["true_positive"]
                   for n in streams if n == "norestart") or 0
        fa = sum(r[n]["false_alarms"] for n in streams)
        d = r.get("norestart", next(iter(r[n] for n in streams)))["mean_delay_iters"]
        return (miss + fa, d if d is not None else 1e9)
    best = min(rows, key=cost)
    print(f"\nBest (fewest missed switches + false alarms): {best['label']} "
          f"= {best['delta']:.3e}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"T": args.T, "rows": rows, "best": best["label"]}, fh,
                      indent=1, default=float)


if __name__ == "__main__":
    main()
