#!/usr/bin/env python
"""
random_matrix_oracle.py — best-possible reference for each random matrix.

An oracle that KNOWS the active transition matrix watches the m most likely
cells of its prediction (one-step row at delay 0, two-step at delay 1).  For
each matrix, k and sensor cost it reports:

  best-reward  the m with the highest reward/step -- what a reward-maximising
               learner should converge to, so the realistic ceiling
  full-budget  m = k -- the most accuracy the sensor budget allows

Not a learner: it reads the matrix, so it is a reference, not a competitor.

Example
-------
    ./track_mdp_env/bin/python examples/random_matrix_oracle.py \\
        --out experiments/random_matrix/oracle.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.matrix_iobt_env import MatrixIoBTEnv                  # noqa: E402
from src.core.random_transition_matrices import parse_matrix_spec   # noqa: E402
from src.core.switching_grid_env import SwitchingTransitionGridEnv  # noqa: E402

TL = 1


def make_env(env, T, k, cost, seed, nrows=5, ncols=5):
    if env == "grid":
        n = nrows * ncols
        e = SwitchingTransitionGridEnv(nrows, ncols, k, k, n * (TL + 1) + 1, TL,
                                       [T], seed=seed, no_moore_constraint=True,
                                       sensor_rew=-cost)
        return e, n, n
    e = MatrixIoBTEnv(k, k, 16 * (TL + 1) + 1, TL, [T], seed=seed,
                      reward_preset="grid", sensor_rew=-cost)
    return e, 10, 16          # 10 real nodes, 16-wide action vector


def run_oracle(env, T, k, cost, m, episodes, seed, nrows=5, ncols=5):
    e, n, width = make_env(env, T, k, cost, seed, nrows, ncols)
    e.force_matrix(0)
    miss = e.missing_state
    T2 = T @ T
    hits = steps = cells = tracked = 0
    reward = 0.0
    for _ in range(episodes):
        e.reset_object_state()
        s, d = miss, 0
        for _ in range(100):
            a = np.zeros(width, dtype=np.int64)
            lost = s == miss
            if not lost:
                p = (T if d == 0 else T2)[s // (TL + 1)]
                a[np.argsort(-p)[:min(m, k)]] = 1
                cells += a.sum()
                tracked += 1
            hits += int(not lost and a[e.object_pos] == 1)
            steps += 1
            r, s, _, d = e.get_reward_next_state(s, a, d)
            reward += r
    return {"m": m, "accuracy": hits / steps, "reward_per_step": reward / steps,
            "cells_per_tracked_step": cells / max(tracked, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrices", default="R0,R1,R2,R3,R4")
    p.add_argument("--matrix-seed", type=int, default=20261006)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--envs", default="grid,iobt")
    p.add_argument("--ks", default="4,5")
    p.add_argument("--nrows", type=int, default=5)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results = []
    print(f"{'env':>4} {'mat':>3} {'k':>2} {'cost':>5} | {'best-reward m':>13} "
          f"{'acc':>7} {'cells':>5} | {'full-budget acc':>15}")
    for env in args.envs.split(","):
        names, mats = parse_matrix_spec(args.matrices, env, args.matrix_seed,
                                        nrows=args.nrows, ncols=args.ncols)
        for name, T in zip(names, mats):
            for k in (int(x) for x in args.ks.split(",")):
                for cost in (0.16, 0.04):
                    rows = [run_oracle(env, T, k, cost, m, args.episodes, 3,
                                       args.nrows, args.ncols)
                            for m in range(1, k + 1)]
                    best = max(rows, key=lambda r: r["reward_per_step"])
                    full = rows[-1]
                    results.append({"env": env, "matrix": name, "k": k,
                                    "cost": -cost, "best_reward": best,
                                    "full_budget": full, "all": rows})
                    print(f"{env:>4} {name:>3} {k:>2} {-cost:>5} | {best['m']:>13} "
                          f"{best['accuracy']*100:6.2f}% "
                          f"{best['cells_per_tracked_step']:5.2f} | "
                          f"{full['accuracy']*100:14.2f}%")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
