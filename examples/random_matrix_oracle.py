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


def make_env(env, T, k, cost, seed, nrows=5, ncols=5, moore_moves=False):
    if env == "grid":
        n = nrows * ncols
        e = SwitchingTransitionGridEnv(nrows, ncols, k, k, n * (TL + 1) + 1, TL,
                                       [T], seed=seed, no_moore_constraint=True,
                                       sensor_rew=-cost, moore_moves=moore_moves)
        return e, n, n
    e = MatrixIoBTEnv(k, k, 16 * (TL + 1) + 1, TL, [T], seed=seed,
                      reward_preset="grid", sensor_rew=-cost)
    return e, 10, 16          # 10 real nodes, 16-wide action vector


def run_oracle(env, T, k, cost, m, episodes, seed, nrows=5, ncols=5,
               moore_moves=False):
    e, n, width = make_env(env, T, k, cost, seed, nrows, ncols, moore_moves)
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


def run_oracle_standard(T, nrows, ncols, cost, theta, episodes, seed,
                        exit_prob=0.005, eval_cap=1000):
    """
    Oracle under standard.md: infinite horizon (exit 0.005, episodes run until
    the target exits), scored with BOTH the upstream metric (rescan steps count
    as found; mean of per-episode ratios) and the tracked-only metric.

    Per-cell decision: knowing T, it switches on every cell whose predicted
    probability is >= theta (one step ahead from the last-seen cell at delay 0,
    two steps at delay 1), so the number of sensors varies cell by cell.
    theta -> 0 watches every possible next cell (full coverage); sweeping theta
    finds the reward-optimal threshold.
    """
    from src.core.grid_env_rect import learning_grid_sarsa_0
    q = learning_grid_sarsa_0(0, nrows, ncols, 4, [0.2817, 0.5633, 0.845, 0.995],
                              nrows * ncols, nrows * ncols, TL, TL,
                              no_moore_constraint=True, seed=seed)
    e = q.grid_env
    e.load_movement_matrix(T)
    e.sensor_rew = -cost
    n, miss = nrows * ncols, q.missing_state
    T2 = T @ T
    ep_acc, found_trk, steps_all, cells, tracked, reward = [], 0, 0, 0, 0, 0.0
    for _ in range(episodes):
        e.reset_object_state()
        s, d, ep_found, ep_steps = miss, 0, 0, 0
        while ep_steps < eval_cap:
            was_missing = s == miss
            a = np.zeros(n, dtype=np.int64)
            if not was_missing:
                p = (T if d == 0 else T2)[s // (TL + 1)]
                a[p >= max(theta, 1e-12)] = 1
                cells += int(a.sum()); tracked += 1
            r, s_next, term, d = e.get_reward_next_state(s, a, d)
            hit = (not was_missing) and d == 0
            found_trk += int(hit)
            ep_found += 1 if was_missing else int(hit)
            ep_steps += 1; steps_all += 1; reward += r
            if term:
                break
            s = s_next
        ep_acc.append(ep_found / max(ep_steps, 1))
    return {"theta": theta, "accuracy": float(np.mean(ep_acc)),
            "tracked_accuracy": found_trk / max(steps_all, 1),
            "cells_per_tracked_step": cells / max(tracked, 1),
            "reward_per_step": reward / max(steps_all, 1)}


THETAS = (0.0, 0.05, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5)


def main_standard(args):
    results = []
    print(f"{'mat':>4} | {'reward-optimal: theta':>21} {'acc':>7} {'tracked':>8} "
          f"{'cells':>5} {'rew/step':>8} | {'full coverage: acc':>18} {'tracked':>8} {'cells':>5}")
    names, mats = parse_matrix_spec(args.matrices, "grid", args.matrix_seed,
                                    nrows=args.nrows, ncols=args.ncols)
    for name, T in zip(names, mats):
        for cost in (float(c) for c in args.costs.split(",")):
            rows = [run_oracle_standard(T, args.nrows, args.ncols, cost, th,
                                        args.episodes, 3) for th in THETAS]
            best, full = max(rows, key=lambda r: r["reward_per_step"]), rows[0]
            results.append({"matrix": name, "cost": -cost, "best_reward": best,
                            "full_coverage": full, "all": rows})
            print(f"{name:>4} | {best['theta']:>21} {best['accuracy']*100:6.2f}% "
                  f"{best['tracked_accuracy']*100:7.2f}% {best['cells_per_tracked_step']:5.2f} "
                  f"{best['reward_per_step']:8.3f} | {full['accuracy']*100:17.2f}% "
                  f"{full['tracked_accuracy']*100:7.2f}% {full['cells_per_tracked_step']:5.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"standard": True, "nrows": args.nrows, "ncols": args.ncols,
                   "matrices": args.matrices, "matrix_seed": args.matrix_seed,
                   "results": results}, fh, indent=1)
    print(f"\nWrote {args.out}")


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
    p.add_argument("--moore-moves", action="store_true",
                   help="Matrices move diagonally (O<i> tokens)")
    p.add_argument("--costs", default="0.16,0.04")
    p.add_argument("--standard", action="store_true",
                   help="standard.md oracle: infinite horizon, per-cell threshold "
                        "decisions, upstream + tracked metrics (grid only)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.standard:
        return main_standard(args)

    results = []
    print(f"{'env':>4} {'mat':>3} {'k':>2} {'cost':>5} | {'best-reward m':>13} "
          f"{'acc':>7} {'cells':>5} | {'full-budget acc':>15}")
    for env in args.envs.split(","):
        names, mats = parse_matrix_spec(args.matrices, env, args.matrix_seed,
                                        nrows=args.nrows, ncols=args.ncols)
        for name, T in zip(names, mats):
            for k in (int(x) for x in args.ks.split(",")):
                for cost in (float(c) for c in args.costs.split(",")):
                    rows = [run_oracle(env, T, k, cost, m, args.episodes, 3,
                                       args.nrows, args.ncols, args.moore_moves)
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
