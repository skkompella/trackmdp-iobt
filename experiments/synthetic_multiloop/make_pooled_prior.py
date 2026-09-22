#!/usr/bin/env python
"""
make_pooled_prior.py — train a Q-table across ALL loops, for the warm restart.

The `warm` restart strategy resets the learner to a prior trained over the whole
loop family rather than wiping it.  That prior has to come from somewhere, and
it must not be a schedule-trained table: training on loop 0, then loop 1, and so
on leaves the table biased toward whichever loop came last.  So here the env
picks a loop uniformly at random each episode (MultiLoopIoBTEnv's default, with
no force_loop pin), which is the pooled distribution the name implies.

Usage:
    python experiments/synthetic_multiloop/make_pooled_prior.py \
        --max-sensors 2 --episodes 5000 --out prior_k2.npy
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from src.core.iobt_loops import sample_loops              # noqa: E402
from src.core.multiloop_iobt_env import MultiLoopIoBTEnv  # noqa: E402
from src.core.tabular_td import IOBT_N, TabularTDAgent    # noqa: E402


def _run_episode():
    """Borrow the rollout from the online runner so training matches exactly."""
    path = os.path.join(ROOT, "examples", "train_multiloop_online.py")
    spec = importlib.util.spec_from_file_location("_online", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-loops", type=int, default=10)
    p.add_argument("--loop-seed", type=int, default=20260921)
    p.add_argument("--time-limit", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=2)
    p.add_argument("--reward-preset", type=str, default="grid")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--max-ep-steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--optimistic-init", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    run_episode = _run_episode()
    loops = sample_loops(args.num_loops, seed=args.loop_seed)
    tl = args.time_limit
    missing = IOBT_N * IOBT_N * (tl + 1) + 1

    env = MultiLoopIoBTEnv(args.max_sensors, args.max_sensors, missing, tl,
                           loops, seed=args.seed, no_moore_constraint=True,
                           reward_preset=args.reward_preset)
    env.force_loop(None)          # uniform over the family — the whole point
    agent = TabularTDAgent(time_limit=tl, max_sensors=args.max_sensors,
                           missing_state=missing, alpha=args.alpha,
                           gamma=args.gamma, lam=args.lam, epsilon=args.epsilon,
                           optimistic_init=args.optimistic_init, seed=args.seed)
    cfg = {"time_limit": tl, "missing_state": missing,
           "max_ep_steps": args.max_ep_steps, "max_sensors": args.max_sensors}

    print(f"Pooled prior: {args.episodes} episodes, loops sampled uniformly, "
          f"max_sensors={args.max_sensors}, tl={tl}")
    for i in range(args.episodes):
        run_episode(agent, env, cfg)
        if (i + 1) % max(1, args.episodes // 5) == 0:
            acc = np.mean([run_episode(agent, env, cfg, learn=False,
                                       explore=False)["accuracy"]
                           for _ in range(20)])
            print(f"  {i+1:6d} episodes  pooled greedy accuracy {acc*100:6.2f}%")

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    np.save(out, agent.q)
    print(f"Saved {out}  shape={agent.q.shape}")


if __name__ == "__main__":
    main()
