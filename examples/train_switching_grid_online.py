#!/usr/bin/env python
"""
train_switching_grid_online.py — online tabular TD on a 5x5 grid whose object
switches between two opposed transition matrices.

The object drifts north-east under matrix A and south-west under matrix B, on
the schedule A,B,A,B,... A tabular SARSA(lambda) learner updates every step,
and a GLR change detector watches the per-step tracking-success stream; when it
fires, the learner is restarted per --restart-strategy.

Why this experiment exists: the multi-loop version of it produced a NULL result
(experiments/synthetic_multiloop/online_results.md) — restart-on-change hurt,
because all ten loops were paths through one graph and so shared transition
structure that a restart discarded. Two opposed drift matrices do not share
that way (per-row total variation between A and B averages 0.66), so this is
the case where detection should finally pay.

Action representation is FACTORED: one weight per cell rather than one per
sensor subset. At max_sensors=6 over 25 cells the subset table would be
51 x 245,505 = 12.5M entries and would never converge; factored is 51 x 25 =
1,275. See src/core/tabular_td.py for why that is a good fit here.

Example
-------
    ./track_mdp_env/bin/python examples/train_switching_grid_online.py \\
        --new-run 820 --no-detector
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.change_detection import (                        # noqa: E402
    GLRChangeDetector, RestartController,
)
from src.core.grid_transition_matrices import (                # noqa: E402
    describe_matrix, make_matrix_A, make_matrix_B, make_matrix_C,
)
from src.core.online_schedule import (                         # noqa: E402
    adaptation_latency, detector_scores, run_episode,
)
from src.core.switching_grid_env import SwitchingTransitionGridEnv  # noqa: E402
from src.core.tabular_td import TabularTDAgent                 # noqa: E402

MATRIX_MAKERS = {"A": make_matrix_A, "B": make_matrix_B, "C": make_matrix_C}


def build_agent(args, cfg, seed=None):
    return TabularTDAgent(
        time_limit=cfg["time_limit"], max_sensors=args.max_sensors,
        missing_state=cfg["missing_state"], alpha=args.alpha, gamma=args.gamma,
        lam=args.lam, epsilon=args.epsilon,
        optimistic_init=args.optimistic_init,
        seed=args.train_seed if seed is None else seed,
        n_nodes=cfg["n_cells"], grid_width=cfg["n_cells"],
        action_mode="factored")


def build_env(args, cfg, matrices, seed=None):
    return SwitchingTransitionGridEnv(
        nrows=args.nrows, ncols=args.ncols,
        max_sensors=args.max_sensors, max_sensors_null=args.max_sensors,
        missing_state=cfg["missing_state"], time_limit=cfg["time_limit"],
        matrices=matrices, seed=args.train_seed if seed is None else seed,
        no_moore_constraint=True, sensor_rew=args.sensor_rew)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrows", type=int, default=5)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=6)
    p.add_argument("--sensor-rew", type=float, default=None)
    p.add_argument("--matrices", type=str, default="A,B",
                   help="Comma-separated regimes to cycle through")
    # schedule
    p.add_argument("--episodes-per-segment", type=int, default=1024,
                   help="1024 x 100 steps = 102,400 env steps, matching 200 "
                        "PPO iterations at batch 512")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--max-ep-steps", type=int, default=100)
    # learner
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--lam", type=float, default=0.0,
                   help="Keep at 0 for factored mode: traces corrupt "
                        "per-cell credit assignment (measured 4.95%% vs "
                        "84.35%% accuracy at lam=0.8 vs 0.0)")
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--optimistic-init", type=float, default=5.0)
    # detector
    p.add_argument("--no-detector", action="store_true",
                   help="Baseline: pure continual learning, never restart")
    p.add_argument("--oracle-restart", action="store_true",
                   help="Control: restart at the KNOWN switch points, no "
                        "detector. Separates 'restarting is harmful' from "
                        "'the detector misfires'")
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--detector-warmup-episodes", type=int, default=5)
    p.add_argument("--detector-cooldown-episodes", type=int, default=5)
    p.add_argument("--restart-strategy", type=str, default="cold",
                   choices=list(RestartController.STRATEGIES))
    p.add_argument("--prior-path", type=str, default=None,
                   help="Q-weights .npy for --restart-strategy warm")
    p.add_argument("--save-q", type=str, default=None)
    # bookkeeping
    p.add_argument("--latency-epsilon", type=float, default=0.05)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--new-run", type=int, default=820)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    names = [n.strip().upper() for n in args.matrices.split(",") if n.strip()]
    matrices = [MATRIX_MAKERS[n](args.nrows, args.ncols) for n in names]

    n_cells = args.nrows * args.ncols
    tl = args.time_limit
    cfg = {"time_limit": tl,
           "missing_state": n_cells * (tl + 1) + 1,
           "max_ep_steps": args.max_ep_steps,
           "max_sensors": args.max_sensors,
           "n_cells": n_cells}

    env   = build_env(args, cfg, matrices)
    agent = build_agent(args, cfg)

    prior = np.load(args.prior_path) if args.prior_path else None
    controller = RestartController(agent, strategy=args.restart_strategy,
                                   prior=prior)
    detector = None if (args.no_detector or args.oracle_restart) else \
        GLRChangeDetector(delta=args.delta, min_samples=args.min_samples)

    out_dir = args.out_dir or os.path.join(
        project_root, "experiments", "switching_grid", f"run{args.new_run}")
    os.makedirs(out_dir, exist_ok=True)

    if args.oracle_restart:
        det_label = "ORACLE (restart at known switches)"
    elif detector is None:
        det_label = "DISABLED (continual-learning baseline)"
    else:
        det_label = f"GLR delta={args.delta}, min_samples={args.min_samples}"

    print("=" * 76)
    print("  TRACK-MDP — ONLINE TABULAR TD ON A SWITCHING 5x5 GRID")
    print("=" * 76)
    for n, T in zip(names, matrices):
        print("  " + describe_matrix(T, args.nrows, args.ncols, f"matrix {n}"))
    print(f"\n  Grid              : {args.nrows}x{args.ncols} = {n_cells} cells")
    print(f"  Q-weights         : {agent.q.shape[0]} states x "
          f"{agent.q.shape[1]} cells = {agent.q.size:,} (factored)")
    print(f"  time_limit        : {tl}")
    print(f"  max_sensors       : {args.max_sensors} of {n_cells}")
    print(f"  sensor_rew        : {env.sensor_rew}")
    print(f"  SARSA(lambda)     : alpha={args.alpha} gamma={args.gamma} "
          f"lambda={args.lam} eps={args.epsilon}")
    print(f"  Detector          : {det_label}")
    print(f"  Restart strategy  : {args.restart_strategy}")
    print(f"  Schedule          : {args.passes} passes x {len(names)} matrices "
          f"x {args.episodes_per_segment} episodes\n")

    history, segments = [], []
    fired_episodes, switch_episodes = [], []
    ep = 0
    suppress_until = 0

    print(f"  {'pass':>4}  {'mat':>4}  {'acc':>7}  {'sensors':>8}  "
          f"{'latency':>8}  {'restarts':>8}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    for pass_idx in range(args.passes):
        for m_idx, m_name in enumerate(names):
            env.force_matrix(m_idx)
            if ep > 0:
                switch_episodes.append(ep)
                if args.oracle_restart:
                    controller.restart(segment_reward=float(
                        segments[-1]["mean_accuracy"]) if segments else 0.0)
                    fired_episodes.append(ep)

            seg_acc, seg_sens = [], []
            for _ in range(args.episodes_per_segment):
                res = run_episode(agent, env, cfg, learn=True, explore=True)
                seg_acc.append(res["accuracy"])
                seg_sens.append(res["sensors"])
                history.append({"episode": ep, "pass": pass_idx,
                                "matrix": m_name, "accuracy": res["accuracy"],
                                "sensors": res["sensors"],
                                "reward": res["reward"]})

                quiet = (ep < args.detector_warmup_episodes
                         or ep < suppress_until)
                if detector is not None and not quiet:
                    for d in res["detections"]:
                        if detector.update(d):
                            fired_episodes.append(ep)
                            controller.restart(
                                segment_reward=float(np.mean(seg_acc)))
                            suppress_until = (ep + 1
                                              + args.detector_cooldown_episodes)
                            break
                elif detector is not None:
                    detector.reset()
                ep += 1

            lat = adaptation_latency(seg_acc, args.latency_epsilon)
            segments.append({
                "pass": pass_idx, "matrix": m_name,
                "start_episode": ep - args.episodes_per_segment,
                "mean_accuracy": float(np.mean(seg_acc)),
                "final_accuracy": float(np.mean(seg_acc[-100:])),
                "mean_sensors": float(np.mean(seg_sens)),
                "adaptation_latency": lat,
            })
            print(f"  {pass_idx:>4}  {m_name:>4}  {np.mean(seg_acc)*100:6.2f}%  "
                  f"{np.mean(seg_sens):8.2f}  "
                  f"{('n/a' if lat is None else lat):>8}  "
                  f"{controller.n_restarts:>8}")

    lats = [s["adaptation_latency"] for s in segments
            if s["adaptation_latency"] is not None]

    def _mean_lat(rows):
        v = [r["adaptation_latency"] for r in rows
             if r["adaptation_latency"] is not None]
        return float(np.mean(v)) if v else None

    summary = {
        "run": args.new_run, "learner": "tabular",
        "grid": [args.nrows, args.ncols], "matrices": names,
        "time_limit": tl, "max_sensors": args.max_sensors,
        "sensor_rew": env.sensor_rew,
        "detector_enabled": detector is not None,
        "oracle_restart": bool(args.oracle_restart),
        "restart_strategy": args.restart_strategy,
        "n_restarts": controller.n_restarts,
        "episodes_per_segment": args.episodes_per_segment,
        "passes": args.passes,
        "env_steps": ep * args.max_ep_steps,
        "q_entries": int(agent.q.size),
        "mean_accuracy": float(np.mean([s["mean_accuracy"] for s in segments])),
        "final_accuracy": float(np.mean([s["final_accuracy"] for s in segments])),
        "mean_sensors": float(np.mean([s["mean_sensors"] for s in segments])),
        "mean_latency_episodes": float(np.mean(lats)) if lats else None,
        "mean_latency_first_pass": _mean_lat([s for s in segments
                                              if s["pass"] == 0]),
        "mean_latency_later_passes": _mean_lat([s for s in segments
                                                if s["pass"] > 0]),
        "ceiling": (args.max_ep_steps - 1) / args.max_ep_steps,
    }
    if detector is not None or args.oracle_restart:
        summary["detector"] = detector_scores(
            fired_episodes, switch_episodes,
            tolerance=max(2, args.episodes_per_segment // 5))

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  Mean accuracy        : {summary['mean_accuracy']*100:.2f}%  "
          f"(final-100 {summary['final_accuracy']*100:.2f}%, "
          f"ceiling {summary['ceiling']*100:.2f}%)")
    print(f"  Mean sensors/step    : {summary['mean_sensors']:.2f}")
    if lats:
        print(f"  Adaptation latency   : mean "
              f"{summary['mean_latency_episodes']:.1f} episodes "
              f"(pass 0 {summary['mean_latency_first_pass']}, "
              f"later {summary['mean_latency_later_passes']})")
    print(f"  Restarts             : {controller.n_restarts}")
    if "detector" in summary:
        d = summary["detector"]
        print(f"  Detector             : {d['true_positive']}/{d['n_switches']} "
              f"switches caught, {d['false_alarms']} false alarms")

    with open(os.path.join(out_dir, "online_summary.json"), "w") as fh:
        json.dump({"summary": summary, "segments": segments}, fh, indent=2)
    with open(os.path.join(out_dir, "online_history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    if args.save_q:
        np.save(args.save_q, agent.q)
        print(f"  Q-weights saved      : {args.save_q}")

    print(f"\n  Wrote {out_dir}/online_summary.json and online_history.csv")


if __name__ == "__main__":
    main()
