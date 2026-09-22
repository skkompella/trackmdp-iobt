#!/usr/bin/env python
"""
train_switching_grid_ppo.py — PPO on a 5x5 grid whose object switches between
two opposed transition matrices, with optional change detection and restart.

The PPO counterpart to train_switching_grid_online.py, on the identical
schedule and the identical environment-step budget, so the two learners are
directly comparable.

Two asymmetries that are inherent to PPO and are reported rather than hidden:

1. DETECTION GRANULARITY.  The tabular learner sees every step and can feed the
   detector per step.  PPO's rollouts live in worker processes, so there is no
   per-step hook; instead one greedy evaluation episode runs after each training
   iteration and its 100 per-step hit indicators go to the detector.  PPO
   therefore detects at batch resolution (512 env steps), not step resolution.

2. RESTART COST.  "Cold" rebuilds the policy network from scratch; "warm"
   restores a checkpoint pre-trained on the equal-weight matrix C.  Both are far
   more expensive than overwriting a Q-table, though negligible against a
   200-iteration segment.

A detail worth knowing about: the envs PPO trains on live in the workers as
pickled copies, so switching the regime on a local object does nothing.  The
switch is pushed to every remote env with env_runner_group.foreach_env(), and
this script VERIFIES it landed rather than assuming — a silent failure there
would mean PPO trained on one matrix all along and the whole arm was void.

Example
-------
    ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py \\
        --new-run 830 --no-detector
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray                                                     # noqa: E402
from ray.rllib.algorithms.ppo import PPO, PPOConfig            # noqa: E402

from src.core.change_detection import GLRChangeDetector        # noqa: E402
from src.core.grid_transition_matrices import (                # noqa: E402
    describe_matrix, make_matrix_A, make_matrix_B, make_matrix_C,
)
from src.core.grid_wrapper_rect import grid_environment_rect   # noqa: E402
from src.core.online_schedule import (                         # noqa: E402
    adaptation_latency, detector_scores,
)
from src.core.switching_grid_env import (                      # noqa: E402
    SwitchingGridLearner, SwitchingTransitionGridEnv,
)
from examples.finetune_deterministic import evaluate_policy    # noqa: E402

MATRIX_MAKERS = {"A": make_matrix_A, "B": make_matrix_B, "C": make_matrix_C}

PPO_HPARAMS = dict(lr=5e-4, train_batch_size=512, num_sgd_iter=20,
                   sgd_minibatch_size=128, clip_param=0.3, entropy_coeff=0.001,
                   vf_loss_coeff=1.0, grad_clip=10.0, num_workers=2,
                   rollout_fragment_length=64)


def build_algo(args, cfg, matrices):
    """Fresh PPO over the switching grid. Used at start and for cold restarts."""
    qobj = SwitchingGridLearner(
        run_number=args.new_run, nrows=args.nrows, ncols=args.ncols,
        max_sensors=args.max_sensors, max_sensors_null=args.max_sensors,
        time_limit=cfg["time_limit"], time_limit_max=cfg["time_limit"],
        matrices=matrices, seed=args.train_seed, no_moore_constraint=True,
        sensor_rew=args.sensor_rew)

    env_config = {"qobj": qobj,
                  "time_limit_schedule": [2000],
                  "time_limit_max": cfg["time_limit"],
                  "max_ep_steps": cfg["max_ep_steps"],
                  "no_moore_constraint": True}

    def _pick(pairs):
        accepted = set(inspect.signature(PPOConfig.training).parameters)
        return {k: v for k, v in pairs if k in accepted}

    tk = dict(gamma=0.99, lr=PPO_HPARAMS["lr"],
              train_batch_size=PPO_HPARAMS["train_batch_size"],
              entropy_coeff=PPO_HPARAMS["entropy_coeff"],
              vf_loss_coeff=PPO_HPARAMS["vf_loss_coeff"],
              grad_clip=PPO_HPARAMS["grad_clip"])
    tk.update(_pick([("num_sgd_iter", PPO_HPARAMS["num_sgd_iter"]),
                     ("num_epochs",   PPO_HPARAMS["num_sgd_iter"])]))
    tk.update(_pick([("sgd_minibatch_size", PPO_HPARAMS["sgd_minibatch_size"]),
                     ("minibatch_size",     PPO_HPARAMS["sgd_minibatch_size"])]))
    tk.update(_pick([("clip_param",   PPO_HPARAMS["clip_param"]),
                     ("clip_epsilon", PPO_HPARAMS["clip_param"])]))

    ppo_cfg = (PPOConfig()
               .environment(grid_environment_rect, env_config=env_config)
               .framework("torch").training(**tk)
               .env_runners(num_env_runners=PPO_HPARAMS["num_workers"],
                            rollout_fragment_length=PPO_HPARAMS["rollout_fragment_length"])
               .resources(num_gpus=0)
               .api_stack(enable_rl_module_and_learner=False,
                          enable_env_runner_and_connector_v2=False))
    ppo_cfg.normalize_actions = False
    return ppo_cfg.build()


def push_matrix(algo, idx):
    """
    Switch the regime inside every worker env, and verify it landed.

    The workers hold pickled COPIES of qobj, so mutating a local object would
    silently leave PPO training on the old matrix forever.
    """
    algo.env_runner_group.foreach_env(
        lambda e: e.qobj.grid_env.force_matrix(idx))
    seen = algo.env_runner_group.foreach_env(
        lambda e: e.qobj.grid_env.current_matrix_idx)
    flat = [v for sub in seen for v in (sub if isinstance(sub, list) else [sub])]
    flat = [v for v in flat if v is not None]
    if flat and any(v != idx for v in flat):
        raise RuntimeError(
            f"matrix switch did not reach every worker env: wanted {idx}, saw {flat}")
    return flat


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrows", type=int, default=5)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=6)
    p.add_argument("--sensor-rew", type=float, default=None)
    p.add_argument("--matrices", type=str, default="A,B")
    p.add_argument("--iterations-per-segment", type=int, default=200,
                   help="200 x 512 = 102,400 env steps, matching 1024 tabular "
                        "episodes of 100 steps")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--max-ep-steps", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=1,
                   help="Greedy episodes per iteration; also the detector's feed")
    # detection / restart
    p.add_argument("--no-detector", action="store_true")
    p.add_argument("--restart-strategy", type=str, default="cold",
                   choices=["cold", "warm"])
    p.add_argument("--prior-checkpoint", type=str, default=None,
                   help="Checkpoint for --restart-strategy warm")
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--detector-warmup-iters", type=int, default=10)
    p.add_argument("--detector-cooldown-iters", type=int, default=10)
    # bookkeeping
    p.add_argument("--latency-epsilon", type=float, default=0.05)
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--new-run", type=int, default=830)
    p.add_argument("--save-checkpoint", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    if args.restart_strategy == "warm" and not args.no_detector \
            and not args.prior_checkpoint:
        raise SystemExit("--restart-strategy warm requires --prior-checkpoint")

    names = [n.strip().upper() for n in args.matrices.split(",") if n.strip()]
    matrices = [MATRIX_MAKERS[n](args.nrows, args.ncols) for n in names]

    n_cells = args.nrows * args.ncols
    tl = args.time_limit
    cfg = {"time_limit": tl, "time_limit_max": tl,
           "missing_state": n_cells * (tl + 1) + 1,
           "max_ep_steps": args.max_ep_steps, "max_sensors": args.max_sensors,
           "max_sensors_null": args.max_sensors,
           "n_cells": n_cells, "no_moore_constraint": True,
           "clip_mode": "random", "clip_seed": args.eval_seed,
           "eval_episodes": args.eval_episodes}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": project_root}},
                 _node_ip_address="127.0.0.2")

    algo = build_algo(args, cfg, matrices)

    # Dedicated seeded eval env — never the one the workers mutate.
    eval_env = SwitchingTransitionGridEnv(
        args.nrows, args.ncols, args.max_sensors, args.max_sensors,
        cfg["missing_state"], tl, matrices, seed=args.eval_seed,
        no_moore_constraint=True, sensor_rew=args.sensor_rew)

    detector = None if args.no_detector else GLRChangeDetector(
        delta=args.delta, min_samples=args.min_samples)

    out_dir = args.out_dir or os.path.join(
        project_root, "experiments", "switching_grid", f"run{args.new_run}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 76)
    print("  TRACK-MDP — PPO ON A SWITCHING 5x5 GRID")
    print("=" * 76)
    for n, T in zip(names, matrices):
        print("  " + describe_matrix(T, args.nrows, args.ncols, f"matrix {n}"))
    print(f"\n  Grid              : {args.nrows}x{args.ncols} = {n_cells} cells")
    print(f"  max_sensors       : {args.max_sensors} of {n_cells}")
    print(f"  sensor_rew        : {eval_env.sensor_rew}")
    print(f"  Detector          : "
          f"{'DISABLED' if detector is None else f'GLR delta={args.delta} (batch resolution)'}")
    print(f"  Restart strategy  : {args.restart_strategy}")
    print(f"  Schedule          : {args.passes} passes x {len(names)} matrices "
          f"x {args.iterations_per_segment} iterations\n")

    history, segments = [], []
    fired_iters, switch_iters = [], []
    it = 0
    suppress_until = 0
    n_restarts = 0

    print(f"  {'pass':>4}  {'mat':>4}  {'acc':>7}  {'sensors':>8}  "
          f"{'latency':>8}  {'restarts':>8}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    for pass_idx in range(args.passes):
        for m_idx, m_name in enumerate(names):
            push_matrix(algo, m_idx)
            eval_env.force_matrix(m_idx)
            if it > 0:
                switch_iters.append(it)

            seg_acc, seg_sens = [], []
            for _ in range(args.iterations_per_segment):
                algo.train()
                m = evaluate_policy(algo, eval_env, cfg, return_detections=True)
                seg_acc.append(m["tracking_accuracy"])
                seg_sens.append(m["mean_sensors_used"])
                history.append({"iteration": it, "pass": pass_idx,
                                "matrix": m_name,
                                "accuracy": m["tracking_accuracy"],
                                "sensors": m["mean_sensors_used"],
                                "reward": m["mean_reward"]})

                quiet = (it < args.detector_warmup_iters or it < suppress_until)
                if detector is not None and not quiet:
                    for d in m["detections"]:
                        if detector.update(d):
                            fired_iters.append(it)
                            n_restarts += 1
                            if args.restart_strategy == "cold":
                                algo.stop()
                                algo = build_algo(args, cfg, matrices)
                            else:
                                algo.restore(args.prior_checkpoint)
                            push_matrix(algo, m_idx)
                            suppress_until = (it + 1
                                              + args.detector_cooldown_iters)
                            break
                elif detector is not None:
                    detector.reset()
                it += 1

            lat = adaptation_latency(seg_acc, args.latency_epsilon)
            segments.append({
                "pass": pass_idx, "matrix": m_name,
                "start_iteration": it - args.iterations_per_segment,
                "mean_accuracy": float(np.mean(seg_acc)),
                "final_accuracy": float(np.mean(seg_acc[-20:])),
                "mean_sensors": float(np.mean(seg_sens)),
                "adaptation_latency": lat})
            print(f"  {pass_idx:>4}  {m_name:>4}  {np.mean(seg_acc)*100:6.2f}%  "
                  f"{np.mean(seg_sens):8.2f}  "
                  f"{('n/a' if lat is None else lat):>8}  {n_restarts:>8}")

    lats = [s["adaptation_latency"] for s in segments
            if s["adaptation_latency"] is not None]

    def _mean_lat(rows):
        v = [r["adaptation_latency"] for r in rows
             if r["adaptation_latency"] is not None]
        return float(np.mean(v)) if v else None

    summary = {
        "run": args.new_run, "learner": "ppo",
        "grid": [args.nrows, args.ncols], "matrices": names,
        "time_limit": tl, "max_sensors": args.max_sensors,
        "sensor_rew": eval_env.sensor_rew,
        "detector_enabled": detector is not None,
        "restart_strategy": args.restart_strategy,
        "n_restarts": n_restarts,
        "iterations_per_segment": args.iterations_per_segment,
        "passes": args.passes,
        "env_steps": it * PPO_HPARAMS["train_batch_size"],
        "mean_accuracy": float(np.mean([s["mean_accuracy"] for s in segments])),
        "final_accuracy": float(np.mean([s["final_accuracy"] for s in segments])),
        "mean_sensors": float(np.mean([s["mean_sensors"] for s in segments])),
        "mean_latency_iterations": float(np.mean(lats)) if lats else None,
        "mean_latency_first_pass": _mean_lat([s for s in segments
                                              if s["pass"] == 0]),
        "mean_latency_later_passes": _mean_lat([s for s in segments
                                                if s["pass"] > 0]),
        "ceiling": (args.max_ep_steps - 1) / args.max_ep_steps,
        "detection_resolution": "one PPO batch (512 env steps)",
    }
    if detector is not None:
        summary["detector"] = detector_scores(
            fired_iters, switch_iters,
            tolerance=max(2, args.iterations_per_segment // 5))

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  Mean accuracy        : {summary['mean_accuracy']*100:.2f}%  "
          f"(final-20 {summary['final_accuracy']*100:.2f}%)")
    print(f"  Mean sensors/step    : {summary['mean_sensors']:.2f}")
    if lats:
        print(f"  Adaptation latency   : mean "
              f"{summary['mean_latency_iterations']:.1f} iterations "
              f"(pass 0 {summary['mean_latency_first_pass']}, "
              f"later {summary['mean_latency_later_passes']})")
    print(f"  Restarts             : {n_restarts}")
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
    if args.save_checkpoint:
        os.makedirs(args.save_checkpoint, exist_ok=True)
        algo.save(args.save_checkpoint)
        print(f"  Checkpoint saved     : {args.save_checkpoint}")

    algo.stop()
    print(f"\n  Wrote {out_dir}/online_summary.json and online_history.csv")


if __name__ == "__main__":
    main()
