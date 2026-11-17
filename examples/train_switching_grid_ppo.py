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
from src.core.gym_wrapper import grid_environment              # noqa: E402
from src.core.matrix_iobt_env import (                         # noqa: E402
    MatrixIoBTEnv, MatrixIoBTLearner,
)
from src.core.random_transition_matrices import (              # noqa: E402
    describe_random, parse_matrix_spec,
)
from src.core.switching_grid_env import (                      # noqa: E402
    SwitchingGridLearner, SwitchingTransitionGridEnv,
)
from examples.finetune_deterministic import evaluate_policy    # noqa: E402

MATRIX_MAKERS = {"A": make_matrix_A, "B": make_matrix_B, "C": make_matrix_C}

PPO_PRESETS = {
    # the switching experiments so far
    "switching": dict(lr=5e-4, train_batch_size=512, num_sgd_iter=20,
                      sgd_minibatch_size=128, clip_param=0.3, entropy_coeff=0.001,
                      vf_loss_coeff=1.0, grad_clip=10.0, num_workers=2,
                      rollout_fragment_length=64),
    # examples/training_grid_rect.py (the RLlib port): RLlib defaults except
    # lr 1e-4, grad_clip 30, 4 workers
    "port": dict(lr=1e-4, train_batch_size=4000, num_sgd_iter=30,
                 sgd_minibatch_size=128, clip_param=0.3, entropy_coeff=0.0,
                 vf_loss_coeff=1.0, grad_clip=30.0, num_workers=4,
                 rollout_fragment_length="auto"),
}
PPO_HPARAMS = PPO_PRESETS["switching"]   # replaced in main() by --ppo-preset


# Arguments that may differ between an original run and its --resume.
_RESUME_FREE_ARGS = {"resume", "checkpoint_dir", "checkpoint_every", "stop_after"}


def save_resume_state(ckpt_dir, algo, args, state):
    """
    Write algo checkpoint + loop state to a fresh step directory, then point
    LATEST at it with an atomic rename, so a kill mid-save never leaves a
    half-written checkpoint as the latest one.  Older step directories go.
    """
    import pickle
    import shutil
    os.makedirs(ckpt_dir, exist_ok=True)
    step_dir = os.path.join(ckpt_dir, f"step_{state['it']:06d}")
    if os.path.isdir(step_dir):
        shutil.rmtree(step_dir)
    algo.save(os.path.join(step_dir, "algo"))
    state = dict(state, args={k: v for k, v in vars(args).items()
                              if k not in _RESUME_FREE_ARGS})
    with open(os.path.join(step_dir, "state.pkl"), "wb") as fh:
        pickle.dump(state, fh)
    tmp = os.path.join(ckpt_dir, "LATEST.tmp")
    with open(tmp, "w") as fh:
        fh.write(os.path.basename(step_dir))
    os.replace(tmp, os.path.join(ckpt_dir, "LATEST"))
    for d in os.listdir(ckpt_dir):
        if d.startswith("step_") and d != os.path.basename(step_dir):
            shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)


def load_resume_state(ckpt_dir, args):
    import pickle
    latest = os.path.join(ckpt_dir, "LATEST")
    if not os.path.exists(latest):
        raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
    step_dir = os.path.join(ckpt_dir, open(latest).read().strip())
    with open(os.path.join(step_dir, "state.pkl"), "rb") as fh:
        state = pickle.load(fh)
    now = {k: v for k, v in vars(args).items() if k not in _RESUME_FREE_ARGS}
    diff = {k: (state["args"].get(k), v) for k, v in now.items()
            if state["args"].get(k) != v}
    if diff:
        raise SystemExit(f"--resume: arguments differ from the original run: {diff}")
    state["algo_path"] = os.path.join(step_dir, "algo")
    return state


def load_policy_weights(checkpoint_dir):
    """Weights of an RLlib checkpoint's default policy, for set_weights()."""
    from ray.rllib.policy.policy import Policy
    pol = Policy.from_checkpoint(
        os.path.join(checkpoint_dir, "policies", "default_policy"))
    weights = {"default_policy": pol.get_weights()}
    del pol
    return weights


def build_algo(args, cfg, matrices):
    """Fresh PPO over the switching env. Used at start and for cold restarts."""
    if args.env == "iobt":
        qobj = MatrixIoBTLearner(
            args.new_run, 6, args.max_sensors, args.max_sensors,
            cfg["time_limit"], cfg["time_limit"], matrices,
            seed=args.train_seed, no_moore_constraint=not args.moore,
            reward_preset="grid", sensor_rew=args.sensor_rew)
        wrapper = grid_environment
    else:
        qobj = SwitchingGridLearner(
            run_number=args.new_run, nrows=args.nrows, ncols=args.ncols,
            max_sensors=args.max_sensors, max_sensors_null=args.max_sensors,
            time_limit=cfg["time_limit"], time_limit_max=cfg["time_limit"],
            matrices=matrices, seed=args.train_seed, no_moore_constraint=not args.moore,
            sensor_rew=args.sensor_rew, moore_moves=args.moore_moves,
            exit_prob=args.exit_prob)
        wrapper = grid_environment_rect

    env_config = {"qobj": qobj,
                  "time_limit_schedule": [2000],
                  "time_limit_max": cfg["time_limit"],
                  "max_ep_steps": args.max_ep_steps,    # 0 = no cap
                  "no_moore_constraint": not args.moore}

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
               .environment(wrapper, env_config=env_config)
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
    p.add_argument("--env", choices=["grid", "iobt"], default="grid",
                   help="grid: 5x5 SwitchingTransitionGridEnv. iobt: the 10-node "
                        "Camp Buckner graph with a matrix-driven object")
    p.add_argument("--moore", action="store_true",
                   help="Track-MDP's local action window ((2t+3)^2 cells around the "
                        "last-seen cell) instead of one action per grid cell. Needed "
                        "for large grids: on 10x10 the full 100-cell action space "
                        "never learned (flat ~2.5% over 1200 iterations)")
    p.add_argument("--matrix-seed", type=int, default=20261006,
                   help="Seed base for random matrices R<i> in --matrices")
    p.add_argument("--concentration", type=float, default=1.0)
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
    p.add_argument("--ppo-preset", choices=sorted(PPO_PRESETS), default="switching",
                   help="PPO hyperparameters: 'switching' (these experiments) or "
                        "'port' (examples/training_grid_rect.py)")
    p.add_argument("--init-checkpoint", type=str, default=None,
                   help="Start from this checkpoint's policy weights instead of scratch")
    p.add_argument("--moore-moves", action="store_true",
                   help="Matrices may move diagonally (needed for O<i> matrices)")
    p.add_argument("--exit-prob", type=float, default=0.0,
                   help="Per-step chance the target exits, ending the episode "
                        "(original Track-MDP: 0.005)")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Save a resumable checkpoint every N iterations and at every "
                        "segment end (0 = only at segment ends)")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Where resumable checkpoints go (default: <out-dir>/resume_ckpt)")
    p.add_argument("--stop-after", type=int, default=0,
                   help="Checkpoint and exit once this many total iterations are done "
                        "(0 = run to the end); continue later with --resume")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the latest checkpoint in --checkpoint-dir; pass "
                        "the same arguments as the original run")
    p.add_argument("--save-stream", type=str, default=None,
                   help="Save the per-iteration eval hit/miss stream the detector "
                        "would see (.npz), for offline delta tuning")
    p.add_argument("--oracle-restart", action="store_true",
                   help="Warm-restart exactly at the true switches (no detector); "
                        "records the stream a perfect detector would leave behind")
    args = p.parse_args()
    global PPO_HPARAMS
    PPO_HPARAMS = PPO_PRESETS[args.ppo_preset]

    if args.oracle_restart and not (args.no_detector and args.prior_checkpoint
                                    and args.restart_strategy == "warm"):
        raise SystemExit("--oracle-restart needs --no-detector, --restart-strategy "
                         "warm and --prior-checkpoint")
    if args.restart_strategy == "warm" and not args.no_detector \
            and not args.prior_checkpoint:
        raise SystemExit("--restart-strategy warm requires --prior-checkpoint")

    names, matrices = parse_matrix_spec(args.matrices, args.env, args.matrix_seed,
                                        args.concentration, args.nrows, args.ncols)

    # IoBT observations and actions live on the 4x4 backing grid (16 cells).
    n_cells = args.nrows * args.ncols if args.env == "grid" else 16
    tl = args.time_limit
    cfg = {"time_limit": tl, "time_limit_max": tl,
           "missing_state": n_cells * (tl + 1) + 1,
           # eval episodes need a finite cap; with exits on, 1000 like the port
           "max_ep_steps": args.max_ep_steps or 1000, "max_sensors": args.max_sensors,
           "max_sensors_null": args.max_sensors,
           "n_cells": n_cells, "no_moore_constraint": not args.moore,
           "clip_mode": "random", "clip_seed": args.eval_seed,
           "eval_episodes": args.eval_episodes}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": project_root}},
                 _node_ip_address="127.0.0.2")

    out_dir = args.out_dir or os.path.join(
        project_root, "experiments", "switching_grid", f"run{args.new_run}")
    os.makedirs(out_dir, exist_ok=True)
    # absolute: RLlib's restore() treats a relative path as a URI and fails
    ckpt_dir = os.path.abspath(args.checkpoint_dir or os.path.join(out_dir, "resume_ckpt"))
    resume_state = load_resume_state(ckpt_dir, args) if args.resume else None

    algo = build_algo(args, cfg, matrices)
    if resume_state is not None:
        # Restoring before any sampling is safe (no worker is mid-episode);
        # restores weights AND optimizer state.
        algo.restore(resume_state["algo_path"])
        algo.env_runner_group.sync_weights()
    elif args.init_checkpoint:
        algo.set_weights(load_policy_weights(args.init_checkpoint))
        algo.env_runner_group.sync_weights()

    prior_weights = None
    if args.restart_strategy == "warm" and args.prior_checkpoint:
        from ray.rllib.policy.policy import Policy
        pol = Policy.from_checkpoint(
            os.path.join(args.prior_checkpoint, "policies", "default_policy"))
        prior_weights = {"default_policy": pol.get_weights()}
        del pol

    # Dedicated seeded eval env — never the one the workers mutate.
    if args.env == "iobt":
        eval_env = MatrixIoBTEnv(
            args.max_sensors, args.max_sensors, cfg["missing_state"], tl,
            matrices, seed=args.eval_seed, no_moore_constraint=not args.moore,
            reward_preset="grid", sensor_rew=args.sensor_rew)
    else:
        eval_env = SwitchingTransitionGridEnv(
            args.nrows, args.ncols, args.max_sensors, args.max_sensors,
            cfg["missing_state"], tl, matrices, seed=args.eval_seed,
            no_moore_constraint=not args.moore, sensor_rew=args.sensor_rew,
        moore_moves=args.moore_moves, exit_prob=args.exit_prob)

    detector = None if args.no_detector else GLRChangeDetector(
        delta=args.delta, min_samples=args.min_samples)

    print("=" * 76)
    print("  TRACK-MDP — PPO ON A SWITCHING 5x5 GRID")
    print("=" * 76)
    for n, T in zip(names, matrices):
        print("  " + describe_random(T, n))
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
    stream = []                     # one array of eval hits per iteration
    it = 0
    suppress_until = 0
    n_restarts = 0
    seg_acc, seg_sens, seg_cells = [], [], []
    start_seg, start_inner = 0, 0
    if resume_state is not None:
        st = resume_state
        (history, segments, fired_iters, switch_iters, stream, it, suppress_until,
         n_restarts, seg_acc, seg_sens, seg_cells, start_seg, start_inner) = (
            st[k] for k in ("history", "segments", "fired_iters", "switch_iters",
                            "stream", "it", "suppress_until", "n_restarts",
                            "seg_acc", "seg_sens", "seg_cells", "seg_idx", "inner"))
        if st["detector"] is not None:
            detector = st["detector"]
        eval_env = st["eval_env"]
        print(f"  Resumed at iteration {it} (segment {start_seg}, "
              f"iteration {start_inner} of it) from {resume_state['algo_path']}\n")

    def checkpoint(seg_idx, inner):
        save_resume_state(ckpt_dir, algo, args, dict(
            history=history, segments=segments, fired_iters=fired_iters,
            switch_iters=switch_iters, stream=stream, it=it,
            suppress_until=suppress_until, n_restarts=n_restarts,
            seg_acc=seg_acc, seg_sens=seg_sens, seg_cells=seg_cells,
            seg_idx=seg_idx, inner=inner, detector=detector, eval_env=eval_env))

    print(f"  {'pass':>4}  {'mat':>4}  {'acc':>7}  {'sensors':>8}  "
          f"{'latency':>8}  {'restarts':>8}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    schedule = [(p_i, m_i, m_n) for p_i in range(args.passes)
                for m_i, m_n in enumerate(names)]
    for seg_idx in range(start_seg, len(schedule)):
        pass_idx, m_idx, m_name = schedule[seg_idx]
        mid_segment = seg_idx == start_seg and start_inner > 0
        push_matrix(algo, m_idx)          # always: resumed workers start fresh
        eval_env.force_matrix(m_idx)
        if not mid_segment:
            if it > 0:
                switch_iters.append(it)
                if args.oracle_restart:
                    algo.set_weights(prior_weights)
                    algo.env_runner_group.sync_weights()
                    n_restarts += 1
            seg_acc, seg_sens, seg_cells = [], [], []
        for inner in range(start_inner if mid_segment else 0,
                           args.iterations_per_segment):
            algo.train()
            m = evaluate_policy(algo, eval_env, cfg, return_detections=True)
            stream.append(np.asarray(m["detections"], dtype=np.int8))
            seg_acc.append(m["tracking_accuracy"])
            seg_sens.append(m["mean_sensors_used"])
            if m["active_tracked"]:
                seg_cells.append(float(np.mean(m["active_tracked"])))
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
                            # Set the base model's WEIGHTS in place rather
                            # than algo.restore(): restore swaps the policy
                            # objects under workers that are mid-episode,
                            # and their half-built batches then mismatch
                            # (measured: "operands could not be broadcast
                            # together with shapes (64,) (63,)" in GAE).
                            algo.set_weights(prior_weights)
                            algo.env_runner_group.sync_weights()
                        push_matrix(algo, m_idx)
                        suppress_until = (it + 1
                                          + args.detector_cooldown_iters)
                        break
            elif detector is not None:
                detector.reset()
            it += 1
            if args.stop_after and it >= args.stop_after \
                    and inner + 1 < args.iterations_per_segment:
                checkpoint(seg_idx, inner + 1)
                algo.stop()
                print(f"\n  Stopped after iteration {it}; continue with --resume")
                return
            if (args.checkpoint_every and (inner + 1) % args.checkpoint_every == 0
                    and inner + 1 < args.iterations_per_segment):
                checkpoint(seg_idx, inner + 1)

        lat = adaptation_latency(seg_acc, args.latency_epsilon)
        segments.append({
            "pass": pass_idx, "matrix": m_name,
            "start_iteration": it - args.iterations_per_segment,
            "mean_accuracy": float(np.mean(seg_acc)),
            "final_accuracy": float(np.mean(seg_acc[-20:])),
            "mean_sensors": float(np.mean(seg_sens)),
            "final_sensors": float(np.mean(seg_sens[-20:])),
            # cells switched on per TRACKED step (missing-state rescans excluded)
            "final_cells_per_tracked_step": (
                float(np.mean(seg_cells[-20:])) if seg_cells else None),
            "adaptation_latency": lat})
        print(f"  {pass_idx:>4}  {m_name:>4}  {np.mean(seg_acc)*100:6.2f}%  "
              f"{np.mean(seg_sens):8.2f}  "
              f"{('n/a' if lat is None else lat):>8}  {n_restarts:>8}")
        checkpoint(seg_idx + 1, 0)
        if args.stop_after and it >= args.stop_after and seg_idx + 1 < len(schedule):
            algo.stop()
            print(f"\n  Stopped after iteration {it}; continue with --resume")
            return

    lats = [s["adaptation_latency"] for s in segments
            if s["adaptation_latency"] is not None]

    def _mean_lat(rows):
        v = [r["adaptation_latency"] for r in rows
             if r["adaptation_latency"] is not None]
        return float(np.mean(v)) if v else None

    summary = {
        "run": args.new_run, "learner": "ppo", "env": args.env,
        "moore": bool(args.moore), "grid_shape": [args.nrows, args.ncols],
        "matrix_seed": args.matrix_seed,
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
        "final_sensors": float(np.mean([s["final_sensors"] for s in segments])),
        "final_cells_per_tracked_step": float(np.mean(
            [s["final_cells_per_tracked_step"] for s in segments
             if s["final_cells_per_tracked_step"] is not None] or [np.nan])),
        "mean_latency_iterations": float(np.mean(lats)) if lats else None,
        "mean_latency_first_pass": _mean_lat([s for s in segments
                                              if s["pass"] == 0]),
        "mean_latency_later_passes": _mean_lat([s for s in segments
                                                if s["pass"] > 0]),
        "ceiling": (cfg["max_ep_steps"] - 1) / cfg["max_ep_steps"],
        "ppo_preset": args.ppo_preset, "init_checkpoint": args.init_checkpoint,
        "moore_moves": args.moore_moves, "exit_prob": args.exit_prob,
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
    print(f"  Mean sensors/step    : {summary['mean_sensors']:.2f}  "
          f"(final-20 {summary['final_sensors']:.2f}; cells per tracked step "
          f"{summary['final_cells_per_tracked_step']:.2f})")
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
    if args.save_stream:
        np.savez_compressed(
            args.save_stream,
            hits=np.concatenate(stream),
            iter_len=np.array([len(x) for x in stream]),
            switch_iters=np.array(switch_iters),
            oracle_restart=args.oracle_restart,
            iterations_per_segment=args.iterations_per_segment)
        print(f"  Stream saved         : {args.save_stream}")
    if args.save_checkpoint:
        os.makedirs(args.save_checkpoint, exist_ok=True)
        algo.save(args.save_checkpoint)
        print(f"  Checkpoint saved     : {args.save_checkpoint}")

    algo.stop()
    print(f"\n  Wrote {out_dir}/online_summary.json and online_history.csv")


if __name__ == "__main__":
    main()
