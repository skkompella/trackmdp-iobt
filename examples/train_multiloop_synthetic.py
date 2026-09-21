#!/usr/bin/env python
"""
train_multiloop_synthetic.py — train ONE tracking policy across N synthetic loops.

The object walks a randomly chosen loop each episode (see
src/core/multiloop_iobt_env.py).  Detection is ideal binary and no audio,
camera or GPS is involved: this isolates the multi-loop tracking problem from
sensor noise.

The agent never observes which loop is active, so it must infer the trajectory
from its own tracking state.  That is what the per-loop evaluation at the end
measures: whether one policy covers every loop, or trades some off against
others.

Training stops on a plateau rather than a fixed budget — see --patience.

Evaluation deliberately runs on a DEDICATED env instance with a fixed seed,
never the env the training rollouts mutate.  Sharing them makes the per-
iteration score depend on training state, which turns "best iteration" into a
lottery; with a random loop per episode on top, that noise would be worse here
than in the live-data pipeline.

Example
-------
    ./track_mdp_env/bin/python examples/train_multiloop_synthetic.py \
        --num-loops 10 --loop-seed 20260921 --new-run 800
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray                                                    # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig                # noqa: E402

from src.core.gym_wrapper import grid_environment             # noqa: E402
from src.core.iobt_loops import (                             # noqa: E402
    IOBT_NUM_NODES, describe_loops, sample_loops,
)
from src.core.multiloop_iobt_env import (                     # noqa: E402
    IOBT_N, MultiLoopIoBTEnv, MultiLoopIoBTLearner,
)

# Reuse the evaluator from the live pipeline so observation construction stays
# single-sourced: if we reimplemented it, it could drift from the gym wrapper
# and every number here would be quietly wrong.
from examples.finetune_deterministic import evaluate_policy   # noqa: E402


DEFAULTS = {
    "num_trans":               6,
    "max_sensors":             6,
    "max_sensors_null":        6,
    "lr":                      5e-4,
    "train_batch_size":        512,
    "num_sgd_iter":            20,
    "sgd_minibatch_size":      128,
    "clip_param":              0.3,
    "entropy_coeff":           0.001,
    "vf_loss_coeff":           1.0,
    "grad_clip":               10.0,
    "num_workers":             2,
    "rollout_fragment_length": 64,
    "eval_episodes":           20,
    "max_ep_steps":            100,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_eval_env(loops, cfg, seed, sensor_rew):
    """A fresh env for evaluation only — never the one training mutates."""
    return MultiLoopIoBTEnv(
        cfg["max_sensors"], cfg["max_sensors_null"],
        cfg["missing_state"], cfg["time_limit"],
        loops, seed=seed, no_moore_constraint=True, sensor_rew=sensor_rew,
    )


def evaluate_per_loop(algo, eval_env, cfg, episodes_per_loop):
    """
    Pin each loop in turn and evaluate on it.

    Returns (per_loop, aggregate).  The env RNG is re-seeded per loop so the
    same start positions are used for every checkpoint, making scores
    comparable across iterations and across runs.
    """
    per_loop = []
    loop_cfg = dict(cfg, eval_episodes=episodes_per_loop)

    for idx in range(eval_env.num_loops):
        eval_env._rng = np.random.default_rng(cfg["eval_seed"] + idx)
        eval_env.force_loop(idx)
        m = evaluate_policy(algo, eval_env, loop_cfg)
        per_loop.append({
            "loop":      idx,
            "nodes":     eval_env.loops[idx],
            "length":    len(eval_env.loops[idx]),
            "accuracy":  m["tracking_accuracy"],
            "sensors":   m["mean_sensors_used"],
            "reward":    m["mean_reward"],
        })

    eval_env.force_loop(None)

    accs = [r["accuracy"] for r in per_loop]
    sens = [r["sensors"] for r in per_loop]
    aggregate = {
        "mean_accuracy":  float(np.mean(accs)),
        "min_accuracy":   float(np.min(accs)),
        "max_accuracy":   float(np.max(accs)),
        "std_accuracy":   float(np.std(accs)),
        "mean_sensors":   float(np.mean(sens)),
        "worst_loop":     int(np.argmin(accs)),
    }
    return per_loop, aggregate


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_per_loop_table(per_loop, aggregate):
    lines = [
        "| Loop | Len | Nodes | Accuracy | Sensors/step |",
        "|------|-----|-------|----------|--------------|",
    ]
    for r in per_loop:
        nodes = "-".join(str(n) for n in r["nodes"])
        lines.append(
            f"| {r['loop']:>4} | {r['length']:>3} | {nodes} | "
            f"{r['accuracy']*100:7.2f}% | {r['sensors']:12.2f} |"
        )
    lines.append("")
    lines.append(
        f"**Mean {aggregate['mean_accuracy']*100:.2f}%** "
        f"(min {aggregate['min_accuracy']*100:.2f}% on loop "
        f"{aggregate['worst_loop']}, max {aggregate['max_accuracy']*100:.2f}%, "
        f"sd {aggregate['std_accuracy']*100:.2f}pp) "
        f"at {aggregate['mean_sensors']:.2f} sensors/step"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-loops",   type=int, default=10)
    p.add_argument("--loop-seed",   type=int, default=20260921,
                   help="Seed for sampling the loop set from all valid cycles")
    p.add_argument("--min-loop-len", type=int, default=3)
    p.add_argument("--max-loop-len", type=int, default=None)
    p.add_argument("--time-limit",  type=int, default=3,
                   help="Failed confirmations tolerated before the object is lost")
    p.add_argument("--max-sensors", type=int, default=6)
    p.add_argument("--sensor-rew",  type=float, default=-0.25,
                   help="Per-sensor energy penalty")
    p.add_argument("--new-run",     type=int, default=800)
    p.add_argument("--max-iterations", type=int, default=600,
                   help="Hard cap; training stops earlier on plateau")
    p.add_argument("--patience",    type=int, default=30,
                   help="Stop after this many evals with no meaningful gain")
    p.add_argument("--min-delta",   type=float, default=0.005,
                   help="Improvement in mean accuracy that resets patience (0.005 = 0.5pp)")
    p.add_argument("--eval-interval", type=int, default=5,
                   help="Evaluate every N training iterations")
    p.add_argument("--eval-episodes-per-loop", type=int, default=5,
                   help="Episodes per loop during training (kept cheap: this "
                        "runs every eval-interval iterations)")
    p.add_argument("--final-eval-episodes-per-loop", type=int, default=20,
                   help="Episodes per loop for the final report, run once on "
                        "the best checkpoint")
    p.add_argument("--eval-seed",   type=int, default=12345)
    p.add_argument("--train-seed",  type=int, default=7)
    p.add_argument("--out-dir",     type=str, default=None)
    args = p.parse_args()

    # ── Loop set ───────────────────────────────────────────────────────────
    loops = sample_loops(args.num_loops, seed=args.loop_seed,
                         min_len=args.min_loop_len, max_len=args.max_loop_len)
    print("=" * 70)
    print("  TRACK-MDP — SYNTHETIC MULTI-LOOP TRAINING")
    print("=" * 70)
    print(describe_loops(loops))

    time_limit = time_limit_max = args.time_limit
    cfg = dict(DEFAULTS)
    cfg.update({
        # n_cells is the BACKING GRID size (4x4 = 16), not the node count.  The
        # gym wrapper sizes the action space and augment vectors off qobj.N**2,
        # so evaluate_policy has to agree or the observation falls outside the
        # declared space.  See finetune_deterministic.py:1802.
        "n_cells":             IOBT_N * IOBT_N,
        "time_limit":          time_limit,
        "time_limit_max":      time_limit_max,
        "missing_state":       IOBT_N * IOBT_N * (time_limit_max + 1) + 1,
        "max_sensors":         args.max_sensors,
        "max_sensors_null":    args.max_sensors,
        "no_moore_constraint": True,
        "eval_seed":           args.eval_seed,
    })

    out_dir = args.out_dir or os.path.join(
        project_root, "experiments", "synthetic_multiloop", f"run{args.new_run}"
    )
    os.makedirs(out_dir, exist_ok=True)
    save_dir = os.path.join(project_root, "runs", f"agent_run{args.new_run}_ppo")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(out_dir, "loops.json"), "w") as fh:
        json.dump({
            "loop_seed":   args.loop_seed,
            "num_loops":   args.num_loops,
            "min_loop_len": args.min_loop_len,
            "max_loop_len": args.max_loop_len,
            "loops":       loops,
        }, fh, indent=2)

    # ── Env + PPO ──────────────────────────────────────────────────────────
    qobj = MultiLoopIoBTLearner(
        args.new_run, cfg["num_trans"], cfg["max_sensors"], cfg["max_sensors_null"],
        time_limit, time_limit_max, loops, seed=args.train_seed,
        no_moore_constraint=True, sensor_rew=args.sensor_rew,
    )

    env_config = {
        "qobj":                qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":      time_limit_max,
        "max_ep_steps":        cfg["max_ep_steps"],
        "no_moore_constraint": True,
    }

    if not ray.is_initialized():
        # "127.0.0.1" is special-cased by Ray and silently rewritten to the
        # external IP, which this sandbox blocks for self-connections.
        ray.init(ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": project_root}},
                 _node_ip_address="127.0.0.2")

    def _pick(pairs):
        """PPOConfig.training renamed several kwargs across RLlib versions."""
        accepted = set(inspect.signature(PPOConfig.training).parameters)
        return {k: v for k, v in pairs if k in accepted}

    training_kwargs = dict(
        gamma=0.99, lr=cfg["lr"], train_batch_size=cfg["train_batch_size"],
        entropy_coeff=cfg["entropy_coeff"], vf_loss_coeff=cfg["vf_loss_coeff"],
        grad_clip=cfg["grad_clip"],
    )
    training_kwargs.update(_pick([("num_sgd_iter", cfg["num_sgd_iter"]),
                                  ("num_epochs",   cfg["num_sgd_iter"])]))
    training_kwargs.update(_pick([("sgd_minibatch_size", cfg["sgd_minibatch_size"]),
                                  ("minibatch_size",     cfg["sgd_minibatch_size"])]))
    training_kwargs.update(_pick([("clip_param",  cfg["clip_param"]),
                                  ("clip_epsilon", cfg["clip_param"])]))

    ppo_cfg = (
        PPOConfig()
        .environment(grid_environment, env_config=env_config)
        .framework("torch")
        .training(**training_kwargs)
        .env_runners(num_env_runners=cfg["num_workers"],
                     rollout_fragment_length=cfg["rollout_fragment_length"])
        .resources(num_gpus=0)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
    )
    ppo_cfg.normalize_actions = False
    algo = ppo_cfg.build()

    eval_env = make_eval_env(loops, cfg, args.eval_seed, args.sensor_rew)

    print(f"\n  Loops             : {args.num_loops} (seed {args.loop_seed})")
    print(f"  time_limit        : {time_limit}")
    print(f"  max_sensors       : {cfg['max_sensors']} of {IOBT_NUM_NODES}")
    print(f"  sensor_rew        : {args.sensor_rew}")
    print(f"  Eval              : every {args.eval_interval} iters, "
          f"{args.eval_episodes_per_loop} eps/loop "
          f"(final: {args.final_eval_episodes_per_loop} eps/loop)")
    print(f"  Plateau stop      : patience {args.patience} evals, "
          f"min_delta {args.min_delta*100:.2f}pp, cap {args.max_iterations} iters")
    print(f"  Checkpoint        : {save_dir}")
    print(f"  Report            : {out_dir}\n")

    # ── Baseline ───────────────────────────────────────────────────────────
    per_loop, agg = evaluate_per_loop(algo, eval_env, cfg,
                                      args.eval_episodes_per_loop)
    print(f"  baseline (untrained): mean acc {agg['mean_accuracy']*100:.2f}%  "
          f"min {agg['min_accuracy']*100:.2f}%  "
          f"sensors {agg['mean_sensors']:.2f}\n")

    best_acc    = agg["mean_accuracy"]
    best_iter   = 0
    best_report = (per_loop, agg)
    saved_best  = False
    since_gain  = 0
    converged   = False
    history     = [dict(iteration=0, **agg)]
    t0          = time.time()

    print(f"  {'iter':>5}  {'mean_acc':>9}  {'min_acc':>8}  {'sensors':>8}  {'patience':>8}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")

    for i in range(1, args.max_iterations + 1):
        algo.train()

        if i % args.eval_interval and i != args.max_iterations:
            continue

        per_loop, agg = evaluate_per_loop(algo, eval_env, cfg,
                                          args.eval_episodes_per_loop)
        history.append(dict(iteration=i, **agg))

        improved     = agg["mean_accuracy"] > best_acc
        meaningfully = agg["mean_accuracy"] > best_acc + args.min_delta

        if improved:
            # Keep the high-water mark even when the gain is below min_delta,
            # so the saved checkpoint is always the best one seen.
            best_acc    = agg["mean_accuracy"]
            best_iter   = i
            best_report = (per_loop, agg)
            algo.save(save_dir)
            saved_best  = True

        # Only a MEANINGFUL gain resets patience — otherwise a long drift of
        # +0.01pp steps would keep training alive forever.
        since_gain = 0 if meaningfully else since_gain + 1
        marker     = "  * new best" if improved else ""

        print(f"  {i:5d}  {agg['mean_accuracy']*100:8.2f}%  "
              f"{agg['min_accuracy']*100:7.2f}%  {agg['mean_sensors']:8.2f}  "
              f"{since_gain:8d}{marker}")

        if since_gain >= args.patience:
            converged = True
            print(f"\n  Converged: no gain > {args.min_delta*100:.2f}pp "
                  f"in {args.patience} evals (stopped at iteration {i}).")
            break

    if not converged:
        print(f"\n  NOT CONVERGED: hit the {args.max_iterations}-iteration cap "
              f"while still improving. Treat the result as a floor.")

    # Final policy goes beside the best one, never over it.
    final_dir = save_dir.rstrip("/") + "_final"
    os.makedirs(final_dir, exist_ok=True)
    algo.save(final_dir if saved_best else save_dir)

    # ── Final per-loop report ──────────────────────────────────────────────
    # Training-time eval is deliberately cheap, so re-measure the BEST
    # checkpoint thoroughly.  This means restoring it: `algo` has trained past
    # that point, and reporting its current weights would not match best_iter.
    if saved_best:
        algo.restore(save_dir)
    per_loop, agg = evaluate_per_loop(algo, eval_env, cfg,
                                      args.final_eval_episodes_per_loop)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  PER-LOOP EVALUATION (best checkpoint, iteration {best_iter})")
    print("=" * 70)
    print(format_per_loop_table(per_loop, agg))
    print(f"\n  Converged   : {converged} at iteration {best_iter}")
    print(f"  Wall clock  : {elapsed/60:.1f} min")
    print(f"  Checkpoint  : {save_dir}")

    with open(os.path.join(out_dir, "per_loop_eval.json"), "w") as fh:
        json.dump({
            "run":               args.new_run,
            "loops":             loops,
            "loop_seed":         args.loop_seed,
            "time_limit":        time_limit,
            "max_sensors":       cfg["max_sensors"],
            "sensor_rew":        args.sensor_rew,
            "converged":         converged,
            "best_iteration":    best_iter,
            "iterations_run":    history[-1]["iteration"],
            "eval_episodes_per_loop": args.final_eval_episodes_per_loop,
            "eval_seed":         args.eval_seed,
            "per_loop":          per_loop,
            "aggregate":         agg,
            "wall_clock_min":    elapsed / 60.0,
        }, fh, indent=2)

    with open(os.path.join(out_dir, "per_loop_eval.md"), "w") as fh:
        fh.write(f"# Synthetic multi-loop run {args.new_run}\n\n")
        fh.write(f"- Loops: {args.num_loops} sampled with seed {args.loop_seed}\n")
        fh.write(f"- time_limit {time_limit}, max_sensors {cfg['max_sensors']} "
                 f"of {IOBT_NUM_NODES}, sensor_rew {args.sensor_rew}\n")
        fh.write(f"- Converged: {converged} (best iteration {best_iter} of "
                 f"{history[-1]['iteration']})\n")
        fh.write(f"- Eval: {args.final_eval_episodes_per_loop} episodes per loop, "
                 f"fixed seed {args.eval_seed}, dedicated env\n\n")
        fh.write(format_per_loop_table(per_loop, agg))
        fh.write("\n")

    with open(os.path.join(out_dir, "training_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    algo.stop()
    print(f"\n  Wrote {out_dir}/per_loop_eval.{{json,md}} and training_log.csv")


if __name__ == "__main__":
    main()
