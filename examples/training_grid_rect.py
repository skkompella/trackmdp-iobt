#!/usr/bin/env python3
"""
training_grid_rect.py — Training and evaluation for a rectangular M×N grid.

Default grid: 2 rows × 3 cols (6 cells).  Change --nrows and --ncols to
experiment with other shapes, e.g. 3×4, 2×5, 4×4 (square).

Based on training_grid.py but parameterised for non-square grids.
All three files (grid_env_rect, gym_wrapper_rect, this file) form a complete
self-contained set — nothing is imported from trainer.py.

Usage
-----
    python examples/training_grid_rect.py                     # 2×3, 200 iters
    python examples/training_grid_rect.py --nrows 3 --ncols 4
    python examples/training_grid_rect.py --iterations 1000 --run 202
    python examples/training_grid_rect.py --resume --run 202  # continue prior run
    python examples/training_grid_rect.py --eval-only --run 202
    python examples/training_grid_rect.py --eval-only --checkpoint runs/agent_run202_ppo
"""

import os
import sys
import csv
import json
import shutil
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from src.core.grid_env_rect  import learning_grid_sarsa_0
from src.core.grid_wrapper_rect import grid_environment_rect


# ===========================================================================
# Defaults
# ===========================================================================

DEFAULTS = {
    "run_number":          201,
    "nrows":               2,       # M — rows
    "ncols":               3,       # N — columns
    "num_trans":           4,
    "max_sensors":         9,
    "max_sensors_null":    9,
    "time_limit":          1,
    "time_limit_max":      1,
    "training_iterations": 5,
    "eval_interval":       1,
    "eval_episodes":       100,
    "num_workers":         4,
    "lr":                  1e-4,
    "max_ep_steps":        100,       # training episode cap; 0 = none (ends on target exit)
    "movement":            "moore",   # target movement: stay + 8 neighbours ("4conn": stay + 4)
    "seed":                None,      # None = unseeded; an int makes runs reproducible
    "sensor_budget":       "none",    # "none" = upstream (k unused); "env" = enforce k in env
    "init_checkpoint":     None,      # start from this checkpoint (branch a run)
}


def _build_augment_vecs(time_limit_max):
    v1 = [(2 * i + 3) ** 2 for i in range(time_limit_max + 1)]
    v  = [0]
    for x in v1:
        v.append(v[-1] + x)
    return v1, v

_AV1, _AV   = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AV[-1]                               # 34
MAX_ACTION_SZ      = (2 * DEFAULTS["time_limit_max"] + 3) ** 2  # 25


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_policy(algo, env, cfg, num_episodes):
    nrows         = cfg["nrows"]
    ncols         = cfg["ncols"]
    n_cells       = nrows * ncols
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors   = cfg["max_sensors"]
    missing_state = n_cells * (time_limit_max + 1) + 1

    av1, av = _build_augment_vecs(time_limit_max)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0
    ep_upstream_acc = []     # per-episode accuracy, upstream Track-MDP definition

    for _ in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0
        ep_found_upstream = 0

        while ep_steps < 1000: #was 1000
            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                action_vec[:av[time_delay]] = actions_list

            if not was_missing:
                state_pos  = current_state // (time_limit + 1)
                state_time = current_state %  (time_limit + 1)
            else:
                state_pos, state_time = n_cells, 0

            obs         = (state_pos, state_time, np.array(action_vec, dtype=np.int64))
            action_full = algo.compute_single_action(obs, explore=False)

            if not was_missing:
                action_clip    = np.array(action_full[-num_sensors:], dtype=int)
                action_sensors = np.multiply(
                    action_clip,
                    env.valid_q_indices_dict[time_delay][current_state]
                )
                obj_rel_pos, obj_in_window = env.realign_obj(
                    env.object_pos, current_state, time_delay
                )
            else:
                action_sensors = np.ones(num_sensors, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1

            if not was_missing:
                # The environment's own budget rule (standard.md): no-op for
                # sensor_budget="none", random keep-k for "env". The same rule
                # applies in training, so evaluation matches what was learned.
                action_sensors = env.apply_budget(action_sensors)

            obj_detected = int(obj_in_window == 1 and
                               action_sensors[int(obj_rel_pos)] == 1)
            # Upstream Track-MDP counts a missing-state (full rescan) step as found.
            ep_found_upstream += 1 if was_missing else obj_detected
            if not was_missing and obj_detected:
                total_found += 1

            reward, next_state, terminal_flag, time_delay = \
                env.get_reward_next_state(current_state, action_sensors, time_delay)

            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(action_sensors.sum())

            if time_delay == 0:
                actions_list = []
            else:
                td_ac        = av1[time_delay - 1]
                action_bool  = ([1] * MAX_ACTION_SZ if was_missing
                                else list(action_full))
                actions_list = actions_list + list(np.array(action_bool)[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        ep_lengths.append(ep_steps)
        sensors_ep.append(ep_sensors / max(ep_steps, 1))
        ep_upstream_acc.append(ep_found_upstream / max(ep_steps, 1))
        total_steps += ep_steps

    return {
        # standard.md: the upstream metric is THE accuracy
        "accuracy":          float(np.mean(ep_upstream_acc)),
        # port metric: hits only while tracking, pooled over all steps
        "tracked_accuracy":  total_found / max(total_steps, 1),
        "mean_reward":       float(np.mean(total_rewards)),
        "std_reward":        float(np.std(total_rewards)),
        "mean_length":       float(np.mean(ep_lengths)),
        "mean_sensors_used": float(np.mean(sensors_ep)),
    }


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _is_rllib_checkpoint(path):
    return (os.path.isfile(os.path.join(path, "rllib_checkpoint.json")) or
            os.path.isfile(os.path.join(path, "algorithm_state.pkl")))


def find_latest_checkpoint(run_number, save_dir=None):
    if save_dir:
        candidates = [save_dir]
    else:
        runs = os.path.join(project_root, "runs")
        candidates = [
            os.path.join(runs, f"agent_run{run_number}_ppo"),
            f"./agent_run{run_number}_ppo",
        ]
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        if _is_rllib_checkpoint(directory):
            return directory, directory
        current = os.path.join(directory, "current")
        if _is_rllib_checkpoint(current):
            return current, directory
        checkpoints = []
        for item in os.listdir(directory):
            full = os.path.join(directory, item)
            if os.path.isdir(full) and item.startswith("checkpoint_"):
                try:
                    checkpoints.append((int(item.split("_")[1]), full))
                except (ValueError, IndexError):
                    pass
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            return checkpoints[-1][1], directory
    return None, candidates[0] if candidates else f"./agent_run{run_number}_ppo"


# ===========================================================================
# Training
# ===========================================================================

def train(cfg, eval_only=False, checkpoint=None, save_dir_override=None, resume=False):
    nrows         = cfg["nrows"]
    ncols         = cfg["ncols"]
    n_cells       = nrows * ncols
    run_number    = cfg["run_number"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    save_dir      = f"./agent_run{run_number}_ppo"

    missing_state = n_cells * (time_limit_max + 1) + 1

    terminal_prob  = 0.005
    state_prob_run = 0.15
    state_trans_cum_prob = [
        i * (1 - terminal_prob - state_prob_run) / float(cfg["num_trans"] - 1)
        for i in range(1, cfg["num_trans"])
    ]
    state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run]

    print(f"\n  Grid       : {nrows} rows × {ncols} cols  ({n_cells} cells)")
    print(f"  num_trans  : {cfg['num_trans']}")
    print(f"  movement   : {cfg['movement']}   max_sensors (k): {cfg['max_sensors']}   "
          f"seed: {cfg['seed']}   train ep cap: {cfg['max_ep_steps'] or 'none'}   "
          f"sensor budget: {cfg['sensor_budget']}")
    print(f"  missing_state = {missing_state}")
    print(f"  cum_probs  : {[round(x, 4) for x in state_trans_cum_prob]}")

    # Single qobj used for both training workers and eval
    qobj = learning_grid_sarsa_0(
        run_number, nrows, ncols, cfg["num_trans"], state_trans_cum_prob,
        cfg["max_sensors"], cfg["max_sensors_null"],
        time_limit, time_limit_max,
        movement=cfg["movement"], seed=cfg["seed"],
        sensor_budget=cfg["sensor_budget"],
    )

    env_config = {
        "qobj":                qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":      time_limit_max,
        "seed":                cfg["seed"],
        "max_ep_steps":        cfg["max_ep_steps"],
    }

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
        )

    # ── Eval-only mode ────────────────────────────────────────────────────
    if eval_only:
        ckpt = checkpoint or find_latest_checkpoint(
            run_number, save_dir=save_dir_override
        )[0]
        if ckpt is None:
            print(f"[ERROR] No checkpoint found. Train first.")
            return
        print(f"Loading checkpoint: {ckpt}")
        algo    = PPO.from_checkpoint(ckpt)
        metrics = evaluate_policy(algo, qobj.grid_env, cfg, cfg["eval_episodes"])
        _print_metrics("EVAL", metrics)
        return

    # ── Build PPO config ──────────────────────────────────────────────────
    ppo_cfg = (
        PPOConfig()
        .environment(grid_environment_rect, env_config=env_config)
        .framework("torch")
        .training(lr=cfg["lr"], grad_clip=30.0)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=cfg["num_workers"])
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    ppo_cfg.normalize_actions = False   # critical for MultiDiscrete

    # Resume from checkpoint only when explicitly requested via --resume.
    # Always search within this script's own save_dir to avoid accidentally
    # loading incompatible checkpoints from other scripts (e.g. in runs/).
    algo = ppo_cfg.build()
    if resume:
        ckpt, _ = find_latest_checkpoint(
            run_number, save_dir=save_dir_override or save_dir
        )
        if ckpt:
            print(f"Resuming from checkpoint: {ckpt}")
            algo.restore(ckpt)
        else:
            print("No checkpoint found to resume — training from scratch.")
    elif cfg.get("init_checkpoint"):
        # Branch from another run's checkpoint (weights + optimizer state) into
        # this run's environment, e.g. a new movement model. Restoring before
        # any sampling is safe: no worker is mid-episode.
        ckpt = os.path.abspath(cfg["init_checkpoint"])
        print(f"Starting from checkpoint: {ckpt}")
        algo.restore(ckpt)
        algo.env_runner_group.sync_weights()
    else:
        print("Training from scratch (use --resume to continue from a checkpoint).")

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nTraining for {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")

    # Every eval: save the current checkpoint to <save_dir>/current, copy it to
    # <save_dir>/best when accuracy improves, and append to the metric history
    # (metrics_history.json / .csv). The best is reported at the end.
    current_dir = os.path.join(save_dir, "current")
    best_dir    = os.path.join(save_dir, "best")
    history     = []
    best        = None
    if cfg.get("init_checkpoint"):
        # Iteration 0: the branched policy on the new environment, before any
        # training -- the accuracy right after the change.
        metrics = evaluate_policy(algo, qobj.grid_env, cfg, cfg["eval_episodes"])
        _print_metrics("EVAL @ iter 0 (before training)", metrics)
        history.append({"iteration": 0, "train_reward_mean": float("nan"), **metrics})
        os.makedirs(save_dir, exist_ok=True)
        _write_history(save_dir, history, None, cfg)
    for i in range(1, cfg["training_iterations"] + 1):
        result      = algo.train()
        reward_mean = (result.get("episode_reward_mean") or
                       result.get("env_runners", {}).get("episode_reward_mean",
                                                         float("nan")))
        print(f"  iter {i:5d}/{cfg['training_iterations']}  reward={reward_mean:+8.3f}")

        if i % cfg["eval_interval"] == 0 or i == cfg["training_iterations"]:
            algo.save(os.path.abspath(current_dir))
            metrics = evaluate_policy(algo, qobj.grid_env, cfg, cfg["eval_episodes"])
            _print_metrics(f"EVAL @ iter {i}", metrics)
            row = {"iteration": i, "train_reward_mean": reward_mean, **metrics}
            history.append(row)
            if best is None or metrics["accuracy"] > best["accuracy"]:
                best = row
                shutil.rmtree(best_dir, ignore_errors=True)
                shutil.copytree(current_dir, best_dir)
                print(f"  ★ New best: {best['accuracy']:.4f} (saved to {best_dir})")
            _write_history(save_dir, history, best, cfg)

    if best is not None:
        print(f"\n  Best accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%) "
              f"at iteration {best['iteration']}  "
              f"[tracked {best['tracked_accuracy']*100:.2f}%, "
              f"{best['mean_sensors_used']:.2f} sensors/step]")
        print(f"  Best checkpoint   : {best_dir}")
        print(f"  Current checkpoint: {current_dir}")
        print(f"  Metric history    : {os.path.join(save_dir, 'metrics_history.csv')}")
    algo.stop()


def _write_history(save_dir, history, best, cfg):
    """Metric history as JSON (with run config and best) and CSV, rewritten each eval."""
    with open(os.path.join(save_dir, "metrics_history.json"), "w") as fh:
        json.dump({"config": {k: v for k, v in cfg.items()},
                   "best": best, "history": history}, fh, indent=1, default=str)
    with open(os.path.join(save_dir, "metrics_history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def _print_metrics(label, m):
    print(f"\n  ── {label} ──")
    print(f"     Accuracy    : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)  "
          f"[upstream metric; tracked-only {m['tracked_accuracy']*100:.2f}%]")
    print(f"     Reward      : {m['mean_reward']:+.3f} ± {m['std_reward']:.3f}")
    print(f"     Ep length   : {m['mean_length']:.1f} steps")
    print(f"     Sensors/step: {m['mean_sensors_used']:.2f}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Track-MDP PPO on a rectangular M×N grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python examples/training_grid_rect.py                       # 2×3, 200 iters (from scratch)
  python examples/training_grid_rect.py --nrows 3 --ncols 4  # 3×4, 200 iters
  python examples/training_grid_rect.py --iterations 1000 --run 202
  python examples/training_grid_rect.py --resume --run 202   # continue prior training
  python examples/training_grid_rect.py --eval-only --run 202
  python examples/training_grid_rect.py --eval-only \\
      --checkpoint agent_run202_ppo/checkpoint_000050
        """
    )
    parser.add_argument("--run",        type=int,   default=DEFAULTS["run_number"])
    parser.add_argument("--nrows",      type=int,   default=DEFAULTS["nrows"],
                        help=f"Number of rows M (default: {DEFAULTS['nrows']})")
    parser.add_argument("--ncols",      type=int,   default=DEFAULTS["ncols"],
                        help=f"Number of cols N (default: {DEFAULTS['ncols']})")
    parser.add_argument("--iterations", type=int,   default=DEFAULTS["training_iterations"])
    parser.add_argument("--eval-interval", type=int, default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval-episodes", type=int, default=DEFAULTS["eval_episodes"])
    parser.add_argument("--workers",    type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--lr",         type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max-sensors", type=int, default=DEFAULTS["max_sensors"],
                        help="Sensor budget k (default: %(default)s)")
    parser.add_argument("--max-ep-steps", type=int, default=DEFAULTS["max_ep_steps"],
                        help="Training episode cap in steps; 0 = no cap, episodes "
                             "end only when the target exits (default: %(default)s)")
    parser.add_argument("--movement",   choices=["moore", "4conn", "original"],
                        default=DEFAULTS["movement"],
                        help="Target movement: moore = stay + 8 neighbours, "
                             "4conn = stay + 4, original = Track-MDP's 4 random "
                             "3x3 destinations per cell (default: %(default)s)")
    parser.add_argument("--sensor-budget", choices=["none", "env"],
                        default=DEFAULTS["sensor_budget"],
                        help="none = upstream, k never applied; env = at most k "
                             "tracking sensors per step, enforced in the environment "
                             "in training and evaluation (default: %(default)s)")
    parser.add_argument("--seed",       type=int,   default=DEFAULTS["seed"],
                        help="Seed the target-movement RNG (default: unseeded)")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Start training from this checkpoint (weights + optimizer), "
                             "e.g. to branch a run onto a new movement model; also "
                             "evaluates at iteration 0, before any training")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from the latest checkpoint in save_dir.")
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Explicit checkpoint path (implies --eval-only).")
    parser.add_argument("--save-dir",   type=str,   default=None,
                        help="Override checkpoint search directory.")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["run_number"]          = args.run
    cfg["nrows"]               = args.nrows
    cfg["ncols"]               = args.ncols
    cfg["training_iterations"] = args.iterations
    cfg["eval_interval"]       = args.eval_interval
    cfg["eval_episodes"]       = args.eval_episodes
    cfg["num_workers"]         = args.workers
    cfg["lr"]                  = args.lr
    cfg["max_sensors"]         = args.max_sensors
    cfg["max_sensors_null"]    = args.max_sensors
    cfg["movement"]            = args.movement
    cfg["max_ep_steps"]        = args.max_ep_steps
    cfg["sensor_budget"]       = args.sensor_budget
    cfg["init_checkpoint"]     = args.init_checkpoint
    cfg["seed"]                = args.seed

    print("=" * 60)
    print("  TRACK-MDP  —  RECTANGULAR GRID TRAINING")
    print("=" * 60)
    print(f"  Run number  : {cfg['run_number']}")
    print(f"  Grid        : {cfg['nrows']} rows × {cfg['ncols']} cols "
          f"({cfg['nrows']*cfg['ncols']} cells)")
    print(f"  Iterations  : {cfg['training_iterations']}")
    print(f"  Eval every  : {cfg['eval_interval']}")
    eval_only = args.eval_only or (args.checkpoint is not None)
    if eval_only:
        mode_str = "eval only"
    elif args.resume:
        mode_str = "train + eval (resuming)"
    else:
        mode_str = "train + eval (from scratch)"
    print(f"  Mode        : {mode_str}")

    try:
        train(cfg, eval_only=eval_only,
              checkpoint=args.checkpoint,
              save_dir_override=args.save_dir,
              resume=args.resume)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()