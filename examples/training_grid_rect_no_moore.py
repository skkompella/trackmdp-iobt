#!/usr/bin/env python3
"""
training_grid_rect_no_moore.py — Training and evaluation for a rectangular M×N grid
with the Moore-neighbourhood window constraint disabled.

Identical to training_grid_rect.py except the sensor action covers all n_cells at
every time-delay level (full-grid sensing).  Use this when physical node transitions
violate Moore adjacency — e.g. the IoBT 6-node layout where n11↔n13, n11↔n16, and
n16↔n14 all jump 2 columns in the 2×3 grid.

Usage
-----
    python examples/training_grid_rect_no_moore.py                     # 2×3, 200 iters
    python examples/training_grid_rect_no_moore.py --nrows 3 --ncols 4
    python examples/training_grid_rect_no_moore.py --iterations 1000 --run 210
    python examples/training_grid_rect_no_moore.py --resume --run 210
    python examples/training_grid_rect_no_moore.py --eval-only --run 210
    python examples/training_grid_rect_no_moore.py --eval-only --checkpoint runs/agent_run210_ppo
"""

import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from src.core.grid_env_rect     import learning_grid_sarsa_0
from src.core.grid_wrapper_rect import grid_environment_rect


# ===========================================================================
# Defaults
# ===========================================================================

DEFAULTS = {
    "run_number":          210,
    "nrows":               2,
    "ncols":               3,
    "num_trans":           4,
    "max_sensors":         6,
    "max_sensors_null":    6,
    "time_limit":          1,
    "time_limit_max":      1,
    "training_iterations": 5,
    "eval_interval":       1,
    "eval_episodes":       100,
    "num_workers":         4,
    "lr":                  1e-4,
}

N_CELLS = DEFAULTS["nrows"] * DEFAULTS["ncols"]   # 6


def _build_augment_vecs(time_limit_max, n_cells):
    """No-moore version: every time-delay level has the same n_cells-wide window."""
    v1 = [n_cells] * (time_limit_max + 1)
    v  = [0]
    for x in v1:
        v.append(v[-1] + x)
    return v1, v


_AV1, _AV          = _build_augment_vecs(DEFAULTS["time_limit_max"], N_CELLS)
ACTION_HISTORY_LEN = _AV[-1]    # 12 for tl_max=1, n_cells=6
MAX_ACTION_SZ      = N_CELLS    # 6


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_policy(algo, env, cfg, num_episodes):
    nrows          = cfg["nrows"]
    ncols          = cfg["ncols"]
    n_cells        = nrows * ncols
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors    = cfg["max_sensors"]
    missing_state  = n_cells * (time_limit_max + 1) + 1

    av1, av = _build_augment_vecs(time_limit_max, n_cells)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < 1000:
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
                # Full-grid sensing: first n_cells elements are per-cell sensors
                action_sensors = np.array(action_full[:n_cells], dtype=int)
                obj_rel_pos    = env.object_pos   # absolute cell index
                obj_in_window  = 1                # always reachable
            else:
                action_sensors = np.ones(n_cells, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1

            if action_sensors.sum() > max_sensors:
                active = np.where(action_sensors == 1)[0][:max_sensors]
                action_sensors = np.zeros(n_cells, dtype=int)
                action_sensors[active] = 1

            obj_detected = int(obj_in_window == 1 and
                               action_sensors[int(obj_rel_pos)] == 1)
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
        total_steps += ep_steps

    return {
        "tracking_accuracy": total_found / max(total_steps, 1),
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
    nrows          = cfg["nrows"]
    ncols          = cfg["ncols"]
    n_cells        = nrows * ncols
    run_number     = cfg["run_number"]
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    save_dir       = save_dir_override or os.path.join(
        project_root, "runs", f"agent_run{run_number}_ppo"
    )

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
    print(f"  missing_state = {missing_state}")
    print(f"  cum_probs  : {[round(x, 4) for x in state_trans_cum_prob]}")

    qobj = learning_grid_sarsa_0(
        run_number, nrows, ncols, cfg["num_trans"], state_trans_cum_prob,
        cfg["max_sensors"], cfg["max_sensors_null"],
        time_limit, time_limit_max,
        no_moore_constraint=True,
    )

    env_config = {
        "qobj":                qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":      time_limit_max,
        "no_moore_constraint": True,
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
            print("[ERROR] No checkpoint found. Train first.")
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
    ppo_cfg.normalize_actions = False

    algo = ppo_cfg.build()
    if resume:
        ckpt, _ = find_latest_checkpoint(run_number, save_dir=save_dir)
        if ckpt:
            print(f"Resuming from checkpoint: {ckpt}")
            algo.restore(ckpt)
        else:
            print("No checkpoint found to resume — training from scratch.")
    else:
        print("Training from scratch (use --resume to continue from a checkpoint).")

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nTraining for {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")

    best_accuracy = 0.0
    for i in range(1, cfg["training_iterations"] + 1):
        result      = algo.train()
        reward_mean = (result.get("episode_reward_mean") or
                       result.get("env_runners", {}).get("episode_reward_mean",
                                                         float("nan")))
        print(f"  iter {i:5d}/{cfg['training_iterations']}  reward={reward_mean:+8.3f}")

        if i % cfg["eval_interval"] == 0 or i == cfg["training_iterations"]:
            algo.save(save_dir)
            metrics = evaluate_policy(algo, qobj.grid_env, cfg, cfg["eval_episodes"])
            _print_metrics(f"EVAL @ iter {i}", metrics)
            if metrics["tracking_accuracy"] > best_accuracy:
                best_accuracy = metrics["tracking_accuracy"]
                print(f"  ★ New best: {best_accuracy:.4f}")

    print(f"\n  Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    algo.stop()


def _print_metrics(label, m):
    print(f"\n  ── {label} ──")
    print(f"     Accuracy    : {m['tracking_accuracy']:.4f}  ({m['tracking_accuracy']*100:.2f}%)")
    print(f"     Reward      : {m['mean_reward']:+.3f} ± {m['std_reward']:.3f}")
    print(f"     Ep length   : {m['mean_length']:.1f} steps")
    print(f"     Sensors/step: {m['mean_sensors_used']:.2f}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Track-MDP PPO on a rectangular M×N grid (no Moore neighbourhood).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python examples/training_grid_rect_no_moore.py                       # 2×3, default iters
  python examples/training_grid_rect_no_moore.py --nrows 3 --ncols 4
  python examples/training_grid_rect_no_moore.py --iterations 500 --run 210
  python examples/training_grid_rect_no_moore.py --resume --run 210
  python examples/training_grid_rect_no_moore.py --eval-only --run 210
  python examples/training_grid_rect_no_moore.py --eval-only \\
      --checkpoint runs/agent_run210_ppo/checkpoint_000050
        """
    )
    parser.add_argument("--run",          type=int,   default=DEFAULTS["run_number"])
    parser.add_argument("--nrows",        type=int,   default=DEFAULTS["nrows"],
                        help=f"Number of rows M (default: {DEFAULTS['nrows']})")
    parser.add_argument("--ncols",        type=int,   default=DEFAULTS["ncols"],
                        help=f"Number of cols N (default: {DEFAULTS['ncols']})")
    parser.add_argument("--iterations",   type=int,   default=DEFAULTS["training_iterations"])
    parser.add_argument("--eval-interval",type=int,   default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval-episodes",type=int,   default=DEFAULTS["eval_episodes"])
    parser.add_argument("--workers",      type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--eval-only",    action="store_true")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume training from the latest checkpoint in save_dir.")
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="Explicit checkpoint path (implies --eval-only).")
    parser.add_argument("--save-dir",     type=str,   default=None,
                        help="Override checkpoint directory.")
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

    print("=" * 60)
    print("  TRACK-MDP  —  RECTANGULAR GRID TRAINING  (no-moore)")
    print("=" * 60)
    print(f"  Run number  : {cfg['run_number']}")
    print(f"  Grid        : {cfg['nrows']} rows × {cfg['ncols']} cols "
          f"({cfg['nrows']*cfg['ncols']} cells)")
    print(f"  Action space: {cfg['nrows']*cfg['ncols']} cells (full grid, no Moore window)")
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
