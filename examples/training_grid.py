#!/usr/bin/env python3
"""
training_grid.py — Self-contained training and evaluation for the 3×3 grid.

Fixes applied vs the original:
  - Removed the call to train() from trainer.py, which hardcoded N=10 and
    run_number=198, making it completely ignore this file's config.
  - Training, checkpointing, and eval all use the same qobj and run_number.
  - No observation/action space mismatch between training and eval.
  - Iteration count is configurable via --iterations (default: 200 for a quick run).

Usage:
    python examples/training_grid.py
    python examples/training_grid.py --iterations 500 --run 193
    python examples/training_grid.py --eval-only --run 193
    python examples/training_grid.py --eval-only --checkpoint runs/agent_run193_ppo
    python examples/training_grid.py --eval-only --save-dir runs/agent_run193_ppo
"""

import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from src.core.environment import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment


# ===========================================================================
# Config — all in one place, no hidden hardcoding elsewhere
# ===========================================================================

DEFAULTS = {
    "run_number":        193,
    "N":                 3,          # 3×3 grid
    "num_trans":         4,
    "max_sensors":       6,
    "max_sensors_null":  6,
    "time_limit":        1,
    "time_limit_max":    1,
    "training_iterations": 200,      # fast default; bump for real training
    "eval_interval":     10,
    "eval_episodes":     100,
    "num_workers":       4,
    "lr":                1e-4,
}

# Augment vecs — must match gym_wrapper.py
def _build_augment_vecs(time_limit_max):
    v1 = [(2*i+3)**2 for i in range(time_limit_max+1)]
    v  = [0]
    for x in v1: v.append(v[-1]+x)
    return v1, v

_AV1, _AV = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AV[-1]                              # 34
MAX_ACTION_SZ      = (2*DEFAULTS["time_limit_max"]+3)**2  # 25


# ===========================================================================
# Evaluation (self-contained, no import from trainer.py)
# ===========================================================================

def evaluate_policy(algo, env, cfg, num_episodes):
    N             = cfg["N"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors   = cfg["max_sensors"]
    missing_state = N*N*(time_limit_max+1)+1
    av1, av       = _build_augment_vecs(time_limit_max)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < 1000:
            num_sensors = (2*time_delay+3)**2
            was_missing = (current_state == missing_state)

            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                action_vec[:av[time_delay]] = actions_list

            if not was_missing:
                state_pos  = current_state // (time_limit+1)
                state_time = current_state %  (time_limit+1)
            else:
                state_pos, state_time = N*N, 0

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

            if action_sensors.sum() > max_sensors:
                active = np.where(action_sensors==1)[0][:max_sensors]
                action_sensors = np.zeros(num_sensors, dtype=int)
                action_sensors[active] = 1

            obj_detected = int(obj_in_window==1 and action_sensors[int(obj_rel_pos)]==1)
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
                td_ac        = av1[time_delay-1]
                action_bool  = [1]*MAX_ACTION_SZ if was_missing else list(action_full)
                actions_list = actions_list + list(np.array(action_bool)[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        ep_lengths.append(ep_steps)
        sensors_ep.append(ep_sensors / max(ep_steps,1))
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
    """
    Return (checkpoint_path, search_dir), or (None, search_dir) on failure.

    Search order (first hit wins):
      1. Explicit save_dir argument
      2. <project_root>/runs/agent_run<N>_ppo
      3. ./agent_run<N>_ppo  (legacy relative path)
    """
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

def train(cfg, eval_only=False, checkpoint=None, save_dir_override=None):
    run_number  = cfg["run_number"]
    N           = cfg["N"]
    time_limit  = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    save_dir    = f"./agent_run{run_number}_ppo"

    terminal_st_prob = 0.005
    state_prob_run   = 0.15
    state_trans_cum_prob = [
        i*(1-terminal_st_prob-state_prob_run)/float(cfg["num_trans"]-1)
        for i in range(1, cfg["num_trans"])
    ]
    state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run]

    # Build ONE qobj — used for both training env and eval
    qobj = learning_grid_sarsa_0(
        run_number, N, cfg["num_trans"], state_trans_cum_prob,
        cfg["max_sensors"], cfg["max_sensors_null"],
        time_limit, time_limit_max
    )

    missing_state = N*N*(time_limit_max+1)+1

    print(f"\n  N={N}  num_trans={cfg['num_trans']}  missing_state={missing_state}")
    print(f"  state_trans_cum_prob = {[round(x,4) for x in state_trans_cum_prob]}")

    env_config = {
        "qobj":               qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":     time_limit_max,
    }

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
        )

    # Eval-only mode
    if eval_only:
        if checkpoint:
            ckpt = checkpoint
        else:
            ckpt, search = find_latest_checkpoint(run_number, save_dir=save_dir_override)
            if ckpt is None:
                print(f"[ERROR] No checkpoint found in {search}. Train first.")
                return
        print(f"Loading checkpoint: {ckpt}")
        algo = PPO.from_checkpoint(ckpt)
        metrics = evaluate_policy(algo, qobj.grid_env, cfg, cfg["eval_episodes"])
        _print_metrics("EVAL", metrics)
        return

    # Build PPO config — same structure as iobt_new_training.py
    ppo_cfg = (
        PPOConfig()
        .environment(grid_environment, env_config=env_config)
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

    # Resume from existing checkpoint if present
    ckpt, _ = find_latest_checkpoint(run_number, save_dir=save_dir_override)
    algo     = ppo_cfg.build()
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}")
        algo.restore(ckpt)
    else:
        print("No existing checkpoint — training from scratch.")

    os.makedirs(save_dir, exist_ok=True)

    print(f"\nTraining for {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")

    best_accuracy = 0.0
    for i in range(1, cfg["training_iterations"]+1):
        result      = algo.train()
        reward_mean = (result.get("episode_reward_mean") or
                       result.get("env_runners", {}).get("episode_reward_mean", float("nan")))
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
    print(f"     Accuracy   : {m['tracking_accuracy']:.4f}  ({m['tracking_accuracy']*100:.2f}%)")
    print(f"     Reward     : {m['mean_reward']:+.3f} ± {m['std_reward']:.3f}")
    print(f"     Ep length  : {m['mean_length']:.1f}")
    print(f"     Sensors/step: {m['mean_sensors_used']:.2f}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Track-MDP PPO agent on a 3×3 grid."
    )
    parser.add_argument("--run",        type=int,   default=DEFAULTS["run_number"])
    parser.add_argument("--iterations", type=int,   default=DEFAULTS["training_iterations"],
                        help=f"Training iterations (default: {DEFAULTS['training_iterations']})")
    parser.add_argument("--eval-interval", type=int, default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval-episodes", type=int, default=DEFAULTS["eval_episodes"])
    parser.add_argument("--workers",    type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--lr",         type=float, default=DEFAULTS["lr"])
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, just evaluate the latest checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path for eval (implies --eval-only).")
    parser.add_argument("--save-dir",   type=str, default=None,
                        help="Override checkpoint search directory.")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["run_number"]          = args.run
    cfg["training_iterations"] = args.iterations
    cfg["eval_interval"]       = args.eval_interval
    cfg["eval_episodes"]       = args.eval_episodes
    cfg["num_workers"]         = args.workers
    cfg["lr"]                  = args.lr

    print("=" * 60)
    print("  TRACK-MDP  —  3×3 BASIC TRAINING")
    print("=" * 60)
    print(f"  Run number : {cfg['run_number']}")
    print(f"  Grid       : {cfg['N']}×{cfg['N']}")
    print(f"  Iterations : {cfg['training_iterations']}")
    print(f"  Eval every : {cfg['eval_interval']}")
    print(f"  Eval eps   : {cfg['eval_episodes']}")
    eval_only = args.eval_only or (args.checkpoint is not None)
    print(f"  Mode       : {'eval only' if eval_only else 'train + eval'}")
    if args.checkpoint:
        print(f"  Checkpoint : {args.checkpoint}")
    if args.save_dir:
        print(f"  Save dir   : {args.save_dir}")

    try:
        train(cfg, eval_only=eval_only,
              checkpoint=args.checkpoint, save_dir_override=args.save_dir)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()