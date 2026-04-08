#!/usr/bin/env python3
"""
iobt_eval.py — Evaluate a trained IoBT PPO policy (from iobt_new_training.py).

Usage
-----
    python examples/iobt_eval.py                    # run 195, 100 episodes
    python examples/iobt_eval.py --run 195          # explicit run number
    python examples/iobt_eval.py --checkpoint path  # explicit checkpoint path
    python examples/iobt_eval.py --episodes 500     # more eval episodes

Environment: 4x4 IoBT grid (Camp Buckner 10-node layout)
  - N=4, num_trans=6, max_sensors=6, time_limit=1
  - Observation: Tuple(state_pos, state_time, action_history[34])
  - Action: MultiDiscrete([2]*25)  — 5x5 sensor window
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import ray
from ray.rllib.algorithms.ppo import PPO

from src.core.iobt_new_env import learning_grid_sarsa_0


# ===========================================================================
# Default configuration — must match iobt_new_training.py
# ===========================================================================

DEFAULTS = {
    "run_number":      195,
    "N":               4,
    "num_trans":       6,
    "max_sensors":     6,
    "max_sensors_null": 6,
    "time_limit":      1,
    "time_limit_max":  1,
    "max_episode_steps": 100,
}

# Replicate gym_wrapper.py's augment_vec for time_limit_max=1:
#   augment_vec_1 = [9, 25]  (sensor-window sizes per time_delay)
#   augment_vec   = [0, 9, 34]
# ACTION_HISTORY_LEN = augment_vec[-1] = 34
def _build_augment_vecs(time_limit_max):
    augment_vec_1 = [(2 * i + 3) ** 2 for i in range(time_limit_max + 1)]
    augment_vec = [0]
    for v in augment_vec_1:
        augment_vec.append(augment_vec[-1] + v)
    return augment_vec_1, augment_vec

_AUGMENT_VEC_1, _AUGMENT_VEC = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AUGMENT_VEC[-1]   # 34
MAX_ACTION_SZ = (2 * DEFAULTS["time_limit_max"] + 3) ** 2  # 25


# ===========================================================================
# Evaluation loop
# ===========================================================================

def evaluate_policy(algo, cfg, num_episodes=10):
    """
    Run num_episodes greedy rollouts and return aggregate metrics.

    Uses iobt_env from iobt_new_env.py directly (bypasses the Gym wrapper)
    and queries the policy with the same Tuple observation format that
    grid_environment (gym_wrapper.py) used during training:
        (state_pos: int, state_time: int, action_history: int[34])

    Returns
    -------
    dict with keys:
        tracking_accuracy  — detections / total steps
        mean_reward        — mean episode reward
        std_reward         — std of episode rewards
        mean_length        — mean episode length
        mean_sensors_used  — mean sensors activated per step
    """
    N             = cfg["N"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors   = cfg["max_sensors"]
    max_steps     = cfg["max_episode_steps"]

    missing_state = N * N * (time_limit_max + 1) + 1
    state_trans_cum_prob = [round((i + 1) / cfg["num_trans"], 4)
                            for i in range(cfg["num_trans"])]

    qobj = learning_grid_sarsa_0(
        run_number=cfg["run_number"],
        N=N,
        num_trans=cfg["num_trans"],
        state_trans_cum_prob=state_trans_cum_prob,
        max_sensors=max_sensors,
        max_sensors_null=cfg["max_sensors_null"],
        time_limit=time_limit,
        time_limit_max=time_limit_max,
    )
    env = qobj.grid_env

    total_rewards   = []
    episode_lengths = []
    sensors_per_ep  = []
    total_steps     = 0
    total_found     = 0

    augment_vec_1, augment_vec = _build_augment_vecs(time_limit_max)

    for ep in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []          # mirrors gym_wrapper's self.actions_list

        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < max_steps:
            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            # ---- Build tuple observation (mirrors gym_wrapper.to_tuple_augment_state) ----
            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                e_index = augment_vec[time_delay]   # augment_vec[time_delay-1+1]
                action_vec[:e_index] = actions_list

            if not was_missing:
                state_pos  = current_state // (time_limit + 1)
                state_time = current_state %  (time_limit + 1)
            else:
                state_pos, state_time = N * N, 0

            obs = (state_pos, state_time, np.array(action_vec, dtype=np.int64))

            # ---- Query policy ----
            action_full = algo.compute_single_action(obs, explore=False)

            # ---- Derive sensor actions ----
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
                # Missing state: broadcast all sensors to re-acquire
                action_sensors = np.ones(num_sensors, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1

            # Enforce sensor budget
            if action_sensors.sum() > max_sensors:
                active = np.where(action_sensors == 1)[0][:max_sensors]
                action_sensors = np.zeros(num_sensors, dtype=int)
                action_sensors[active] = 1

            # ---- Detection check (skip missing-state re-acquisition) ----
            obj_detected = int(obj_in_window == 1 and
                               action_sensors[int(obj_rel_pos)] == 1)
            if not was_missing and obj_detected:
                total_found += 1

            # ---- Step environment ----
            reward, next_state, terminal_flag, time_delay = \
                env.get_reward_next_state(current_state, action_sensors, time_delay)

            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(action_sensors.sum())

            # ---- Update action history (mirrors gym_wrapper.step) ----
            # time_delay is now the NEW delay returned by get_reward_next_state
            if time_delay == 0:
                actions_list = []
            else:
                td_ac = augment_vec_1[time_delay - 1]
                action_bool = [1] * MAX_ACTION_SZ if was_missing else list(action_full)
                actions_list = actions_list + list(np.array(action_bool)[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        sensors_per_ep.append(ep_sensors / max(ep_steps, 1))
        total_steps += ep_steps

        print("episode reward mean:", ep, ep_reward)

    accuracy = total_found / max(total_steps, 1)

    return {
        "tracking_accuracy": accuracy,
        "mean_reward":       float(np.mean(total_rewards)),
        "std_reward":        float(np.std(total_rewards)),
        "mean_length":       float(np.mean(episode_lengths)),
        "mean_sensors_used": float(np.mean(sensors_per_ep)),
    }


# ===========================================================================
# Checkpoint discovery
# ===========================================================================

def _is_rllib_checkpoint(path):
    return (os.path.isfile(os.path.join(path, "rllib_checkpoint.json")) or
            os.path.isfile(os.path.join(path, "algorithm_state.pkl")))


def find_latest_checkpoint(run_number, save_dir=None):
    """
    Return (checkpoint_path, search_dir), or (None, search_dir) on failure.

    Search order (first hit wins):
      1. Explicit save_dir
      2. <project_root>/runs/agent_run<N>_ppo
      3. <project_root>/runs/agent_iobt_run<N>_ppo
    """
    if save_dir:
        candidates = [save_dir]
    else:
        runs = os.path.join(_PROJECT_ROOT, "runs")
        candidates = [
            os.path.join(runs, f"agent_run{run_number}_ppo"),
            os.path.join(runs, f"agent_iobt_run{run_number}_ppo"),
        ]

    for directory in candidates:
        if not os.path.isdir(directory):
            continue

        # Directory itself is the checkpoint (RLlib saves directly here)
        if _is_rllib_checkpoint(directory):
            return directory, directory

        # Older layout: checkpoint_XXX subdirectories
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

    return None, candidates[0] if candidates else f"runs/agent_run{run_number}_ppo"


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained IoBT PPO policy (iobt_new_training.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python examples/iobt_eval.py
  python examples/iobt_eval.py --run 195 --episodes 500
  python examples/iobt_eval.py --checkpoint runs/agent_run195_ppo
        """
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory. Overrides --run.")
    parser.add_argument("--run",        type=int, default=DEFAULTS["run_number"],
                        help=f"Run number (default: {DEFAULTS['run_number']})")
    parser.add_argument("--save-dir",   type=str, default=None,
                        help="Override checkpoint search directory.")
    parser.add_argument("--episodes",   type=int, default=10,
                        help="Number of evaluation episodes (default: 100).")
    parser.add_argument("--max-sensors", type=int, default=None,
                        help="Override max_sensors.")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.max_sensors is not None:
        cfg["max_sensors"] = args.max_sensors

    # ---- Locate checkpoint ----
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        search_dir      = os.path.dirname(checkpoint_path)
    else:
        checkpoint_path, search_dir = find_latest_checkpoint(
            args.run, save_dir=args.save_dir
        )

    if checkpoint_path is None:
        print(f"[ERROR] No checkpoint found in: {search_dir}")
        print("  Train first with iobt_new_training.py, then re-run this script.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  IoBT PPO — POLICY EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Max sensors  : {cfg['max_sensors']}")
    print()

    # ---- Load policy ----
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT}})

    print("Loading policy …")
    algo = PPO.from_checkpoint(checkpoint_path)
    print("Policy loaded.\n")

    # ---- Evaluate ----
    print(f"Running {args.episodes} greedy evaluation episodes …")
    metrics = evaluate_policy(algo, cfg, num_episodes=args.episodes)

    # ---- Report ----
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Tracking Accuracy   : {metrics['tracking_accuracy']:.4f}  "
          f"({metrics['tracking_accuracy']*100:.2f}%)")
    print(f"  Mean Reward         : {metrics['mean_reward']:+.3f}  "
          f"± {metrics['std_reward']:.3f}")
    print(f"  Mean Episode Length : {metrics['mean_length']:.1f} steps")
    print(f"  Mean Sensors / Step : {metrics['mean_sensors_used']:.2f}")
    print("=" * 60)

    acc = metrics["tracking_accuracy"]
    if acc >= 0.80:
        print("  Outstanding tracking performance.")
    elif acc >= 0.65:
        print("  Good tracking performance.")
    elif acc >= 0.50:
        print("  Moderate — consider longer training or tuning.")
    else:
        print("  Poor — check environment config or continue training.")
    print()

    ray.shutdown()


if __name__ == "__main__":
    main()
