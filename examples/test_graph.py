#!/usr/bin/env python3
"""
Grid Environment — Policy Evaluation
======================================
Loads the agent_run194_ppo PPO checkpoint and evaluates it on the 10x10 grid
tracking environment it was actually trained on.

The environment is reconstructed from the exact parameters stored in the
checkpoint (N=10, num_trans=4, max_sensors=6, time_limit_max=1).

Usage
-----
    python examples/test_graph.py
    python examples/test_graph.py --episodes 500
    python examples/test_graph.py --checkpoint ./agent_run194_ppo
    python examples/test_graph.py --stochastic
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from src.core.environment import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment

# ---------------------------------------------------------------------------
# Parameters matching agent_run194_ppo (extracted from algorithm_state.pkl)
# ---------------------------------------------------------------------------
TRAIN_PARAMS = {
    "run_number":          194,
    "N":                   10,
    "num_trans":           4,
    "prob_list_cum":       [0.2816666666666667, 0.5633333333333334,
                            0.8450000000000001, 0.9950000000000001],
    "max_sensors":         6,
    "max_sensors_null":    6,
    "time_limit":          1,
    "time_limit_max":      1,
    "time_limit_schedule": [2000],
}

DEFAULT_CHECKPOINT = os.path.join(_HERE, "agent_run194_ppo")


# ===========================================================================
# Environment factory
# ===========================================================================

def make_env():
    """Recreate the grid_environment with the original training parameters."""
    p = TRAIN_PARAMS
    qobj = learning_grid_sarsa_0(
        run_number=p["run_number"],
        N=p["N"],
        num_trans=p["num_trans"],
        state_trans_cum_prob=p["prob_list_cum"],
        max_sensors=p["max_sensors"],
        max_sensors_null=p["max_sensors_null"],
        time_limit=p["time_limit"],
        time_limit_max=p["time_limit_max"],
    )
    return grid_environment(env_config={
        "qobj":                qobj,
        "time_limit_schedule": p["time_limit_schedule"],
        "time_limit_max":      p["time_limit_max"],
    })


def _make_env_for_tune(env_config):
    return make_env()


# Register so RLlib can find the env when loading the checkpoint
tune.register_env("GridTracking-eval", _make_env_for_tune)


# ===========================================================================
# Evaluation loop
# ===========================================================================

def evaluate_policy(algo, num_episodes=200, greedy=True, max_steps=500):
    """
    Roll out `num_episodes` episodes and return per-episode statistics.

    Parameters
    ----------
    algo        : loaded PPO instance
    num_episodes: number of evaluation episodes
    greedy      : if True, use deterministic (argmax) policy
    max_steps   : truncate episodes at this many steps (grid env has no
                  built-in step limit)
    """
    env = make_env()

    episode_rewards  = []
    episode_lengths  = []
    episode_found    = []
    episode_sensors  = []
    terminal_count   = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = ep_steps = ep_found = ep_sensors = 0
        done = False

        while not done and ep_steps < max_steps:
            action = algo.compute_single_action(obs, explore=not greedy)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(np.array(action).sum())

            if reward > 0:
                ep_found += 1

            done = terminated or truncated or (ep_steps >= max_steps)
            if terminated:
                terminal_count += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        episode_found.append(ep_found / max(ep_steps, 1))
        episode_sensors.append(ep_sensors / max(ep_steps, 1))

        if (ep + 1) % 50 == 0:
            print(f"    ... {ep + 1}/{num_episodes} episodes done", flush=True)

    return {
        "episode_rewards":     episode_rewards,
        "episode_lengths":     episode_lengths,
        "success_rates":       episode_found,
        "sensor_usage":        episode_sensors,
        "terminal_episodes":   terminal_count,
        "mean_success_rate":   float(np.mean(episode_found)),
        "std_success_rate":    float(np.std(episode_found)),
        "mean_total_reward":   float(np.mean(episode_rewards)),
        "std_total_reward":    float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_sensor_usage":   float(np.mean(episode_sensors)),
    }


# ===========================================================================
# Report printer
# ===========================================================================

def print_report(summary, num_episodes, checkpoint_path):
    sr      = summary["success_rates"]
    sensors = summary["sensor_usage"]
    rewards = summary["episode_rewards"]
    W = 62

    print("\n" + "=" * W)
    print("  GRID ENVIRONMENT — COMPREHENSIVE EVALUATION RESULTS")
    print("=" * W)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Episodes   : {num_episodes}")
    print("=" * W)

    print("\nBasic Performance Metrics:")
    print("-" * 35)
    print(f"  Mean Success Rate    : "
          f"{summary['mean_success_rate']:.4f} ± {summary['std_success_rate']:.4f}")
    print(f"  Mean Episode Length  : {summary['mean_episode_length']:.2f} steps")
    print(f"  Mean Total Reward    : "
          f"{summary['mean_total_reward']:.3f} ± {summary['std_total_reward']:.3f}")
    print(f"  Mean Sensors / Step  : {summary['mean_sensor_usage']:.3f}")
    print(f"  Terminal Episodes    : {summary['terminal_episodes']} / {num_episodes} "
          f"({summary['terminal_episodes'] / num_episodes * 100:.1f}%)")

    print("\nPerformance Distribution:")
    print("-" * 35)
    bins = [
        ("Excellent (>= 80%)", lambda r: r >= 0.80),
        ("Good      (60-79%)", lambda r: 0.60 <= r < 0.80),
        ("Fair      (40-59%)", lambda r: 0.40 <= r < 0.60),
        ("Poor      (< 40%)",  lambda r: r <  0.40),
    ]
    total = len(sr)
    for label, fn in bins:
        count = sum(1 for r in sr if fn(r))
        bar_fill = int(count / total * 20)
        bar = "\u2588" * bar_fill + "\u2591" * (20 - bar_fill)
        print(f"  {label} : {count:4d} eps  [{bar}] {count / total * 100:5.1f}%")

    print("\nEfficiency Analysis:")
    print("-" * 35)
    print(f"  Best  Success Rate   : {max(sr):.4f}")
    print(f"  Worst Success Rate   : {min(sr):.4f}")
    print(f"  Most  Efficient      : {min(sensors):.3f} sensors / step")
    print(f"  Least Efficient      : {max(sensors):.3f} sensors / step")
    eff = summary["mean_success_rate"] / max(summary["mean_sensor_usage"], 1e-9)
    print(f"  Efficiency Score     : {eff:.4f}  (success rate / sensor rate)")

    print("\nReward Percentiles:")
    print("-" * 35)
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct:2d} : {np.percentile(rewards, pct):+.3f}")

    print("\nOverall Assessment:")
    print("-" * 35)
    msr = summary["mean_success_rate"]
    msu = summary["mean_sensor_usage"]

    if msr >= 0.80:
        print("  Outstanding! Excellent tracking on the grid.")
    elif msr >= 0.70:
        print("  Very Good! Strong tracking with minor room for improvement.")
    elif msr >= 0.60:
        print("  Good. Solid performance; consider more training iterations.")
    elif msr >= 0.40:
        print("  Fair. Needs improvement — try tuning lr or batch size.")
    else:
        print("  Poor. Model may not have converged; retrain from scratch.")

    if msu < 0.30:
        print("  Highly efficient sensor usage.")
    elif msu < 0.60:
        print("  Good sensor efficiency.")
    else:
        print("  High sensor usage — consider penalising sensors more.")

    print("=" * W + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent_run194_ppo on the 10x10 grid tracking environment."
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint directory (default: agent_run194_ppo)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of evaluation episodes (default: %(default)s)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode before truncation (default: %(default)s)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of greedy")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"\nCheckpoint not found: {args.checkpoint}")
        sys.exit(1)

    p = TRAIN_PARAMS
    print("\n" + "=" * 62)
    print("  GRID ENVIRONMENT - POLICY EVALUATION")
    print("=" * 62)
    print(f"\n  Checkpoint  : {args.checkpoint}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Policy mode : {'stochastic' if args.stochastic else 'greedy'}")
    print(f"  Grid size   : {p['N']}x{p['N']}")
    print(f"  Max sensors : {p['max_sensors']}")
    print(f"  Time limit  : {p['time_limit']}")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": project_root}},
    )
    try:
        print(f"\n  Loading model...", flush=True)
        algo = PPO.from_checkpoint(args.checkpoint)
        print("  Model loaded successfully.")

        print(f"\n  Running evaluation ({args.episodes} episodes)...\n")
        summary = evaluate_policy(
            algo,
            num_episodes=args.episodes,
            greedy=not args.stochastic,
            max_steps=args.max_steps,
        )

        print_report(summary, args.episodes, args.checkpoint)

    except Exception as exc:
        print(f"\nEvaluation failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
