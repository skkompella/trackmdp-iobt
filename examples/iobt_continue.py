#!/usr/bin/env python3
"""
iobt_continue.py — Load a trained IoBT PPO checkpoint and continue training
on a new map topology.

Intended use
------------
A model is first trained on a "prior" environment (iobt_6node_env: fully-
connected 6-node graph with equal transition probabilities).  This script
loads that checkpoint and fine-tunes it on a *specific* scenario defined by
SCENARIO_MAP below, where the graph topology and/or transition biases differ
from the prior.  The hypothesis is that accuracy will rise faster than
training from scratch because the policy already knows how to do Track-MDP.

Defining the scenario
---------------------
Edit SCENARIO_MAP at the top of this file.  It must be a dict:
    {node_idx: [neighbour_idx, ...], ...}
for all nodes 0 … N_NODES-1.  Self-loops are NOT included — object_move()
adds the stay-in-place option automatically.

Checkpoint search order
-----------------------
    1. --checkpoint  (explicit path)
    2. ./runs/agent_run<N>_ppo/  highest checkpoint_XXXXXX/ subdir
    3. ./agent_run<N>_ppo/       highest checkpoint_XXXXXX/ subdir

Usage
-----
    python examples/iobt_continue.py --run 197
    python examples/iobt_continue.py --run 197 --iterations 500
    python examples/iobt_continue.py --checkpoint runs/agent_run197_ppo --iterations 200
    python examples/iobt_continue.py --run 197 --new-run 210 --eval-episodes 50

Output
------
    ./runs/agent_run<new_run>_ppo/    — new checkpoints every --eval-interval iters
    The original checkpoint is never modified.
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from src.core.iobt_6node_env import learning_grid_sarsa_0, TERMINAL_PROB
from src.core.gym_wrapper import grid_environment


# ===========================================================================
# ─── SCENARIO MAP ───────────────────────────────────────────────────────────
#
# Define the specific topology you want to continue-train on.
# Keys   = node indices  (0-indexed, must cover 0 … N_NODES-1)
# Values = neighbour list (nodes the object can move to from this node)
#
# Self-loops are added automatically inside iobt_6node_env.object_move(),
# so do NOT include a node in its own neighbour list.
#
# Example A — same as prior (fully connected, equal probs):
#   SCENARIO_MAP = {i: [j for j in range(6) if j != i] for i in range(6)}
#
# Example B — sparse ring topology (node i -> i-1, i+1 mod 6):
#   SCENARIO_MAP = {i: [(i-1)%6, (i+1)%6] for i in range(6)}
#
# Example C — star topology (node 0 is hub, all others only connect to 0):
#   SCENARIO_MAP = {0: [1,2,3,4,5], 1:[0], 2:[0], 3:[0], 4:[0], 5:[0]}
#
# ===========================================================================

SCENARIO_MAP: dict = {
    # ── Edit this section to define your scenario ──────────────────────────
    # Default: fully connected (same as prior — change to your specific map)
    0: [1, 2, 3, 4, 5],
    1: [0, 2, 3, 4, 5],
    2: [0, 1, 3, 4, 5],
    3: [0, 1, 2, 4, 5],
    4: [0, 1, 2, 3, 5],
    5: [0, 1, 2, 3, 4],
}

N_NODES = len(SCENARIO_MAP)   # must be 6 for iobt_6node_env


# ===========================================================================
# Config defaults — must match original training
# ===========================================================================

DEFAULTS = {
    "run_number":        197,
    "N":                 3,            # 3×3 backing grid for 6-node env
    "num_trans":         6,
    "max_sensors":       6,
    "max_sensors_null":  6,
    "time_limit":        1,
    "time_limit_max":    1,
    "training_iterations": 500,
    "eval_interval":     10,
    "eval_episodes":     100,
    "lr":                1e-4,
    "train_batch_size":  4000,
    "num_workers":       4,
}


# ===========================================================================
# Augment-vector helpers (must match gym_wrapper.py)
# ===========================================================================

def _build_augment_vecs(time_limit_max):
    augment_vec_1 = [(2 * i + 3) ** 2 for i in range(time_limit_max + 1)]
    augment_vec   = [0]
    for v in augment_vec_1:
        augment_vec.append(augment_vec[-1] + v)
    return augment_vec_1, augment_vec

_AUGMENT_VEC_1, _AUGMENT_VEC = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AUGMENT_VEC[-1]           # 34
MAX_ACTION_SZ      = (2 * DEFAULTS["time_limit_max"] + 3) ** 2   # 25


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _is_checkpoint(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "rllib_checkpoint.json")) or
            os.path.isfile(os.path.join(path, "algorithm_state.pkl")))


def find_latest_checkpoint(run_number: int, save_dir: str | None = None) -> str | None:
    """
    Return the path of the highest-numbered checkpoint for run_number,
    or None if none is found.
    """
    candidates = []
    if save_dir:
        candidates.append(save_dir)
    else:
        runs = os.path.join(_PROJECT_ROOT, "runs")
        candidates += [
            os.path.join(runs,       f"agent_run{run_number}_ppo"),
            os.path.join(_PROJECT_ROOT, f"agent_run{run_number}_ppo"),
        ]

    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        if _is_checkpoint(directory):
            return directory

        subdirs = []
        for item in os.listdir(directory):
            full = os.path.join(directory, item)
            if os.path.isdir(full) and item.startswith("checkpoint_"):
                try:
                    subdirs.append((int(item.split("_")[1]), full))
                except (ValueError, IndexError):
                    pass
        if subdirs:
            subdirs.sort(key=lambda x: x[0])
            return subdirs[-1][1]

    return None


# ===========================================================================
# Evaluation (mirrors iobt_eval.py)
# ===========================================================================

def evaluate_policy(algo, env, cfg: dict, num_episodes: int) -> dict:
    """
    Greedy rollout evaluation on the scenario environment.

    Returns dict: tracking_accuracy, mean_reward, std_reward,
                  mean_length, mean_sensors_used.
    """
    N             = cfg["N"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors   = cfg["max_sensors"]
    max_steps     = cfg.get("max_episode_steps", 1000)
    missing_state = N * N * (time_limit_max + 1) + 1

    augment_vec_1, augment_vec = _build_augment_vecs(time_limit_max)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < max_steps:
            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                action_vec[:augment_vec[time_delay]] = actions_list

            if not was_missing:
                state_pos  = current_state // (time_limit + 1)
                state_time = current_state %  (time_limit + 1)
            else:
                state_pos, state_time = N * N, 0

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
                active = np.where(action_sensors == 1)[0][:max_sensors]
                action_sensors = np.zeros(num_sensors, dtype=int)
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
                td_ac        = augment_vec_1[time_delay - 1]
                action_bool  = [1] * MAX_ACTION_SZ if was_missing else list(action_full)
                actions_list = actions_list + list(np.array(action_bool)[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        ep_lengths.append(ep_steps)
        sensors_ep.append(ep_sensors / max(ep_steps, 1))
        total_steps += ep_steps

    accuracy = total_found / max(total_steps, 1)
    return {
        "tracking_accuracy": accuracy,
        "mean_reward":       float(np.mean(total_rewards)),
        "std_reward":        float(np.std(total_rewards)),
        "mean_length":       float(np.mean(ep_lengths)),
        "mean_sensors_used": float(np.mean(sensors_ep)),
    }


# ===========================================================================
# Continue-training entry point
# ===========================================================================

def continue_training(cfg: dict, checkpoint_path: str, new_run: int) -> None:
    """
    Load checkpoint and run further PPO training iterations on SCENARIO_MAP.
    """
    N             = cfg["N"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    missing_state = N * N * (time_limit_max + 1) + 1

    state_trans_cum_prob = [
        round((i + 1) / cfg["num_trans"], 4)
        for i in range(cfg["num_trans"])
    ]

    # Build environment for eval (not used by RLlib workers directly)
    qobj = learning_grid_sarsa_0(
        run_number        = cfg["run_number"],
        N                 = N,
        num_trans         = cfg["num_trans"],
        state_trans_cum_prob = state_trans_cum_prob,
        max_sensors       = cfg["max_sensors"],
        max_sensors_null  = cfg["max_sensors_null"],
        time_limit        = time_limit,
        time_limit_max    = time_limit_max,
    )

    # Patch the env's map with SCENARIO_MAP so the eval env reflects it
    qobj.grid_env._node_dests, qobj.grid_env._node_cum_probs = \
        _build_scenario_tables(SCENARIO_MAP)
    qobj.grid_env.obj_trans_matrix = _build_scenario_matrix(
        SCENARIO_MAP, N * N, cfg["num_trans"]
    )

    # Save directory for new checkpoints
    save_dir = os.path.join(_PROJECT_ROOT, "runs", f"agent_run{new_run}_ppo")
    os.makedirs(save_dir, exist_ok=True)

    # RLlib env config — gym_wrapper.grid_environment reads these
    env_config = {
        "run_number":           cfg["run_number"],
        "N":                    N,
        "num_trans":            cfg["num_trans"],
        "state_trans_cum_prob": state_trans_cum_prob,
        "max_sensors":          cfg["max_sensors"],
        "max_sensors_null":     cfg["max_sensors_null"],
        "time_limit":           time_limit,
        "time_limit_max":       time_limit_max,
        "missing_state":        missing_state,
        "qobj":                 qobj,
        # Pass scenario map so gym_wrapper workers use it
        "scenario_map":         SCENARIO_MAP,
    }

    # Rebuild PPO config to match original training
    ppo_cfg = (
        PPOConfig()
        .environment(grid_environment, env_config=env_config)
        .framework("torch")
        .training(
            lr               = cfg["lr"],
            train_batch_size = cfg["train_batch_size"],
            # Keep all other hyperparams at RLlib defaults unless overridden
        )
        .rollouts(
            num_rollout_workers = cfg["num_workers"],
            rollout_fragment_length = "auto",
        )
        .evaluation(
            evaluation_interval  = None,    # manual eval below
            evaluation_num_workers = 0,
        )
        # normalize_actions=False is critical — MultiDiscrete + normalisation
        # corrupts actions (known training bug documented in transcript)
        .experimental(_disable_preprocessor_api=False)
    )
    ppo_cfg.normalize_actions = False

    print("\nRestoring from checkpoint …")
    algo = PPO(config=ppo_cfg)
    algo.restore(checkpoint_path)
    print(f"Checkpoint restored: {checkpoint_path}")

    # ── Print scenario summary ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SCENARIO MAP")
    print("=" * 65)
    for node, nbrs in sorted(SCENARIO_MAP.items()):
        dests  = [node] + nbrs
        n      = len(dests)
        p_each = (1.0 - TERMINAL_PROB) / n
        print(f"  node {node+1:2d}  neighbours={[n+1 for n in nbrs]:<30s}"
              f"  p_each={p_each:.4f}")
    print("=" * 65)

    # ── Baseline eval before any new training ─────────────────────────────
    print(f"\nBaseline evaluation ({cfg['eval_episodes']} episodes) …")
    baseline = evaluate_policy(
        algo, qobj.grid_env, cfg, cfg["eval_episodes"]
    )
    _print_metrics("BASELINE", baseline)

    best_accuracy = baseline["tracking_accuracy"]
    best_ckpt     = checkpoint_path

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nContinuing training for {cfg['training_iterations']} iterations …")
    print(f"Saving to: {save_dir}")
    print(f"Evaluating every {cfg['eval_interval']} iterations\n")

    for i in range(1, cfg["training_iterations"] + 1):
        result = algo.train()

        reward_mean = result.get("episode_reward_mean", float("nan"))
        ep_len_mean = result.get("episode_len_mean",    float("nan"))
        print(
            f"  iter {i:5d}/{cfg['training_iterations']}  "
            f"reward={reward_mean:+8.3f}  "
            f"ep_len={ep_len_mean:7.1f}"
        )

        if i % cfg["eval_interval"] == 0 or i == cfg["training_iterations"]:
            metrics = evaluate_policy(
                algo, qobj.grid_env, cfg, cfg["eval_episodes"]
            )
            _print_metrics(f"EVAL @ iter {i}", metrics)

            # Save checkpoint every eval
            ckpt = algo.save(save_dir)
            print(f"  Checkpoint saved → {ckpt}")

            # Track best
            if metrics["tracking_accuracy"] > best_accuracy:
                best_accuracy = metrics["tracking_accuracy"]
                best_ckpt     = ckpt
                print(f"  ★ New best accuracy: {best_accuracy:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CONTINUE-TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Source checkpoint  : {checkpoint_path}")
    print(f"  New checkpoints    : {save_dir}")
    print(f"  Best accuracy      : {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"  Best checkpoint    : {best_ckpt}")
    print(f"  Baseline accuracy  : {baseline['tracking_accuracy']:.4f}")
    delta = best_accuracy - baseline["tracking_accuracy"]
    print(f"  Improvement        : {delta:+.4f}")
    print("=" * 65)

    algo.stop()


# ===========================================================================
# Scenario helpers — patch iobt_6node_env with SCENARIO_MAP at runtime
# ===========================================================================

def _build_scenario_tables(scenario_map: dict):
    """
    Build per-node (dests, cum_probs) tables from scenario_map.
    Same math as iobt_6node_env._build_equal_prob_tables().
    """
    node_dests     = {}
    node_cum_probs = {}
    for node, nbrs in scenario_map.items():
        dests   = [node] + nbrs
        n       = len(dests)
        p_each  = (1.0 - TERMINAL_PROB) / n
        cum     = np.array([p_each * (i + 1) for i in range(n)])
        node_dests[node]     = dests
        node_cum_probs[node] = cum
    return node_dests, node_cum_probs


def _build_scenario_matrix(scenario_map: dict, n_sq: int, num_trans: int) -> list:
    """
    Build the padded obj_trans_matrix from scenario_map.
    Dead cells (indices not in scenario_map) stay put.
    """
    matrix = []
    for i in range(n_sq):
        if i in scenario_map:
            row = [i] + scenario_map[i]
            while len(row) < num_trans:
                row.append(i)
            matrix.append(row[:num_trans])
        else:
            matrix.append([i] * num_trans)
    matrix.append([n_sq] * num_trans)   # terminal row
    return matrix


# ===========================================================================
# Metrics printer
# ===========================================================================

def _print_metrics(label: str, m: dict) -> None:
    print(f"\n  ── {label} ──")
    print(f"     Tracking accuracy : {m['tracking_accuracy']:.4f}"
          f"  ({m['tracking_accuracy']*100:.2f}%)")
    print(f"     Mean reward       : {m['mean_reward']:+.3f}"
          f"  ± {m['std_reward']:.3f}")
    print(f"     Mean ep length    : {m['mean_length']:.1f} steps")
    print(f"     Mean sensors/step : {m['mean_sensors_used']:.2f}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained IoBT PPO checkpoint and continue training "
            "on the topology defined by SCENARIO_MAP in this file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow
--------
  1. Edit SCENARIO_MAP at the top of this file to define your new topology.
  2. Run this script pointing at the prior (fully-connected) checkpoint.
  3. The script evaluates the baseline, trains further, and saves new ckpts.

Examples
--------
  python examples/iobt_continue.py --run 197
  python examples/iobt_continue.py --run 197 --iterations 500 --new-run 210
  python examples/iobt_continue.py \\
      --checkpoint runs/agent_run197_ppo/checkpoint_000050 \\
      --new-run 210 --iterations 300 --eval-episodes 50
        """
    )
    parser.add_argument(
        "--run", type=int, default=DEFAULTS["run_number"],
        help=f"Run number of the source checkpoint to load "
             f"(default: {DEFAULTS['run_number']})"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit path to checkpoint directory (overrides --run search)"
    )
    parser.add_argument(
        "--new-run", type=int, default=None,
        help="Run number for the output checkpoints (default: --run + 10)"
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULTS["training_iterations"],
        help=f"Number of additional training iterations "
             f"(default: {DEFAULTS['training_iterations']})"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=DEFAULTS["eval_interval"],
        help=f"Evaluate and checkpoint every N iterations "
             f"(default: {DEFAULTS['eval_interval']})"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=DEFAULTS["eval_episodes"],
        help=f"Episodes per evaluation (default: {DEFAULTS['eval_episodes']})"
    )
    parser.add_argument(
        "--max-sensors", type=int, default=None,
        help="Override max_sensors from config"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULTS["num_workers"],
        help=f"Number of RLlib rollout workers (default: {DEFAULTS['num_workers']})"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULTS["lr"],
        help=f"Learning rate (default: {DEFAULTS['lr']})"
    )
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["training_iterations"] = args.iterations
    cfg["eval_interval"]       = args.eval_interval
    cfg["eval_episodes"]       = args.eval_episodes
    cfg["num_workers"]         = args.workers
    cfg["lr"]                  = args.lr
    if args.max_sensors is not None:
        cfg["max_sensors"] = args.max_sensors

    new_run = args.new_run if args.new_run is not None else args.run + 10

    # Validate SCENARIO_MAP
    if set(SCENARIO_MAP.keys()) != set(range(N_NODES)):
        print(f"[ERROR] SCENARIO_MAP must have keys 0 … {N_NODES-1}. "
              f"Got: {sorted(SCENARIO_MAP.keys())}")
        sys.exit(1)
    for node, nbrs in SCENARIO_MAP.items():
        if node in nbrs:
            print(f"[ERROR] SCENARIO_MAP node {node} includes itself in neighbours. "
                  f"Remove self-loops — object_move() adds stay-in-place automatically.")
            sys.exit(1)

    # Locate checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_latest_checkpoint(args.run)
        if checkpoint_path is None:
            print(f"[ERROR] No checkpoint found for run {args.run}.")
            print(f"  Train first with iobt_new_training.py (or similar).")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  IoBT PPO — CONTINUE TRAINING")
    print("=" * 65)
    print(f"  Source run       : {args.run}")
    print(f"  Source checkpoint: {checkpoint_path}")
    print(f"  New run          : {new_run}")
    print(f"  Iterations       : {cfg['training_iterations']}")
    print(f"  Eval interval    : {cfg['eval_interval']}")
    print(f"  Eval episodes    : {cfg['eval_episodes']}")
    print(f"  Learning rate    : {cfg['lr']}")
    print(f"  Max sensors      : {cfg['max_sensors']}")
    print(f"  Workers          : {cfg['num_workers']}")

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT}},
        )

    try:
        continue_training(cfg, checkpoint_path, new_run)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()