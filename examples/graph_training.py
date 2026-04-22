#!/usr/bin/env python3
"""
IoBT Environment Training Script
==================================
Trains a PPO agent on the 10-node IoBT graph environment defined in
iobt_environment.py.

This file is a direct adaptation of graph_training.py.  The only
differences are:
  - Imports iobt_env / learning_iobt_sarsa / IOBT_ADJACENCY / NUM_IOBT_NODES
    instead of their graph_environment counterparts.
  - The Gymnasium wrapper wraps iobt_env (sensor window = full graph,
    action vector length = NUM_IOBT_NODES).
  - Checkpoint directory prefix is agent_iobt_run<N>_ppo.

Everything else — PPO config builder, RLlib version shims, training loop,
evaluation helpers, CLI — is identical to graph_training.py.

IoBT graph topology (1-indexed):
    1  -> 2, 8, 5
    2  -> 1, 8, 7, 3
    3  -> 2, 7, 4
    4  -> 3, 6, 7
    5  -> 6, 3, 4, 1, 2
    6  -> 5, 4
    7  -> 2, 3, 9
    8  -> 10, 9, 1, 2
    9  -> 10, 7
    10 -> 9, 8

Usage
-----
    python examples/iobt_training.py
    python examples/iobt_training.py --iterations 3000 --lr 5e-5
    python examples/iobt_training.py --no-interactive
    python examples/iobt_training.py --eval-only
"""

import os
import sys
import argparse
import time
import json
import math
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO

# ---------------------------------------------------------------------------
# IoBT environment — imports from src.core.graph_environment
# (the only block that differs from graph_training.py)
# ---------------------------------------------------------------------------
from src.core.graph_environment import (
    iobt_env,
    learning_iobt_sarsa,
    build_iobt_cum_probs,
    IOBT_ADJACENCY,
    NUM_IOBT_NODES,
)


# ===========================================================================
# Default hyper-parameters
# ===========================================================================

DEFAULTS: Dict[str, Any] = {
    # --- Environment ---
    "num_trans":        4,      # Stochastic transition slots per node
    "terminal_prob":    0.005,  # P(object enters terminal state)
    "stay_prob":        0.15,   # P(object stays at current node)
    "max_sensors":      4,      # Max sensors deployable per step
    "max_sensors_null": 4,      # Max sensors in missing/null state
    "time_limit":       1,      # Missed detections before missing state
    "time_limit_max":   1,      # Hard cap (sets missing_state sentinel)
    # --- Training ---
    "run_number":           200,
    "training_iterations":  2000,
    "eval_interval":        10,   # Evaluate every N iterations
    "eval_episodes":        100,  # Episodes per evaluation
    "checkpoint_interval":  100,
    # --- PPO / RLlib ---
    "learning_rate":        1e-4,
    "gamma":                0.99,
    "lambda_gae":           0.95,
    "clip_param":           0.2,
    "entropy_coeff":        0.01,
    "num_sgd_iter":         10,
    "sgd_minibatch_size":   128,
    "train_batch_size":     4000,
    "num_rollout_workers":  4,
    "num_envs_per_worker":  2,
    "fcnet_hiddens":        [256, 256],
    "fcnet_activation":     "relu",
}


# ===========================================================================
# Gymnasium wrapper around iobt_env
# ===========================================================================

class IoBTTrackingEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around :class:`iobt_env`.

    Observation
    -----------
    Flat vector of length NUM_IOBT_NODES + 2:
        [one_hot over nodes (NUM_IOBT_NODES dims),
         missing_state flag (1 dim),
         normalised time delay (1 dim)]

    Action
    ------
    MultiDiscrete([2] * NUM_IOBT_NODES) — one binary slot per graph node.
    The agent may activate up to max_sensors slots; extras are masked off.
    The sensor window is always the full graph (BFS reachability within
    time_delay+1 hops from the belief node), so the action vector length
    is fixed at NUM_IOBT_NODES regardless of time delay.

    Reward
    ------
    Passed through from iobt_env.get_reward_next_state.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = {**DEFAULTS, **(config or {})}
        self._cfg = cfg

        cum_probs = build_iobt_cum_probs(
            cfg["num_trans"], cfg["terminal_prob"], cfg["stay_prob"]
        )
        # missing_state must be strictly above the highest real encoded state:
        #   (NUM_IOBT_NODES-1)*(time_limit_max+1) + time_limit_max
        # Using N*N*(time_limit_max+1)+1 with N=10 creates 200 phantom states
        # that corrupt the sensor validity masks in get_valid_q_indices_dict.
        N = 10
        missing_state = NUM_IOBT_NODES * (cfg["time_limit_max"] + 1) + 1

        self._env = iobt_env(
            N=N,
            num_trans=cfg["num_trans"],
            state_trans_cum_prob=cum_probs,
            max_sensors=cfg["max_sensors"],
            max_sensors_null=cfg["max_sensors_null"],
            missing_state=missing_state,
            time_limit=cfg["time_limit"],
        )

        self._missing_state = missing_state
        self._time_limit    = cfg["time_limit"]
        self._max_sensors   = cfg["max_sensors"]

        # Action vector length = number of graph nodes (fixed, no spatial radius)
        self._action_len = NUM_IOBT_NODES

        # ----------------------------------------------------------------
        # Spaces
        # ----------------------------------------------------------------
        obs_dim = NUM_IOBT_NODES + 2   # one_hot + missing flag + delay
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # MultiDiscrete([2]*n) is equivalent to MultiBinary(n) but is
        # supported by RLlib's old API stack.
        self.action_space = spaces.MultiDiscrete([2] * self._action_len)

        self._current_state: int = missing_state
        self._time_delay:    int = 0
        self._step_count:    int = 0
        self._max_episode_steps: int = cfg.get("max_episode_steps", 200)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset_object_state()
        self._current_state = self._missing_state
        self._time_delay    = 0
        self._step_count    = 0
        return self._observe(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        action = np.array(action, dtype=int)
        # Enforce sensor budget — keep only the first max_sensors active bits
        if action.sum() > self._max_sensors:
            active_indices = np.where(action == 1)[0][: self._max_sensors]
            action = np.zeros_like(action)
            action[active_indices] = 1

        # Capture is_missing BEFORE the step — obj_found was computed using
        # self._current_state, so the filter must use the same state.
        was_missing = (self._current_state == self._missing_state)

        reward, next_state, terminal_obj, next_delay, obj_found = \
            self._env.get_reward_next_state(
                self._current_state, action, self._time_delay
            )

        self._current_state = next_state
        self._time_delay    = next_delay

        terminated = bool(terminal_obj)
        truncated  = self._step_count >= self._max_episode_steps

        return self._observe(), float(reward), terminated, truncated, {
            "obj_found":      int(obj_found),
            "is_missing":     int(was_missing),
        }

    def _observe(self) -> np.ndarray:
        """
        [one_hot_node | missing_flag | normalised_delay]
        Slot NUM_IOBT_NODES is the missing-state indicator.
        """
        obs = np.zeros(NUM_IOBT_NODES + 2, dtype=np.float32)
        if self._current_state == self._missing_state:
            obs[NUM_IOBT_NODES] = 1.0
        else:
            node = self._current_state // (self._time_limit + 1)
            node = min(node, NUM_IOBT_NODES - 1)
            obs[node] = 1.0
        obs[NUM_IOBT_NODES + 1] = min(
            self._time_delay / max(self._time_limit, 1), 1.0
        )
        return obs

    def render(self):
        pos = self._env.object_pos
        node_str = str(pos + 1) if pos < NUM_IOBT_NODES else "TERMINAL"
        print(
            f"[IoBTTrackingEnv] object=node{node_str}  "
            f"state={self._current_state}  delay={self._time_delay}"
        )


# ---------------------------------------------------------------------------
# Register with Ray Tune — propagates to all remote worker processes
# ---------------------------------------------------------------------------
def _make_iobt_env(env_config):
    return IoBTTrackingEnv(env_config)

tune.register_env("IoBTTracking-v0", _make_iobt_env)


# ===========================================================================
# Evaluation helpers
# ===========================================================================

EVAL_SEED = 42   # Fixed seed used for every evaluation run

def evaluate_policy(
    algo: PPO,
    cfg: Dict[str, Any],
    num_episodes: int = 100,
) -> Dict[str, float]:
    """
    Run num_episodes greedy rollouts and return aggregate metrics.

    The evaluation environment is seeded with EVAL_SEED before construction
    so that the obj_trans_matrix (sampled once at __init__ via np.random.choice)
    is identical across all evaluation calls.  Without this, each call builds a
    new random transition matrix and accuracy bounces wildly independent of the
    policy quality.
    """
    import random
    # Seed ONLY for environment construction — this fixes the obj_trans_matrix
    # (sampled once via np.random.choice in iobt_env.__init__) so that every
    # evaluation call faces identical transition dynamics.
    # After construction we immediately restore the RNG state so that the 100
    # evaluation episodes sample freely from the full state space.  Seeding the
    # episodes themselves would lock all 100 to the same object trajectories,
    # causing accuracy to plateau regardless of policy improvement.
    rng_state_np  = np.random.get_state()
    rng_state_py  = random.getstate()
    np.random.seed(EVAL_SEED)
    random.seed(EVAL_SEED)
    env = IoBTTrackingEnv({**DEFAULTS, **cfg})
    np.random.set_state(rng_state_np)
    random.setstate(rng_state_py)
    total_rewards, episode_lengths, sensors_per_ep, found_flags = [], [], [], []

    total_steps_global = 0
    total_found_global = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = ep_steps = ep_sensors = ep_found = 0
        done = False

        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(np.array(action).sum())
            # Only count detections in non-missing states — missing-state
            # "detections" are a re-acquisition mechanic (reward=0) and
            # inflate the metric without reflecting real tracking.
            if not info.get("is_missing", 0):
                ep_found   += info.get("obj_found", 0)
            done = terminated or truncated

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        sensors_per_ep.append(ep_sensors / max(ep_steps, 1))
        found_flags.append(ep_found / max(ep_steps, 1))
        total_steps_global += ep_steps
        total_found_global += ep_found

    # Note: total_steps_global includes all steps (missing + non-missing).
    # The numerator only counts non-missing detections, so accuracy reflects
    # the fraction of ALL steps where real tracking occurred — a strict metric.

    # Global accuracy: total detections / total timesteps across all episodes.
    # This is statistically stable regardless of episode-length variance.
    # Per-episode mean (mean of ratios) is dominated by short episodes and
    # oscillates even when the policy is steady.
    global_accuracy = total_found_global / max(total_steps_global, 1)

    return {
        "mean_reward":       float(np.mean(total_rewards)),
        "std_reward":        float(np.std(total_rewards)),
        "mean_length":       float(np.mean(episode_lengths)),
        "success_rate":      global_accuracy,
        "mean_sensors_used": float(np.mean(sensors_per_ep)),
    }


def print_eval_summary(metrics: Dict[str, float], iteration: int) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Evaluation at iteration {iteration}")
    print(f"{'─' * 50}")
    print(f"  Mean reward       : {metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Success rate      : {metrics['success_rate']:.3%}")
    print(f"  Mean episode len  : {metrics['mean_length']:.1f}")
    print(f"  Mean sensors/step : {metrics['mean_sensors_used']:.3f}")
    print(f"{'─' * 50}\n")


def _print_performance_assessment(metrics: Dict[str, float]) -> None:
    sr = metrics["success_rate"]
    su = metrics["mean_sensors_used"]

    print("Performance assessment:")
    if sr >= 0.80:
        print("  🌟  Outstanding! Excellent IoBT tracking.")
    elif sr >= 0.70:
        print("  🎯  Very good! Strong tracking with minor room for improvement.")
    elif sr >= 0.60:
        print("  👍  Good. Solid performance; consider more iterations.")
    else:
        print("  📈  Needs improvement. Try more iterations or tuning lr / batch size.")

    if su < 0.30:
        print("  ⚡  Highly efficient sensor usage.")
    elif su < 0.60:
        print("  🔋  Good sensor efficiency.")
    else:
        print("  ⚠️   High sensor usage — consider penalising sensors more.")
    print()


# ===========================================================================
# PPO config builder  (identical to graph_training.py)
# ===========================================================================

def _ppo_accepted_params() -> set:
    import inspect
    try:
        return set(inspect.signature(PPOConfig.training).parameters.keys())
    except Exception:
        return set()


def _resolve_alias(aliases: List[Tuple[str, Any]], accepted: set):
    for name, value in aliases:
        if not accepted or name in accepted:
            yield name, value
            return


def build_ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    accepted = _ppo_accepted_params()

    base = (
        PPOConfig()
        .environment(
            env="IoBTTracking-v0",
            env_config=cfg,
            # Disable action normalisation — it only applies to continuous
            # action spaces and corrupts MultiDiscrete actions, causing a
            # mismatch between training (where RLlib rescales) and evaluation
            # (where compute_single_action returns raw values).
            normalize_actions=False,
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        )
    )

    versioned: List[Tuple[str, Any]] = []
    for aliases in [
        [("lambda_",            cfg["lambda_gae"]),
         ("gae_lambda",         cfg["lambda_gae"])],
        [("clip_param",         cfg["clip_param"]),
         ("clip_epsilon",       cfg["clip_param"])],
        [("sgd_minibatch_size", cfg["sgd_minibatch_size"]),
         ("minibatch_size",     cfg["sgd_minibatch_size"])],
        [("num_sgd_iter",       cfg["num_sgd_iter"]),
         ("num_epochs",         cfg["num_sgd_iter"])],
    ]:
        for name, value in _resolve_alias(aliases, accepted):
            versioned.append((name, value))

    training_kwargs: Dict[str, Any] = dict(
        lr=cfg["learning_rate"],
        gamma=cfg["gamma"],
        entropy_coeff=cfg["entropy_coeff"],
        train_batch_size=cfg["train_batch_size"],
        model={
            "fcnet_hiddens":   cfg["fcnet_hiddens"],
            "fcnet_activation": cfg["fcnet_activation"],
        },
        **dict(versioned),
    )
    base = base.training(**training_kwargs)

    workers_set = False
    for method, kwargs in [
        ("env_runners", dict(num_env_runners=cfg["num_rollout_workers"],
                             num_envs_per_env_runner=cfg["num_envs_per_worker"])),
        ("rollouts",    dict(num_rollout_workers=cfg["num_rollout_workers"],
                             num_envs_per_worker=cfg["num_envs_per_worker"])),
    ]:
        if hasattr(base, method):
            try:
                base = getattr(base, method)(**kwargs)
                workers_set = True
                break
            except (TypeError, ValueError):
                continue

    if not workers_set:
        print("  WARNING: could not configure rollout workers — check RLlib version.")

    base = base.evaluation(evaluation_interval=None)
    return base


# ===========================================================================
# Checkpoint helpers  (identical to graph_training.py except dir prefix)
# ===========================================================================

def _checkpoint_dir(cfg: Dict[str, Any]) -> str:
    base = cfg.get("save_dir", f"./agent_iobt_run{cfg['run_number']}_ppo")
    os.makedirs(base, exist_ok=True)
    return base


def save_checkpoint(algo: PPO, cfg: Dict[str, Any], iteration: int) -> str:
    directory = _checkpoint_dir(cfg)
    path = algo.save(directory)
    print(f"  💾  Checkpoint saved → {path}  (iteration {iteration})")
    return path


def find_latest_checkpoint(cfg: Dict[str, Any]) -> Optional[str]:
    directory = _checkpoint_dir(cfg)
    checkpoints = []
    if os.path.isdir(directory):
        for item in os.listdir(directory):
            full = os.path.join(directory, item)
            if os.path.isdir(full) and item.startswith("checkpoint_"):
                try:
                    num = int(item.split("_")[1])
                    checkpoints.append((num, full))
                except (ValueError, IndexError):
                    pass
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def save_training_log(log: List[Dict], cfg: Dict[str, Any]) -> None:
    directory = _checkpoint_dir(cfg)
    log_path = os.path.join(directory, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  📄  Training log saved → {log_path}")


# ===========================================================================
# Main training loop  (identical to graph_training.py)
# ===========================================================================

def train(cfg: Optional[Dict[str, Any]] = None) -> Optional[PPO]:
    merged      = {**DEFAULTS, **(cfg or {})}
    total_iters = merged["training_iterations"]

    # ---- Header ----
    print("\n" + "=" * 70)
    print("  IoBT ENVIRONMENT — PPO TRAINING")
    print("=" * 70)
    print("\nIoBT graph topology (1-indexed nodes):")
    for node, neighbours in IOBT_ADJACENCY.items():
        print(f"    Node {node + 1:2d}  ->  {[n + 1 for n in neighbours]}")

    print("\nTraining configuration:")
    print("-" * 40)
    col_w = max(len(k) for k in merged) + 2
    for k, v in merged.items():
        print(f"  {k:<{col_w}}: {v}")
    print()

    # ---- Ray ----
    # Pass the project root as PYTHONPATH so that every remote worker
    # process can import src.* — workers spawn with a clean environment
    # and don't inherit the main process's sys.path.
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
        )

    # ---- Build algo ----
    ppo_config = build_ppo_config(merged)
    build_fn   = getattr(ppo_config, "build_algo", None) or ppo_config.build
    algo       = build_fn()

    print("\nAlgorithm built successfully.")
    try:
        local_env = algo.workers.local_worker().env
        print(f"Observation space : {local_env.observation_space}")
        print(f"Action space      : {local_env.action_space}")
    except Exception:
        _tmp = IoBTTrackingEnv(merged)
        print(f"Observation space : {_tmp.observation_space}")
        print(f"Action space      : {_tmp.action_space}")
    print(f"Saving to         : {_checkpoint_dir(merged)}\n")

    # ---- Training loop ----
    training_log:        List[Dict] = []
    tracking_accuracies: List[float] = []
    avg_sensors_list:    List[float] = []
    eval_iterations:     List[int]   = []
    t_start = time.time()

    try:
        for iteration in range(1, total_iters + 1):
            result = algo.train()

            # Handle metric location change across Ray versions (mirrors trainer.py)
            if "episode_reward_mean" in result:
                reward_mean = result["episode_reward_mean"]
            elif "env_runners" in result and "episode_reward_mean" in result["env_runners"]:
                reward_mean = result["env_runners"]["episode_reward_mean"]
            else:
                reward_mean = 0.0

            print(f"episode reward mean: {iteration}  {reward_mean}")

            training_log.append({
                "iteration":           iteration,
                "episode_reward_mean": reward_mean,
                "timesteps_total":     result.get("timesteps_total", 0),
            })

            # Periodic evaluation (matches trainer.py's every-50-steps block)
            if iteration % merged["eval_interval"] == 0:
                save_checkpoint(algo, merged, iteration)

                print(f"\n{'='*60}")
                print(f"EVALUATION AT ITERATION {iteration}")
                print(f"{'='*60}")

                eval_metrics = evaluate_policy(
                    algo, merged, num_episodes=merged["eval_episodes"]
                )

                tracking_acc  = eval_metrics["success_rate"]
                avg_sensors   = eval_metrics["mean_sensors_used"]

                tracking_accuracies.append(tracking_acc)
                avg_sensors_list.append(avg_sensors)
                eval_iterations.append(iteration)

                print(f"Tracking Accuracy: {tracking_acc:.4f} ({tracking_acc*100:.2f}%)")
                print(f"Average Sensors per Step: {avg_sensors:.2f}")
                print(f"{'='*60}\n")

                save_training_log(training_log, merged)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    except Exception as exc:
        print(f"\nTraining failed at iteration {iteration}: {exc}")
        raise

    finally:
        save_checkpoint(algo, merged, iteration)
        save_training_log(training_log, merged)

    # ---- Final summary (matches trainer.py's final block) ----
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL EVALUATION SUMMARY")
    print("="*60)

    if tracking_accuracies:
        print(f"Training iterations completed: {len(training_log)}")
        print(f"Evaluations performed: {len(tracking_accuracies)}")
        print(f"Best tracking accuracy: {max(tracking_accuracies):.4f} ({max(tracking_accuracies)*100:.2f}%)")
        print(f"Final tracking accuracy: {tracking_accuracies[-1]:.4f} ({tracking_accuracies[-1]*100:.2f}%)")
        print(f"Best avg sensors per step: {min(avg_sensors_list):.2f}")
        print(f"Final avg sensors per step: {avg_sensors_list[-1]:.2f}")
    else:
        print("No evaluations were performed during training.")

    print("="*60)
    print("\n Done")

    return algo


# ===========================================================================
# Evaluation-only entry point
# ===========================================================================

def evaluate_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    num_episodes: int = 200,
) -> None:
    merged = {**DEFAULTS, **(cfg or {})}

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(merged)

    if checkpoint_path is None:
        print("❌  No checkpoint found. Train a model first.")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    algo = PPO.from_checkpoint(checkpoint_path)

    print(f"\nRunning evaluation ({num_episodes} episodes)...")
    metrics = evaluate_policy(algo, merged, num_episodes=num_episodes)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Tracking Accuracy: {metrics['success_rate']:.4f} ({metrics['success_rate']*100:.2f}%)")
    print(f"Average Sensors per Step: {metrics['mean_sensors_used']:.2f}")
    print(f"Mean Reward: {metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
    print(f"{'='*60}\n")

    ray.shutdown()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on the 10-node IoBT graph environment."
    )
    parser.add_argument("--iterations", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--workers",    type=int,   default=None)
    parser.add_argument("--run",        type=int,   default=None)
    parser.add_argument("--save-dir",   type=str,   default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--no-interactive", action="store_true")
    return parser.parse_args()


def _interactive_config() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    print("\nInteractive configuration (press Enter to accept defaults):\n")
    fields = [
        ("training_iterations", "Training iterations", int),
        ("learning_rate",       "Learning rate",       float),
        ("num_rollout_workers", "Rollout workers",     int),
        ("max_sensors",         "Max sensors",         int),
        ("eval_interval",       "Eval interval",       int),
    ]
    for key, label, cast in fields:
        raw = input(f"  {label} [{DEFAULTS[key]}]: ").strip()
        if raw:
            try:
                overrides[key] = cast(raw)
            except ValueError:
                print(f"    ⚠  Invalid — using default ({DEFAULTS[key]}).")
    return overrides


def main() -> None:
    args = _parse_args()

    cli_overrides: Dict[str, Any] = {}
    if args.iterations is not None: cli_overrides["training_iterations"] = args.iterations
    if args.lr         is not None: cli_overrides["learning_rate"]        = args.lr
    if args.workers    is not None: cli_overrides["num_rollout_workers"]   = args.workers
    if args.run        is not None: cli_overrides["run_number"]            = args.run
    if args.save_dir   is not None: cli_overrides["save_dir"]              = args.save_dir

    interactive_overrides: Dict[str, Any] = {}
    if not args.no_interactive and not args.eval_only:
        try:
            interactive_overrides = _interactive_config()
        except (EOFError, KeyboardInterrupt):
            print("\n  (skipping interactive config)")

    merged = {**DEFAULTS, **interactive_overrides, **cli_overrides}

    if args.eval_only:
        evaluate_from_checkpoint(checkpoint_path=args.checkpoint, cfg=merged)
    else:
        algo = train(cfg=merged)
        if algo is not None and ray.is_initialized():
            ray.shutdown()


# ===========================================================================

if __name__ == "__main__":
    main()