#!/usr/bin/env python3
"""
training_grid_rect.py — Training and evaluation for a rectangular M×N grid.
Supports PPO (default) and Rainbow DQN via --algo.

Default grid: 2 rows x 3 cols.  Change --nrows / --ncols for other shapes.

Algorithm notes
---------------
PPO   — uses MultiDiscrete([2]*25) action space (grid_environment_rect).
        One binary decision per sensor cell in the max window.

Rainbow DQN — uses Discrete(N_ACTIONS) action space (grid_environment_dqn).
        Actions are pre-enumerated sensor subsets (combinations of up to
        max_sensors cells).  Includes C51 (distributional RL), NoisyNets,
        Dueling networks, Double DQN, N-step returns, and Prioritized replay.
        NOTE: DQN action spaces grow with max_sensors.  For tractability,
        use --max-sensors <= 4 with Rainbow DQN on small grids.

Usage
-----
    python examples/training_grid_rect.py                          # PPO, 2x3
    python examples/training_grid_rect.py --algo rainbow           # Rainbow DQN, 2x3
    python examples/training_grid_rect.py --algo rainbow --max-sensors 3
    python examples/training_grid_rect.py --nrows 3 --ncols 4 --run 201
    python examples/training_grid_rect.py --eval-only --run 200
    python examples/training_grid_rect.py --eval-only --algo rainbow --run 201
"""

import os
import sys
import argparse
from itertools import combinations
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers import ReplayBuffer

from src.core.grid_env_rect    import learning_grid_sarsa_0
from src.core.grid_wrapper_rect import grid_environment_rect
from src.core.gym_wrapper_dqn  import grid_environment_dqn, build_action_map


# ---------------------------------------------------------------------------
# Monkey-patch for Ray 2.54.0: after config.validate() resolves the replay
# buffer "type" string to the actual class, the old-stack check
#   if "EpisodeReplayBuffer" in config[...]["type"]
# raises TypeError because `in` doesn't work on a class object.
# We fix it by normalising `type` to its name string before the check.
# ---------------------------------------------------------------------------
def _patched_create_local_replay_buffer_if_necessary(self, config):
    if not config.get("replay_buffer_config") or config["replay_buffer_config"].get(
        "no_local_replay_buffer"
    ):
        return
    buf_type = config["replay_buffer_config"]["type"]
    type_name = buf_type if isinstance(buf_type, str) else buf_type.__name__
    if "EpisodeReplayBuffer" in type_name:
        config["replay_buffer_config"][
            "metrics_num_episodes_for_smoothing"
        ] = self.config.metrics_num_episodes_for_smoothing
    return from_config(ReplayBuffer, config["replay_buffer_config"])

Algorithm._create_local_replay_buffer_if_necessary = (
    _patched_create_local_replay_buffer_if_necessary
)


# ===========================================================================
# Defaults
# ===========================================================================

DEFAULTS = {
    "run_number":          201,
    "algo":                "ppo",
    "nrows":               2,
    "ncols":               3,
    "num_trans":           4,
    "max_sensors":         4,           # kept low for DQN tractability
    "max_sensors_null":    4,
    "time_limit":          1,
    "time_limit_max":      1,
    "training_iterations": 200,
    "eval_interval":       10,
    "eval_episodes":       100,
    # PPO
    "ppo_num_workers":     4,
    "ppo_lr":              1e-4,
    "ppo_train_batch":     4000,
    # Rainbow DQN
    "dqn_num_workers":     4,
    "dqn_lr":              5e-4,
    "dqn_train_batch":     64,
    "dqn_n_step":          3,
    "dqn_num_atoms":       51,
    "dqn_v_min":           -200.0,
    "dqn_v_max":           200.0,
    "dqn_buffer_size":     50_000,
    "dqn_alpha":           0.6,
    "dqn_beta":            0.4,
}


# ===========================================================================
# Augment vector helpers
# ===========================================================================

def _build_augment_vecs(time_limit_max):
    v1 = [(2 * i + 3) ** 2 for i in range(time_limit_max + 1)]
    v  = [0]
    for x in v1:
        v.append(v[-1] + x)
    return v1, v

_AV1, _AV       = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AV[-1]                                # 34
MAX_ACTION_SZ      = (2 * DEFAULTS["time_limit_max"] + 3) ** 2  # 25


# ===========================================================================
# Evaluation — PPO (MultiDiscrete)
# ===========================================================================

def _evaluate_ppo(algo, env, cfg, num_episodes):
    n_cells        = cfg["nrows"] * cfg["ncols"]
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors    = cfg["max_sensors"]
    missing_state  = n_cells * (time_limit_max + 1) + 1
    av1, av        = _build_augment_vecs(time_limit_max)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < 1000:
            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                action_vec[:av[time_delay]] = actions_list

            state_pos  = n_cells if was_missing else current_state // (time_limit + 1)
            state_time = 0       if was_missing else current_state %  (time_limit + 1)
            obs         = (state_pos, state_time, np.array(action_vec, dtype=np.int64))
            action_full = algo.compute_single_action(obs, explore=False)

            if not was_missing:
                action_clip    = np.array(action_full[-num_sensors:], dtype=int)
                action_sensors = np.multiply(
                    action_clip,
                    env.valid_q_indices_dict[time_delay][current_state])
                obj_rel_pos, obj_in_window = env.realign_obj(
                    env.object_pos, current_state, time_delay)
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
                td_ac        = av1[time_delay - 1]
                action_bool  = [1] * MAX_ACTION_SZ if was_missing else list(action_full)
                actions_list = actions_list + list(np.array(action_bool)[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        ep_lengths.append(ep_steps)
        sensors_ep.append(ep_sensors / max(ep_steps, 1))
        total_steps += ep_steps

    return _metrics(total_found, total_steps, total_rewards, ep_lengths, sensors_ep)


# ===========================================================================
# Evaluation — Rainbow DQN (Discrete)
# ===========================================================================

def _evaluate_dqn(algo, env, cfg, num_episodes):
    n_cells        = cfg["nrows"] * cfg["ncols"]
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors    = cfg["max_sensors"]
    missing_state  = n_cells * (time_limit_max + 1) + 1
    max_window_sz  = (2 * time_limit_max + 3) ** 2
    av1, av        = _build_augment_vecs(time_limit_max)
    action_map     = build_action_map(max_window_sz, max_sensors)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(num_episodes):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < 1000:
            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                action_vec[:av[time_delay]] = actions_list

            state_pos  = n_cells if was_missing else current_state // (time_limit + 1)
            state_time = 0       if was_missing else current_state %  (time_limit + 1)
            obs        = (state_pos, state_time, np.array(action_vec, dtype=np.int64))
            action_idx = int(algo.compute_single_action(obs, explore=False))

            if was_missing:
                action_bool    = [1] * max_window_sz
                action_sensors = np.ones(num_sensors, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1
            else:
                action_bool    = action_map[action_idx].tolist()
                action_sensors = np.array(action_bool[-num_sensors:], dtype=int)
                action_sensors = np.multiply(
                    action_sensors,
                    env.valid_q_indices_dict[time_delay][current_state])
                obj_rel_pos, obj_in_window = env.realign_obj(
                    env.object_pos, current_state, time_delay)

            obj_detected = int(obj_in_window == 1 and
                               action_sensors[int(obj_rel_pos)] == 1)
            if not was_missing and obj_detected:
                total_found += 1

            reward, next_state, terminal_flag, time_delay = \
                env.get_reward_next_state(current_state, action_bool, time_delay)

            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(np.array(action_bool).sum())

            if time_delay == 0:
                actions_list = []
            else:
                td_ac        = av1[time_delay - 1]
                actions_list = actions_list + list(action_bool[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        ep_lengths.append(ep_steps)
        sensors_ep.append(ep_sensors / max(ep_steps, 1))
        total_steps += ep_steps

    return _metrics(total_found, total_steps, total_rewards, ep_lengths, sensors_ep)


def _metrics(found, steps, rewards, lengths, sensors):
    return {
        "tracking_accuracy": found / max(steps, 1),
        "mean_reward":       float(np.mean(rewards)),
        "std_reward":        float(np.std(rewards)),
        "mean_length":       float(np.mean(lengths)),
        "mean_sensors_used": float(np.mean(sensors)),
    }


def evaluate_policy(algo, env, cfg, num_episodes):
    if cfg["algo"] == "rainbow":
        return _evaluate_dqn(algo, env, cfg, num_episodes)
    return _evaluate_ppo(algo, env, cfg, num_episodes)


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
# Algorithm config builders
# ===========================================================================

def _build_ppo(cfg, env_config):
    ppo_cfg = (
        PPOConfig()
        .environment(grid_environment_rect, env_config=env_config)
        .framework("torch")
        .training(lr=cfg["ppo_lr"],
                  train_batch_size=cfg["ppo_train_batch"],
                  grad_clip=30.0)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=cfg["ppo_num_workers"])
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
    )
    ppo_cfg.normalize_actions = False
    return ppo_cfg


def _build_rainbow(cfg, env_config):
    n_actions = len(build_action_map(
        (2 * cfg["time_limit_max"] + 3) ** 2,
        cfg["max_sensors"]
    ))
    print(f"  Rainbow DQN action space: {n_actions:,} discrete actions")

    dqn_cfg = (
        DQNConfig()
        .environment(grid_environment_dqn, env_config=env_config)
        .framework("torch")
        .training(
            lr               = cfg["dqn_lr"],
            train_batch_size = cfg["dqn_train_batch"],
            num_atoms        = cfg["dqn_num_atoms"],   # C51
            v_min            = cfg["dqn_v_min"],
            v_max            = cfg["dqn_v_max"],
            noisy            = True,                   # NoisyNets
            dueling          = True,                   # Dueling heads
            double_q         = True,                   # Double DQN
            n_step           = cfg["dqn_n_step"],      # N-step returns
            replay_buffer_config={
                "type":                     "MultiAgentPrioritizedReplayBuffer",
                "capacity":                 cfg["dqn_buffer_size"],
                "prioritized_replay_alpha": cfg["dqn_alpha"],
                "prioritized_replay_beta":  cfg["dqn_beta"],
                "prioritized_replay_eps":   1e-6,
            },
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=cfg["dqn_num_workers"])
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
    )
    return dqn_cfg


# ===========================================================================
# Training
# ===========================================================================

def train(cfg, eval_only=False, checkpoint=None, save_dir_override=None):
    nrows          = cfg["nrows"]
    ncols          = cfg["ncols"]
    n_cells        = nrows * ncols
    run_number     = cfg["run_number"]
    algo_name      = cfg["algo"]
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    save_dir       = save_dir_override or f"./agent_run{run_number}_{algo_name}"

    missing_state = n_cells * (time_limit_max + 1) + 1

    terminal_prob  = 0.005
    state_prob_run = 0.15
    state_trans_cum_prob = [
        i * (1 - terminal_prob - state_prob_run) / float(cfg["num_trans"] - 1)
        for i in range(1, cfg["num_trans"])
    ]
    state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run]

    print(f"\n  Grid       : {nrows}x{ncols} ({n_cells} cells)")
    print(f"  Algorithm  : {algo_name.upper()}")
    print(f"  max_sensors: {cfg['max_sensors']}")
    print(f"  missing_state = {missing_state}")

    qobj = learning_grid_sarsa_0(
        run_number, nrows, ncols, cfg["num_trans"], state_trans_cum_prob,
        cfg["max_sensors"], cfg["max_sensors_null"], time_limit, time_limit_max,
    )
    env_config = {
        "qobj":                qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":      time_limit_max,
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": project_root}})

    # Eval-only
    if eval_only:
        ckpt = checkpoint
        if ckpt is None:
            ckpt, search = find_latest_checkpoint(run_number, save_dir=save_dir)
            if ckpt is None:
                print(f"[ERROR] No checkpoint found. Train first.")
                return
        print(f"Loading checkpoint: {ckpt}")
        AlgoCls = DQN if algo_name == "rainbow" else PPO
        algo    = AlgoCls.from_checkpoint(ckpt)
        metrics = evaluate_policy(algo, qobj.grid_env, cfg, cfg["eval_episodes"])
        _print_metrics("EVAL", metrics)
        return

    # Build config + algo
    if algo_name == "rainbow":
        algo_cfg = _build_rainbow(cfg, env_config)
        AlgoCls  = DQN
    else:
        algo_cfg = _build_ppo(cfg, env_config)
        AlgoCls  = PPO

    ckpt, _ = find_latest_checkpoint(run_number, save_dir=save_dir)
    algo    = algo_cfg.build()
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}")
        algo.restore(ckpt)
    else:
        print("No existing checkpoint — training from scratch.")

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nTraining {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")

    best_accuracy = 0.0
    for i in range(1, cfg["training_iterations"] + 1):
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
                print(f"  * New best: {best_accuracy:.4f}")

    print(f"\n  Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    algo.stop()


def _print_metrics(label, m):
    print(f"\n  -- {label} --")
    print(f"     Accuracy    : {m['tracking_accuracy']:.4f}  ({m['tracking_accuracy']*100:.2f}%)")
    print(f"     Reward      : {m['mean_reward']:+.3f} +/- {m['std_reward']:.3f}")
    print(f"     Ep length   : {m['mean_length']:.1f} steps")
    print(f"     Sensors/step: {m['mean_sensors_used']:.2f}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Track-MDP on a rectangular MxN grid with PPO or Rainbow DQN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm notes
---------------
  PPO (default) -- MultiDiscrete action space, any max_sensors.
  Rainbow DQN   -- Discrete enumerated sensor subsets.
                   Action space size by max_sensors (window=25):
                     max_sensors=2:   325 actions
                     max_sensors=3: 2,625 actions
                     max_sensors=4: 15,275 actions
                   Recommended: --max-sensors 3 or 4 with Rainbow.

Examples
--------
  python examples/training_grid_rect.py
  python examples/training_grid_rect.py --algo rainbow --max-sensors 3
  python examples/training_grid_rect.py --nrows 3 --ncols 4 --run 201
  python examples/training_grid_rect.py --eval-only --algo rainbow --run 201
        """
    )

    # Grid
    parser.add_argument("--nrows", type=int, default=DEFAULTS["nrows"])
    parser.add_argument("--ncols", type=int, default=DEFAULTS["ncols"])
    parser.add_argument("--run",   type=int, default=DEFAULTS["run_number"])

    # Algorithm
    parser.add_argument("--algo", choices=["ppo", "rainbow"],
                        default=DEFAULTS["algo"],
                        help="Training algorithm: ppo (default) or rainbow")

    # Shared
    parser.add_argument("--max-sensors",    type=int,   default=DEFAULTS["max_sensors"],
                        help="Max sensors per step (keep <=4 for Rainbow)")
    parser.add_argument("--iterations",     type=int,   default=DEFAULTS["training_iterations"])
    parser.add_argument("--eval-interval",  type=int,   default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval-episodes",  type=int,   default=DEFAULTS["eval_episodes"])
    parser.add_argument("--eval-only",      action="store_true")
    parser.add_argument("--checkpoint",     type=str,   default=None)
    parser.add_argument("--save-dir",       type=str,   default=None)

    # PPO options
    ppo = parser.add_argument_group("PPO options")
    ppo.add_argument("--ppo-workers",   type=int,   default=DEFAULTS["ppo_num_workers"])
    ppo.add_argument("--ppo-lr",        type=float, default=DEFAULTS["ppo_lr"])
    ppo.add_argument("--ppo-batch",     type=int,   default=DEFAULTS["ppo_train_batch"])

    # Rainbow DQN options
    dqn = parser.add_argument_group("Rainbow DQN options")
    dqn.add_argument("--dqn-workers",   type=int,   default=DEFAULTS["dqn_num_workers"])
    dqn.add_argument("--dqn-lr",        type=float, default=DEFAULTS["dqn_lr"])
    dqn.add_argument("--dqn-batch",     type=int,   default=DEFAULTS["dqn_train_batch"])
    dqn.add_argument("--dqn-n-step",    type=int,   default=DEFAULTS["dqn_n_step"])
    dqn.add_argument("--dqn-atoms",     type=int,   default=DEFAULTS["dqn_num_atoms"],
                     help="C51 atoms (51=Rainbow, 1=standard DQN)")
    dqn.add_argument("--dqn-v-min",     type=float, default=DEFAULTS["dqn_v_min"])
    dqn.add_argument("--dqn-v-max",     type=float, default=DEFAULTS["dqn_v_max"])
    dqn.add_argument("--dqn-buffer",    type=int,   default=DEFAULTS["dqn_buffer_size"])

    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update({
        "run_number":          args.run,
        "algo":                args.algo,
        "nrows":               args.nrows,
        "ncols":               args.ncols,
        "max_sensors":         args.max_sensors,
        "max_sensors_null":    args.max_sensors,
        "training_iterations": args.iterations,
        "eval_interval":       args.eval_interval,
        "eval_episodes":       args.eval_episodes,
        "ppo_num_workers":     args.ppo_workers,
        "ppo_lr":              args.ppo_lr,
        "ppo_train_batch":     args.ppo_batch,
        "dqn_num_workers":     args.dqn_workers,
        "dqn_lr":              args.dqn_lr,
        "dqn_train_batch":     args.dqn_batch,
        "dqn_n_step":          args.dqn_n_step,
        "dqn_num_atoms":       args.dqn_atoms,
        "dqn_v_min":           args.dqn_v_min,
        "dqn_v_max":           args.dqn_v_max,
        "dqn_buffer_size":     args.dqn_buffer,
    })

    n_cells = cfg["nrows"] * cfg["ncols"]

    print("=" * 65)
    print(f"  TRACK-MDP  --  RECTANGULAR GRID  ({cfg['algo'].upper()})")
    print("=" * 65)
    print(f"  Run number  : {cfg['run_number']}")
    print(f"  Grid        : {cfg['nrows']}x{cfg['ncols']} ({n_cells} cells)")
    print(f"  Algorithm   : {cfg['algo'].upper()}")
    print(f"  Max sensors : {cfg['max_sensors']}")
    print(f"  Iterations  : {cfg['training_iterations']}")
    print(f"  Eval every  : {cfg['eval_interval']}")

    if cfg["algo"] == "rainbow":
        max_w  = (2 * cfg["time_limit_max"] + 3) ** 2
        n_acts = sum(len(list(combinations(range(max_w), k)))
                     for k in range(1, cfg["max_sensors"] + 1))
        print(f"  DQN actions : {n_acts:,}  (window={max_w}, max_sensors={cfg['max_sensors']})")

    eval_only = args.eval_only or (args.checkpoint is not None)
    print(f"  Mode        : {'eval only' if eval_only else 'train + eval'}")

    try:
        train(cfg, eval_only=eval_only,
              checkpoint=args.checkpoint,
              save_dir_override=args.save_dir)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()