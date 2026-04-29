#!/usr/bin/env python3
"""
train_respeaker.py — Fine-tune a pretrained PPO checkpoint on real respeaker
power data collected by mqtt_collector.py.

Mirrors finetune_deterministic.py exactly: load a source checkpoint, run a
baseline eval, then fine-tune using the same fast-convergence PPO
hyperparameters.  The only difference is that the object moves through the
real sequence of positions derived from respeaker power readings (highest-
power node = object location) rather than a deterministic circle.

The 2×3 grid (6 cells) maps 1-to-1 to the 6 IOBT nodes:
    orin_11 → cell 0   orin_12 → cell 1   orin_13 → cell 2
    orin_14 → cell 3   orin_15 → cell 4   orin_16 → cell 5

Usage
-----
    python examples/train_respeaker.py --run 201
    python examples/train_respeaker.py --run 201 --new-run 302 --iterations 300
    python examples/train_respeaker.py --eval-only --checkpoint runs/agent_run302_ppo
"""

import os
import sys
import argparse
import glob
import inspect
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from src.core.grid_env_rect import learning_grid_sarsa_0
from src.core.respeaker_env import RespeakerGridEnv, respeaker_grid_environment


# ===========================================================================
# Defaults  — matched to finetune_deterministic.py RECT_DEFAULTS
# ===========================================================================

DEFAULTS = {
    # Environment
    "run_number":      200,
    "new_run":         401,   # default: run_number + 1
    "nrows":           2,
    "ncols":           3,
    "num_trans":       4,
    "max_sensors":     6,
    "max_sensors_null": 6,
    "time_limit":      1,
    "time_limit_max":  1,
    # Fast-convergence PPO hyperparameters (identical to finetune_deterministic.py)
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
    # Training schedule
    "training_iterations": 200,
    "eval_interval":       1,
    "eval_episodes":       20,
    "max_ep_steps":        100,
    # Respeaker-specific
    "data_dir":            os.path.join(project_root, "data"),
}

def _build_augment_vecs(tlm):
    v1 = [(2 * i + 3) ** 2 for i in range(tlm + 1)]
    v  = [0]
    for x in v1:
        v.append(v[-1] + x)
    return v1, v

_AV1, _AV        = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AV[-1]
MAX_ACTION_SZ      = (2 * DEFAULTS["time_limit_max"] + 3) ** 2


# ===========================================================================
# Data loading
# ===========================================================================

def load_respeaker_positions(data_dir):
    """
    Scan data_dir for respeaker_power_*.npy files (excluding *_ts_* files).
    For each row (timestep) the ground-truth cell index = argmax of the 6
    node power values.  All sessions are concatenated in filename order.

    Returns
    -------
    positions : np.int32 array of shape (total_timesteps,)
    """
    data_dir = os.path.abspath(data_dir)
    files = sorted(
        f for f in glob.glob(os.path.join(data_dir, "respeaker_power_[0-9]*.npy"))
        if "_ts_" not in os.path.basename(f)
    )
    if not files:
        raise FileNotFoundError(f"No respeaker_power_*.npy files found in '{data_dir}'")

    all_positions = []
    for f in files:
        arr = np.load(f)          # float32 (steps, 6)
        gt  = arr.argmax(axis=1)  # int64  (steps,)  — cell 0..5
        all_positions.extend(gt.tolist())

    positions = np.array(all_positions, dtype=np.int32)
    counts = np.bincount(positions, minlength=6)
    print(f"  Loaded {len(files)} session(s) → {len(positions)} timesteps")
    print(f"  Cell counts [0-5]: {counts.tolist()}")
    return positions


# ===========================================================================
# Build qobj with respeaker env
# ===========================================================================

def _build_cum_probs(num_trans):
    terminal_prob  = 0.005
    state_prob_run = 0.15
    cum = [
        i * (1 - terminal_prob - state_prob_run) / float(num_trans - 1)
        for i in range(1, num_trans)
    ]
    cum.append(cum[-1] + state_prob_run)
    return cum


def build_qobj(cfg, real_positions):
    """
    Create a learning_grid_sarsa_0 whose grid_env is a RespeakerGridEnv
    that replays the supplied real position sequence.
    """
    cum_probs = _build_cum_probs(cfg["num_trans"])

    qobj = learning_grid_sarsa_0(
        cfg["run_number"], cfg["nrows"], cfg["ncols"],
        cfg["num_trans"], cum_probs,
        cfg["max_sensors"], cfg["max_sensors_null"],
        cfg["time_limit"], cfg["time_limit_max"],
    )
    qobj.grid_env = RespeakerGridEnv(
        cfg["nrows"], cfg["ncols"],
        cfg["num_trans"], cum_probs,
        cfg["max_sensors"], cfg["max_sensors_null"],
        qobj.missing_state, cfg["time_limit"],
        real_positions,
    )
    return qobj


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_policy(algo, env, cfg):
    n_cells        = cfg["nrows"] * cfg["ncols"]
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors    = cfg["max_sensors"]
    missing_state  = n_cells * (time_limit_max + 1) + 1
    av1, av        = _build_augment_vecs(time_limit_max)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(cfg["eval_episodes"]):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < cfg["max_ep_steps"]:
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
            total_steps += 1

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

def _is_checkpoint(path):
    return (os.path.isfile(os.path.join(path, "rllib_checkpoint.json")) or
            os.path.isfile(os.path.join(path, "algorithm_state.pkl")))


def find_latest_checkpoint(run_number, save_dir=None):
    candidates = []
    if save_dir:
        candidates.append(save_dir)
    else:
        runs = os.path.join(project_root, "runs")
        candidates += [
            os.path.join(runs, f"agent_run{run_number}_ppo"),
            f"./agent_run{run_number}_ppo",
        ]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        if _is_checkpoint(d):
            return d, d
        ckpts = []
        for item in os.listdir(d):
            full = os.path.join(d, item)
            if os.path.isdir(full) and item.startswith("checkpoint_"):
                try:
                    ckpts.append((int(item.split("_")[1]), full))
                except (ValueError, IndexError):
                    pass
        if ckpts:
            ckpts.sort(key=lambda x: x[0])
            return ckpts[-1][1], d
    return None, candidates[0] if candidates else f"./agent_run{run_number}_ppo"


# ===========================================================================
# Fine-tune
# ===========================================================================

def _print_metrics(label, m):
    print(f"\n  ── {label} ──")
    print(f"     Accuracy    : {m['tracking_accuracy']:.4f}  ({m['tracking_accuracy']*100:.2f}%)")
    print(f"     Reward      : {m['mean_reward']:+.3f} ± {m['std_reward']:.3f}")
    print(f"     Ep length   : {m['mean_length']:.1f} steps")
    print(f"     Sensors/step: {m['mean_sensors_used']:.2f}\n")


def finetune(cfg, real_positions, source_checkpoint, new_run, eval_only=False):
    time_limit_max = cfg["time_limit_max"]
    save_dir       = f"./agent_run{new_run}_ppo"

    qobj = build_qobj(cfg, real_positions)

    env_config = {
        "qobj":                qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":      time_limit_max,
        "max_ep_steps":        cfg["max_ep_steps"],
    }

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
        )

    if eval_only:
        print(f"Loading checkpoint: {source_checkpoint}")
        algo    = PPO.from_checkpoint(source_checkpoint)
        metrics = evaluate_policy(algo, qobj.grid_env, cfg)
        _print_metrics("EVAL", metrics)
        return

    # ── Build fast-convergence PPO config (identical to finetune_deterministic) ──
    try:
        _accepted = set(inspect.signature(PPOConfig.training).parameters.keys())
    except Exception:
        _accepted = set()

    def _pick(aliases):
        for name, val in aliases:
            if not _accepted or name in _accepted:
                return {name: val}
        return {}

    training_kwargs = dict(
        lr               = cfg["lr"],
        train_batch_size = cfg["train_batch_size"],
        entropy_coeff    = cfg["entropy_coeff"],
        vf_loss_coeff    = cfg["vf_loss_coeff"],
        grad_clip        = cfg["grad_clip"],
    )
    training_kwargs.update(_pick([("num_sgd_iter",      cfg["num_sgd_iter"]),
                                   ("num_epochs",         cfg["num_sgd_iter"])]))
    training_kwargs.update(_pick([("sgd_minibatch_size", cfg["sgd_minibatch_size"]),
                                   ("minibatch_size",     cfg["sgd_minibatch_size"])]))
    training_kwargs.update(_pick([("clip_param",         cfg["clip_param"]),
                                   ("clip_epsilon",        cfg["clip_param"])]))

    ppo_cfg = (
        PPOConfig()
        .environment(respeaker_grid_environment, env_config=env_config)
        .framework("torch")
        .training(**training_kwargs)
        .env_runners(
            num_env_runners         = cfg["num_workers"],
            rollout_fragment_length = cfg["rollout_fragment_length"],
        )
        .resources(num_gpus=0)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    ppo_cfg.normalize_actions = False

    print(f"Building PPO and restoring from: {source_checkpoint}")
    algo = ppo_cfg.build()
    algo.restore(source_checkpoint)
    print("Checkpoint restored.\n")

    os.makedirs(save_dir, exist_ok=True)

    # ── Baseline eval before any fine-tuning ──────────────────────────────
    print("Baseline (prior policy on respeaker data) ...")
    baseline     = evaluate_policy(algo, qobj.grid_env, cfg)
    _print_metrics("BASELINE", baseline)
    best_accuracy = baseline["tracking_accuracy"]
    best_ckpt     = source_checkpoint

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"Fine-tuning for {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")
    print(f"  {'iter':>5}  {'reward':>10}  {'accuracy':>10}  {'sensors':>8}  {'ep_len':>8}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    for i in range(1, cfg["training_iterations"] + 1):
        result      = algo.train()
        reward_mean = (result.get("episode_reward_mean") or
                       result.get("env_runners", {}).get("episode_reward_mean",
                                                          float("nan")))

        if i % cfg["eval_interval"] == 0 or i == cfg["training_iterations"]:
            metrics = evaluate_policy(algo, qobj.grid_env, cfg)
            print(
                f"  {i:5d}  {reward_mean:+10.3f}  "
                f"{metrics['tracking_accuracy']:10.4f}  "
                f"{metrics['mean_sensors_used']:8.2f}  "
                f"{metrics['mean_length']:8.1f}"
            )
            if metrics["tracking_accuracy"] > best_accuracy:
                best_accuracy = metrics["tracking_accuracy"]
                best_ckpt     = algo.save(save_dir)
                print(f"         ★ New best {best_accuracy:.4f} — saved")
        else:
            print(f"  {i:5d}  {reward_mean:+10.3f}")

    final_ckpt = algo.save(save_dir)

    print(f"\n{'='*65}")
    print("  FINE-TUNE COMPLETE")
    print(f"{'='*65}")
    print(f"  Source checkpoint : {source_checkpoint}")
    print(f"  New checkpoints   : {save_dir}")
    print(f"  Baseline accuracy : {baseline['tracking_accuracy']:.4f}")
    print(f"  Best accuracy     : {best_accuracy:.4f}  ({best_accuracy*100:.2f}%)")
    print(f"  Best checkpoint   : {best_ckpt}")
    print(f"  Final checkpoint  : {final_ckpt}")
    delta = best_accuracy - baseline["tracking_accuracy"]
    print(f"  Improvement       : {delta:+.4f}")
    print(f"{'='*65}")

    algo.stop()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a PPO checkpoint on real respeaker power data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python examples/train_respeaker.py --run 201
  python examples/train_respeaker.py --run 201 --new-run 302 --iterations 300
  python examples/train_respeaker.py --eval-only --checkpoint runs/agent_run302_ppo
        """,
    )
    parser.add_argument("--run",           type=int,   default=None,
                        help="Source run number to load checkpoint from")
    parser.add_argument("--new-run",       type=int,   default=400,
                        help="Output run number (default: --run + 1)")
    parser.add_argument("--checkpoint",    type=str,   default=None,
                        help="Explicit source checkpoint path (overrides --run)")
    parser.add_argument("--save-dir",      type=str,   default=None)
    parser.add_argument("--data-dir",      default=DEFAULTS["data_dir"],
                        help="Directory containing respeaker_power_*.npy files")
    parser.add_argument("--iterations",    type=int,   default=DEFAULTS["training_iterations"])
    parser.add_argument("--eval-interval", type=int,   default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval-episodes", type=int,   default=DEFAULTS["eval_episodes"])
    parser.add_argument("--max-ep-steps",  type=int,   default=DEFAULTS["max_ep_steps"])
    parser.add_argument("--workers",       type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    parser.add_argument("--train-batch",   type=int,   default=None)
    parser.add_argument("--entropy",       type=float, default=None)
    parser.add_argument("--eval-only",     action="store_true")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["training_iterations"] = args.iterations
    cfg["eval_interval"]       = args.eval_interval
    cfg["eval_episodes"]       = args.eval_episodes
    cfg["max_ep_steps"]        = args.max_ep_steps
    cfg["num_workers"]         = args.workers
    cfg["lr"]                  = args.lr
    if args.train_batch is not None:
        cfg["train_batch_size"] = args.train_batch
    if args.entropy is not None:
        cfg["entropy_coeff"] = args.entropy

    # ── Locate source checkpoint ──────────────────────────────────────────
    eval_only = args.eval_only

    if args.checkpoint:
        source_ckpt = os.path.abspath(args.checkpoint)
        if not os.path.exists(source_ckpt):
            print(f"[ERROR] Checkpoint not found: {source_ckpt}")
            sys.exit(1)
        source_run = args.run or DEFAULTS["run_number"]
    else:
        if args.run is None:
            parser.error("--run is required (source checkpoint run number)")
        source_run  = args.run
        cfg["run_number"] = source_run
        source_ckpt, search_dir = find_latest_checkpoint(
            source_run, save_dir=args.save_dir
        )
        if source_ckpt is None:
            print(f"[ERROR] No checkpoint found for run {source_run} in {search_dir}")
            sys.exit(1)

    new_run = args.new_run if args.new_run is not None else source_run + 1

    print("=" * 65)
    print("  TRACK-MDP  —  RESPEAKER DATA FINE-TUNE")
    print("=" * 65)
    print(f"  Source run        : {source_run}")
    print(f"  Source checkpoint : {source_ckpt}")
    print(f"  Output run        : {new_run}")
    print(f"  Grid              : {cfg['nrows']} × {cfg['ncols']}")
    print(f"  Data dir          : {args.data_dir}")
    print(f"  Iterations        : {cfg['training_iterations']}")
    print(f"  Eval every        : {cfg['eval_interval']} iteration(s)")
    print(f"  Max ep steps      : {cfg['max_ep_steps']}")
    print()
    print("  Fast-convergence PPO hyperparameters:")
    for k in ("lr", "train_batch_size", "num_sgd_iter", "sgd_minibatch_size",
              "clip_param", "entropy_coeff", "rollout_fragment_length", "num_workers"):
        print(f"    {k:30s}: {cfg[k]}")
    print(f"  Mode              : {'eval only' if eval_only else 'fine-tune'}")
    print()

    real_positions = load_respeaker_positions(args.data_dir)

    try:
        finetune(cfg, real_positions, source_ckpt, new_run, eval_only=eval_only)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
