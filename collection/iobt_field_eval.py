#!/usr/bin/env python3
"""
iobt_field_eval.py — Evaluate a trained IoBT PPO policy against real field
detections captured by iobt_collector.py and processed by iobt_step_builder.py.

The detections array (shape T×10) is the ground truth: detections[t, i] = 1
means node_{i+1} saw the vehicle in timestep t.  At each step the policy
decides which nodes to activate; a detection is credited only when both
conditions hold simultaneously.

Key difference from iobt_eval.py
---------------------------------
  iobt_eval.py   — environment drives object movement via object_move().
  iobt_field_eval.py — the detections array drives object position.
                       env.object_pos is overridden from detections[t]
                       before every call to get_reward_next_state().
                       object_move() is called internally by that function
                       but its result is immediately discarded.

The Track-MDP state machine (current_state, time_delay, actions_list) evolves
identically to iobt_eval.py — it is a pure function of policy decisions and
detections, not of the simulation dynamics.

Usage
-----
    python examples/iobt_field_eval.py \\
        --detections data/detections_20250812_165716.npy \\
        --timestamps data/timestamps_20250812_165716.npy \\
        --checkpoint runs/agent_run195_ppo

    # With all options
    python examples/iobt_field_eval.py \\
        --detections data/detections_20250812_165716.npy \\
        --timestamps data/timestamps_20250812_165716.npy \\
        --checkpoint runs/agent_run195_ppo \\
        --chunk-size 200        # split array into episodes of 200 steps
        --max-sensors 6 \\
        --run 195 \\
        --no-header             # suppress banner (for scripted use)
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import ray
from ray.rllib.algorithms.ppo import PPO

from src.core.iobt_equal_env import learning_grid_sarsa_0


# ===========================================================================
# Configuration — must match training
# ===========================================================================

DEFAULTS = {
    "run_number":       195,
    "N":                4,
    "num_trans":        6,
    "max_sensors":      6,
    "max_sensors_null": 6,
    "time_limit":       1,
    "time_limit_max":   1,
    "max_episode_steps": 5000,
}

N_NODES = 10   # IoBT physical nodes; col i = node_{i+1}


def _build_augment_vecs(time_limit_max):
    augment_vec_1 = [(2 * i + 3) ** 2 for i in range(time_limit_max + 1)]
    augment_vec   = [0]
    for v in augment_vec_1:
        augment_vec.append(augment_vec[-1] + v)
    return augment_vec_1, augment_vec

_AUGMENT_VEC_1, _AUGMENT_VEC = _build_augment_vecs(DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AUGMENT_VEC[-1]          # 34
MAX_ACTION_SZ      = (2 * DEFAULTS["time_limit_max"] + 3) ** 2   # 25


# ===========================================================================
# Ground-truth helpers
# ===========================================================================

def object_pos_from_row(row: np.ndarray, N: int) -> int:
    """
    Convert a detections row (length-10 binary) to an env object_pos value.

    Returns
    -------
    int   0-9  if the vehicle is visible at a node (first active col)
          N*N  if the row is all-zero (vehicle not visible / outside network)
    """
    if row.sum() == 0:
        return N * N      # terminal sentinel — no node sees the vehicle
    return int(np.argmax(row))   # first col that is 1


# ===========================================================================
# Field replay evaluation
# ===========================================================================

def evaluate_on_field_data(
    algo,
    cfg:           dict,
    detections:    np.ndarray,      # uint8 (T, N_NODES)
    timestamps:    np.ndarray,      # float64 (T,)  — Unix seconds
    chunk_size:    int | None = None,
    verbose:       bool = True,
) -> dict:
    """
    Replay the detections array through the trained policy and compute metrics.

    Parameters
    ----------
    algo        : loaded RLlib PPO algorithm (greedy, explore=False)
    cfg         : config dict (merged onto DEFAULTS)
    detections  : uint8 array (T, 10) from iobt_step_builder.py
    timestamps  : float64 array (T,)  — one Unix ts per timestep
    chunk_size  : if given, split the array into episodes of this many steps.
                  if None, the whole array is one episode.
    verbose     : print per-step output

    Returns
    -------
    dict with per-episode breakdown and aggregate metrics
    """
    T = detections.shape[0]
    assert detections.shape[1] == N_NODES, \
        f"detections must be (T, {N_NODES}), got {detections.shape}"
    assert len(timestamps) == T, "timestamps length must match detections rows"

    N             = cfg["N"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors   = cfg["max_sensors"]

    missing_state = N * N * (time_limit_max + 1) + 1

    state_trans_cum_prob = [
        round((i + 1) / cfg["num_trans"], 4)
        for i in range(cfg["num_trans"])
    ]
    qobj = learning_grid_sarsa_0(
        run_number        = cfg["run_number"],
        N                 = N,
        num_trans         = cfg["num_trans"],
        state_trans_cum_prob = state_trans_cum_prob,
        max_sensors       = max_sensors,
        max_sensors_null  = cfg["max_sensors_null"],
        time_limit        = time_limit,
        time_limit_max    = time_limit_max,
    )
    env = qobj.grid_env

    augment_vec_1, augment_vec = _build_augment_vecs(time_limit_max)

    # Split array into episode chunks
    if chunk_size is None or chunk_size >= T:
        chunks = [(0, T)]
    else:
        chunks = [(i, min(i + chunk_size, T))
                  for i in range(0, T, chunk_size)]

    episode_results = []
    total_steps     = 0
    total_found     = 0     # detections while NOT in missing state

    for ep_idx, (t_start, t_end) in enumerate(chunks):
        ep_det   = detections[t_start:t_end]   # view, no copy
        ep_ts    = timestamps[t_start:t_end]

        # ── Episode-level state ──────────────────────────────────────────
        current_state = missing_state
        time_delay    = 0
        actions_list  = []

        ep_reward  = 0.0
        ep_steps   = 0
        ep_found   = 0       # detections while NOT in missing state
        ep_sensors = 0
        ep_missing = 0       # steps spent in missing state

        # Per-step records for verbose output
        step_log = []

        for t_rel, row in enumerate(ep_det):
            t_abs = t_start + t_rel

            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            # ── 1. Set env.object_pos from ground truth ──────────────────
            #    get_reward_next_state() reads this to decide if the object
            #    was caught.  We pin it here before every call.
            env.object_pos = object_pos_from_row(row, N)
            obj_visible    = (env.object_pos < N * N)   # vehicle in network

            # ── 2. Build observation (identical to iobt_eval.py) ─────────
            action_vec = [1] * ACTION_HISTORY_LEN
            if actions_list:
                e_index = augment_vec[time_delay]
                action_vec[:e_index] = actions_list

            if not was_missing:
                state_pos  = current_state // (time_limit + 1)
                state_time = current_state %  (time_limit + 1)
            else:
                state_pos, state_time = N * N, 0

            obs = (state_pos, state_time, np.array(action_vec, dtype=np.int64))

            # ── 3. Query policy ───────────────────────────────────────────
            action_full = algo.compute_single_action(obs, explore=False)

            # ── 4. Build sensor mask (identical to iobt_eval.py) ─────────
            if not was_missing:
                action_clip    = np.array(action_full[-num_sensors:], dtype=int)
                action_sensors = np.multiply(
                    action_clip,
                    env.valid_q_indices_dict[time_delay][current_state],
                )
                obj_rel_pos, obj_in_window = env.realign_obj(
                    env.object_pos, current_state, time_delay
                )
            else:
                # Broadcast all sensors — re-acquisition mode
                action_sensors = np.ones(num_sensors, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1

            # Enforce sensor budget
            if action_sensors.sum() > max_sensors:
                active = np.where(action_sensors == 1)[0][:max_sensors]
                action_sensors = np.zeros(num_sensors, dtype=int)
                action_sensors[active] = 1

            # ── 5. Detection check ────────────────────────────────────────
            # The mask the policy chose (action_sensors) is applied to the
            # field ground truth (row from detections array).
            # obj_in_window / obj_rel_pos come from the env geometry given
            # the ground-truth object position we set in step 1.
            obj_detected = int(
                obj_in_window == 1
                and action_sensors[int(obj_rel_pos)] == 1
            )
            if not was_missing:
                if obj_detected:
                    ep_found   += 1
                    total_found += 1
            else:
                ep_missing += 1

            # ── 6. Step environment ───────────────────────────────────────
            # get_reward_next_state() uses env.object_pos (set in step 1),
            # then calls object_move() internally — we discard that result
            # by re-setting env.object_pos at the top of the next iteration.
            reward, next_state, terminal_flag, new_delay = \
                env.get_reward_next_state(current_state, action_sensors, time_delay)

            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(action_sensors.sum())
            total_steps += 1

            # ── 7. Update action history (identical to iobt_eval.py) ─────
            if new_delay == 0:
                actions_list = []
            else:
                td_ac       = augment_vec_1[new_delay - 1]
                action_bool = ([1] * MAX_ACTION_SZ
                               if was_missing else list(action_full))
                actions_list = (actions_list
                                + list(np.array(action_bool)[-td_ac:]))

            # ── 8. Verbose per-step line ──────────────────────────────────
            if verbose:
                ts_str     = _unix_to_str(float(ep_ts[t_rel]))
                state_str  = "MISSING" if was_missing else str(current_state)
                search_str = " [SEARCHING]" if was_missing else ""
                gt_nodes   = [i for i in range(N_NODES) if row[i]]
                print(
                    f"  ep {ep_idx:3d}  t={t_abs:5d}  {ts_str}  "
                    f"state={state_str:<5s}  delay={time_delay}  "
                    f"gt={gt_nodes}  "
                    f"detected={obj_detected}  "
                    f"reward={reward:+.2f}{search_str}"
                )

            # ── 9. Advance state ──────────────────────────────────────────
            current_state = next_state
            time_delay    = new_delay

            # terminal_flag means the env thinks the object left (object_pos
            # == N*N).  In field replay this corresponds to a row of all zeros
            # — we can stop the episode here if desired, or just continue.
            # We continue regardless so the full array is consumed.

        # ── Episode summary ───────────────────────────────────────────────
        non_missing_steps = ep_steps - ep_missing
        ep_accuracy = ep_found / max(non_missing_steps, 1)
        ep_result   = {
            "episode":           ep_idx,
            "t_start":           t_start,
            "t_end":             t_end,
            "steps":             ep_steps,
            "missing_steps":     ep_missing,
            "non_missing_steps": non_missing_steps,
            "found":             ep_found,
            "tracking_accuracy": ep_accuracy,
            "total_reward":      float(ep_reward),
            "avg_sensors_used":  ep_sensors / max(ep_steps, 1),
            "ts_start":          float(ep_ts[0])  if len(ep_ts) else None,
            "ts_end":            float(ep_ts[-1]) if len(ep_ts) else None,
        }
        episode_results.append(ep_result)

        if verbose or len(chunks) > 1:
            print(
                f"  --- Episode {ep_idx}  "
                f"steps={ep_steps}  "
                f"non-missing={non_missing_steps}  "
                f"found={ep_found}  "
                f"accuracy={ep_accuracy:.3f}  "
                f"reward={ep_reward:+.2f} ---"
            )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    all_rewards   = [r["total_reward"]      for r in episode_results]
    all_acc       = [r["tracking_accuracy"] for r in episode_results]
    all_lengths   = [r["steps"]             for r in episode_results]
    all_sensors   = [r["avg_sensors_used"]  for r in episode_results]

    # Global accuracy: total found / total non-missing steps
    total_non_missing = sum(r["non_missing_steps"] for r in episode_results)
    global_accuracy   = total_found / max(total_non_missing, 1)

    return {
        # Global (pool all steps)
        "global_tracking_accuracy": global_accuracy,
        "total_steps":              total_steps,
        "total_non_missing_steps":  total_non_missing,
        "total_found":              total_found,
        # Per-episode aggregates
        "mean_accuracy":    float(np.mean(all_acc)),
        "std_accuracy":     float(np.std(all_acc)),
        "mean_reward":      float(np.mean(all_rewards)),
        "std_reward":       float(np.std(all_rewards)),
        "mean_length":      float(np.mean(all_lengths)),
        "mean_sensors_used": float(np.mean(all_sensors)),
        "num_episodes":     len(episode_results),
        # Full breakdown
        "episodes":         episode_results,
    }


# ===========================================================================
# Utilities
# ===========================================================================

def _unix_to_str(ts: float) -> str:
    from datetime import datetime
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond//1000:03d}"


def find_latest_checkpoint(run_number, save_dir=None):
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
        # Directory itself is a checkpoint
        if (os.path.isfile(os.path.join(directory, "rllib_checkpoint.json")) or
                os.path.isfile(os.path.join(directory, "algorithm_state.pkl"))):
            return directory, directory
        # Subdirectory layout
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
        description=(
            "Evaluate a trained IoBT PPO policy on a real field detections array."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input files
-----------
  --detections  uint8 (T, 10) .npy array from iobt_step_builder.py
  --timestamps  float64 (T,) .npy array (Unix seconds, one per step)
                Optional — auto-inferred from detections filename if omitted.

Examples
--------
  python examples/iobt_field_eval.py \\
      --detections data/detections_20250812_165716.npy \\
      --checkpoint runs/agent_run195_ppo

  # Split into 200-step episodes, suppress per-step output
  python examples/iobt_field_eval.py \\
      --detections data/detections_20250812_165716.npy \\
      --chunk-size 200 --quiet

  # Save metrics to JSON
  python examples/iobt_field_eval.py \\
      --detections data/detections_20250812_165716.npy \\
      --out results/field_eval_run195.json
        """
    )
    parser.add_argument("--detections", required=True,
                        help="Path to detections_*.npy (T×10 uint8)")
    parser.add_argument("--timestamps", default=None,
                        help="Path to timestamps_*.npy (T float64). "
                             "Auto-detected if omitted.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint directory. Overrides --run.")
    parser.add_argument("--run",  type=int, default=DEFAULTS["run_number"],
                        help=f"Run number (default: {DEFAULTS['run_number']})")
    parser.add_argument("--save-dir", default=None,
                        help="Override checkpoint search directory.")
    parser.add_argument("--max-sensors", type=int, default=None,
                        help="Override max_sensors from config.")
    parser.add_argument("--chunk-size",  type=int, default=None,
                        help="Split detections array into episodes of this "
                             "many steps (default: whole array = 1 episode).")
    parser.add_argument("--out", default=None,
                        help="Save full metrics JSON to this path.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output (episode summaries still shown).")
    parser.add_argument("--no-header", action="store_true",
                        help="Suppress the startup banner.")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.max_sensors is not None:
        cfg["max_sensors"] = args.max_sensors

    # ── Load arrays ───────────────────────────────────────────────────────
    det_path = Path(args.detections)
    if not det_path.exists():
        print(f"[ERROR] Detections file not found: {det_path}")
        sys.exit(1)

    detections = np.load(det_path)
    if detections.ndim != 2 or detections.shape[1] != N_NODES:
        print(f"[ERROR] Expected shape (T, {N_NODES}), got {detections.shape}")
        sys.exit(1)

    T = detections.shape[0]

    # Auto-detect timestamps file
    if args.timestamps:
        ts_path = Path(args.timestamps)
    else:
        ts_path = det_path.parent / det_path.name.replace(
            "detections_", "timestamps_"
        )

    if ts_path.exists():
        timestamps = np.load(ts_path)
        if len(timestamps) != T:
            print(f"[WARN] timestamps length {len(timestamps)} != T={T}. "
                  f"Using step indices instead.")
            timestamps = np.arange(T, dtype=np.float64)
    else:
        print(f"[INFO] No timestamps file found at {ts_path}. "
              f"Using step indices.")
        timestamps = np.arange(T, dtype=np.float64)

    # ── Print header ──────────────────────────────────────────────────────
    if not args.no_header:
        print("\n" + "=" * 65)
        print("  IoBT PPO — FIELD DATA EVALUATION")
        print("=" * 65)
        print(f"  Detections   : {det_path}  {detections.shape}")
        print(f"  Timestamps   : {ts_path.name if ts_path.exists() else 'step indices'}")
        print(f"  Total steps  : {T}")
        print(f"  Chunk size   : {args.chunk_size or 'full array (1 episode)'}")
        print(f"  Max sensors  : {cfg['max_sensors']}")
        print(f"  Active steps : {int(detections.sum(axis=1).astype(bool).sum())} "
              f"/ {T}  ({int(detections.any(axis=1).sum())/T*100:.1f}% with detections)")
        print()

    # ── Locate checkpoint ─────────────────────────────────────────────────
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path, search_dir = find_latest_checkpoint(
            args.run, save_dir=args.save_dir
        )
        if checkpoint_path is None:
            print(f"[ERROR] No checkpoint found in: {search_dir}")
            sys.exit(1)

    print(f"  Checkpoint   : {checkpoint_path}\n")

    # ── Init Ray + load policy ────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT}},
        )

    print("Loading policy …")
    algo = PPO.from_checkpoint(checkpoint_path)
    print("Policy loaded.\n")

    # ── Run evaluation ────────────────────────────────────────────────────
    metrics = evaluate_on_field_data(
        algo        = algo,
        cfg         = cfg,
        detections  = detections,
        timestamps  = timestamps,
        chunk_size  = args.chunk_size,
        verbose     = not args.quiet,
    )

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Episodes evaluated      : {metrics['num_episodes']}")
    print(f"  Total steps             : {metrics['total_steps']}")
    print(f"  Non-missing steps       : {metrics['total_non_missing_steps']}")
    print(f"  Total detections        : {metrics['total_found']}")
    print()
    print(f"  Global tracking accuracy: {metrics['global_tracking_accuracy']:.4f}"
          f"  ({metrics['global_tracking_accuracy']*100:.2f}%)")
    print(f"  Mean accuracy / episode : {metrics['mean_accuracy']:.4f}"
          f"  ± {metrics['std_accuracy']:.4f}")
    print(f"  Mean reward  / episode  : {metrics['mean_reward']:+.3f}"
          f"  ± {metrics['std_reward']:.3f}")
    print(f"  Mean episode length     : {metrics['mean_length']:.1f} steps")
    print(f"  Mean sensors / step     : {metrics['mean_sensors_used']:.2f}")
    print("=" * 65)

    acc = metrics["global_tracking_accuracy"]
    if   acc >= 0.80: print("  Outstanding tracking performance.")
    elif acc >= 0.65: print("  Good tracking performance.")
    elif acc >= 0.50: print("  Moderate — consider longer training or tuning.")
    else:             print("  Poor — check environment config or continue training.")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved → {out_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()