#!/usr/bin/env python3
"""
iobt_inference.py — Live inference abstraction for the IoBT Track-MDP policy.

This module wraps the trained PPO policy in a clean interface that the field
telemetry system can drive step-by-step without knowing anything about RLlib,
Track-MDP state encoding, or action-history construction.

Architecture (three layers)
---------------------------
  IoBTInference          — core class: holds policy + env state, exposes
                           reset() / step() / close()
  run_episode()          — episode runner: calls reset/step in a loop,
                           accepts any data-source callable
  IoBTLogger             — injected logger: writes .jsonl per-step records
                           and a .json summary sidecar

Observation format (must match iobt_eval.py / gym_wrapper.py training format)
--------------
  Tuple(state_pos: int, state_time: int, action_history: int[34])

The live caller only needs to supply:
  node_detections : length-10 binary list/array  — 1 if node i fired, else 0
  ground_truth    : length-10 binary list/array  — 1 at the true vehicle node

Everything else (state encoding, action-history accumulation, sensor masking,
reward computation) is handled internally.

Typical usage
-------------
    from iobt_inference import IoBTInference, IoBTLogger, run_episode

    logger = IoBTLogger("logs/", test_id="2026-04-05_001_run195")
    model  = IoBTInference.from_checkpoint(
                 "runs/agent_run195_ppo",
                 cfg={"run_number": 195, ...},
                 logger=logger)

    def my_data_source(action):
        # send action to field hardware, receive next sensor readings
        node_detections = hardware.query(action)
        ground_truth    = gps.current_node()
        return node_detections, ground_truth

    for ep in range(10):
        metrics = run_episode(model, my_data_source, episode_id=ep)
        print(metrics)

    logger.close()
    model.close()
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.iobt_new_env import learning_grid_sarsa_0


# ===========================================================================
# Constants — must stay in sync with iobt_eval.py
# ===========================================================================

N_NODES    = 10          # physical IoBT sensor nodes
_N_GRID    = 4           # grid_env parameter (4×4 backing grid)
_TIME_LIMIT_MAX = 1

# Sensor-window sizes per time-delay level (from gym_wrapper._build_augment_vecs)
_AUGMENT_VEC_1 = [(2 * i + 3) ** 2 for i in range(_TIME_LIMIT_MAX + 1)]  # [9, 25]
_AUGMENT_VEC   = [0]
for _v in _AUGMENT_VEC_1:
    _AUGMENT_VEC.append(_AUGMENT_VEC[-1] + _v)                            # [0, 9, 34]

ACTION_HISTORY_LEN = _AUGMENT_VEC[-1]   # 34  — length of action_history in obs tuple
_MAX_ACTION_SZ     = (2 * _TIME_LIMIT_MAX + 3) ** 2   # 25  — largest window at max delay

SCHEMA_VERSION = 1   # bump whenever the log record structure changes


# ===========================================================================
# Default configuration — mirrors iobt_eval.py DEFAULTS
# ===========================================================================

DEFAULTS: Dict[str, Any] = {
    "run_number":        195,
    "N":                 _N_GRID,
    "num_trans":         6,
    "max_sensors":       6,
    "max_sensors_null":  6,
    "time_limit":        1,
    "time_limit_max":    _TIME_LIMIT_MAX,
    "max_episode_steps": 1000,
}


# ===========================================================================
# Logger
# ===========================================================================

class IoBTLogger:
    """
    Writes structured logs for each test run.

    File layout
    -----------
    <log_dir>/<test_id>.jsonl   — one JSON record per step (appendable)
    <log_dir>/<test_id>.json    — summary sidecar written on close()

    Each step record contains:
        schema_version, test_id, episode_id, step,
        node_detections, ground_truth, action,
        obj_detected, was_missing, reward,
        current_state, time_delay,
        timestamp_utc

    The summary record contains:
        schema_version, test_id, total_episodes, total_steps,
        tracking_accuracy, mean_reward, std_reward,
        mean_length, mean_sensors_used,
        episode_summaries (list of per-episode dicts),
        closed_at_utc
    """

    def __init__(self, log_dir: str, test_id: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.test_id  = test_id
        self._lock    = threading.Lock()

        self._step_path    = os.path.join(log_dir, f"{test_id}.jsonl")
        self._summary_path = os.path.join(log_dir, f"{test_id}.json")

        self._step_file = open(self._step_path, "a", buffering=1)  # line-buffered

        # Running totals for summary
        self._episode_summaries: List[Dict] = []
        self._total_steps  = 0
        self._total_found  = 0
        self._all_rewards:  List[float] = []
        self._all_lengths:  List[int]   = []
        self._all_sensors:  List[float] = []

    # ------------------------------------------------------------------
    # Per-step logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        episode_id:      int,
        step:            int,
        node_detections: List[int],
        ground_truth:    List[int],
        action:          List[int],
        obj_detected:    int,
        was_missing:     bool,
        reward:          float,
        current_state:   int,
        time_delay:      int,
    ) -> None:
        """Append one step record to the .jsonl file (thread-safe)."""
        record = {
            "schema_version": SCHEMA_VERSION,
            "test_id":        self.test_id,
            "episode_id":     episode_id,
            "step":           step,
            "node_detections": list(node_detections),
            "ground_truth":   list(ground_truth),
            "action":         list(action),
            "obj_detected":   obj_detected,
            "was_missing":    bool(was_missing),
            "reward":         float(reward),
            "current_state":  int(current_state),
            "time_delay":     int(time_delay),
            "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._step_file.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Per-episode logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode_id:     int,
        ep_steps:       int,
        ep_found:       int,
        ep_reward:      float,
        ep_sensors_avg: float,
    ) -> None:
        """Record per-episode summary (thread-safe)."""
        accuracy = ep_found / max(ep_steps, 1)
        summary  = {
            "episode_id":       episode_id,
            "steps":            ep_steps,
            "found":            ep_found,
            "tracking_accuracy": accuracy,
            "total_reward":     float(ep_reward),
            "avg_sensors_used": float(ep_sensors_avg),
        }
        with self._lock:
            self._episode_summaries.append(summary)
            self._total_steps += ep_steps
            self._total_found += ep_found
            self._all_rewards.append(ep_reward)
            self._all_lengths.append(ep_steps)
            self._all_sensors.append(ep_sensors_avg)

    # ------------------------------------------------------------------
    # Close / flush
    # ------------------------------------------------------------------

    def close(self) -> Dict:
        """Flush, write summary sidecar, and return the summary dict."""
        with self._lock:
            self._step_file.flush()
            self._step_file.close()

            overall_accuracy = self._total_found / max(self._total_steps, 1)
            summary = {
                "schema_version":    SCHEMA_VERSION,
                "test_id":           self.test_id,
                "total_episodes":    len(self._episode_summaries),
                "total_steps":       self._total_steps,
                "tracking_accuracy": overall_accuracy,
                "mean_reward":       float(np.mean(self._all_rewards))  if self._all_rewards  else 0.0,
                "std_reward":        float(np.std(self._all_rewards))   if self._all_rewards  else 0.0,
                "mean_length":       float(np.mean(self._all_lengths))  if self._all_lengths  else 0.0,
                "mean_sensors_used": float(np.mean(self._all_sensors))  if self._all_sensors  else 0.0,
                "episode_summaries": self._episode_summaries,
                "closed_at_utc":     datetime.now(timezone.utc).isoformat(),
            }

            with open(self._summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            return summary


class _NullLogger:
    """Drop-in no-op logger used when the caller passes logger=None."""
    def log_step(self, *args, **kwargs):  pass
    def log_episode(self, *args, **kwargs): pass
    def close(self): return {}


# ===========================================================================
# Core inference class
# ===========================================================================

class IoBTInference:
    """
    Stateful wrapper around the trained PPO policy for live field inference.

    Internal state (reset on every reset() call)
    --------------------------------------------
    _current_state  : Track-MDP encoded state
    _time_delay     : steps since last confirmed detection
    _actions_list   : growing action-history buffer (mirrors gym_wrapper)
    _step_count     : steps taken in current episode
    _ep_reward      : cumulative episode reward
    _ep_found       : detections in current episode (excluding missing-state)
    _ep_sensors     : total sensor-activations in current episode

    Thread safety
    -------------
    reset() and step() acquire _lock, so a telemetry thread and a control
    thread can operate concurrently without corrupting internal state.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        algo,
        env,
        missing_state: int,
        cfg:    Dict[str, Any],
        logger: Optional[Any] = None,
    ) -> None:
        """
        Low-level constructor.  Prefer IoBTInference.from_checkpoint().

        Parameters
        ----------
        algo          : loaded RLlib PPO algorithm
        env           : grid_env instance (from learning_grid_sarsa_0.grid_env)
        missing_state : encoded missing-state sentinel (N*N*(T+1)+1)
        cfg           : merged configuration dict
        logger        : IoBTLogger instance or None
        """
        self._algo          = algo
        self._env           = env
        self._missing_state = missing_state
        self._cfg           = cfg
        self._time_limit    = cfg["time_limit"]
        self._max_sensors   = cfg["max_sensors"]
        self._max_steps     = cfg["max_episode_steps"]
        self._logger        = logger if logger is not None else _NullLogger()
        self._lock          = threading.Lock()

        # Episode state — initialised properly in reset()
        self._current_state: int       = missing_state
        self._time_delay:    int       = 0
        self._actions_list:  List[int] = []
        self._step_count:    int       = 0
        self._ep_reward:     float     = 0.0
        self._ep_found:      int       = 0
        self._ep_sensors:    int       = 0
        self._episode_id:    int       = -1

    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        cfg:    Optional[Dict[str, Any]] = None,
        logger: Optional[Any]            = None,
    ) -> "IoBTInference":
        """
        Load a PPO checkpoint and return a ready-to-use IoBTInference instance.

        Parameters
        ----------
        checkpoint_path : path to a checkpoint_XXXXXX/ directory
        cfg             : dict of overrides merged onto DEFAULTS
        logger          : IoBTLogger or None
        """
        import ray
        from ray.rllib.algorithms.ppo import PPO

        merged = {**DEFAULTS, **(cfg or {})}

        # Build the environment the same way iobt_eval.py does
        state_trans_cum_prob = [
            round((i + 1) / merged["num_trans"], 4)
            for i in range(merged["num_trans"])
        ]
        qobj = learning_grid_sarsa_0(
            run_number=merged["run_number"],
            N=merged["N"],
            num_trans=merged["num_trans"],
            state_trans_cum_prob=state_trans_cum_prob,
            max_sensors=merged["max_sensors"],
            max_sensors_null=merged["max_sensors_null"],
            time_limit=merged["time_limit"],
            time_limit_max=merged["time_limit_max"],
        )

        missing_state = merged["N"] ** 2 * (merged["time_limit_max"] + 1) + 1

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT}},
            )

        algo = PPO.from_checkpoint(checkpoint_path)
        return cls(algo, qobj.grid_env, missing_state, merged, logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, episode_id: int = 0) -> None:
        """
        Start a new episode.

        Resets all internal state and places the simulated object at a
        random starting node.  Must be called before the first step() of
        every episode.

        Parameters
        ----------
        episode_id : identifier stored in log records for this episode
        """
        with self._lock:
            self._env.reset_object_state()
            self._current_state = self._missing_state
            self._time_delay    = 0
            self._actions_list  = []
            self._step_count    = 0
            self._ep_reward     = 0.0
            self._ep_found      = 0
            self._ep_sensors    = 0
            self._episode_id    = episode_id

    def step(
        self,
        node_detections: List[int],
        ground_truth:    List[int],
    ) -> Dict[str, Any]:
        """
        Advance the model by one timestep.

        Parameters
        ----------
        node_detections : length-10 binary array
            node_detections[i] = 1 if sensor node i fired this timestep.
            In simulation mode this is ignored — the environment drives
            detection internally.  In live mode this is the real signal.

        ground_truth : length-10 binary array
            ground_truth[i] = 1 if the vehicle is known to be at node i.
            Used for reward/accuracy computation only; NOT fed to the policy.

        Returns
        -------
        dict with keys:
            action          : length-N_NODES binary list  — which nodes to activate next
            reward          : float reward for this step
            done            : bool — True if the episode has ended
            obj_detected    : int (0/1) — 1 if vehicle was tracked this step
            was_missing     : bool — True if we were in the lost/missing state
            is_searching    : bool — True if the all-sensors broadcast is active
                              (field dashboard can show a "searching" indicator)
            current_state   : int — encoded Track-MDP state after this step
            time_delay      : int — steps since last confirmed detection
            step            : int — step index within the current episode
            episode_id      : int — episode identifier set in reset()
        """
        with self._lock:
            self._step_count += 1
            step = self._step_count

            # ---- Validate inputs ----
            node_detections = self._validate_array(node_detections, "node_detections")
            ground_truth    = self._validate_array(ground_truth,    "ground_truth")

            num_sensors_in_window = (2 * self._time_delay + 3) ** 2
            was_missing = (self._current_state == self._missing_state)

            # ---- Build tuple observation (mirrors iobt_eval.py exactly) ----
            obs = self._build_obs(was_missing)

            # ---- Query policy ----
            action_full = self._algo.compute_single_action(obs, explore=False)

            # ---- Derive sensor mask ----
            if not was_missing:
                action_clip    = np.array(action_full[-num_sensors_in_window:], dtype=int)
                action_sensors = np.multiply(
                    action_clip,
                    self._env.valid_q_indices_dict[self._time_delay][self._current_state],
                )
                obj_rel_pos, obj_in_window = self._env.realign_obj(
                    self._env.object_pos, self._current_state, self._time_delay
                )
            else:
                # Broadcast all sensors to re-acquire — emit is_searching=True
                action_sensors  = np.ones(num_sensors_in_window, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1

            # Enforce sensor budget
            if action_sensors.sum() > self._max_sensors:
                active = np.where(action_sensors == 1)[0][: self._max_sensors]
                action_sensors = np.zeros(num_sensors_in_window, dtype=int)
                action_sensors[active] = 1

            # ---- Detection check ----
            obj_detected = int(
                obj_in_window == 1 and action_sensors[int(obj_rel_pos)] == 1
            )
            if not was_missing and obj_detected:
                self._ep_found += 1

            # ---- Environment step ----
            reward, next_state, terminal_flag, new_delay = \
                self._env.get_reward_next_state(
                    self._current_state, action_sensors, self._time_delay
                )

            # ---- Update action-history buffer (mirrors iobt_eval.py) ----
            if new_delay == 0:
                self._actions_list = []
            else:
                td_ac       = _AUGMENT_VEC_1[new_delay - 1]
                action_bool = (
                    [1] * _MAX_ACTION_SZ if was_missing else list(action_full)
                )
                self._actions_list = (
                    self._actions_list
                    + list(np.array(action_bool)[-td_ac:])
                )

            # ---- Accumulate episode stats ----
            self._ep_reward  += reward
            self._ep_sensors += int(action_sensors.sum())

            # ---- Build the outward action vector (length = N_NODES) ----
            # action_sensors is sized to the current window; pad/trim to N_NODES
            # so the field system always receives the same-length vector.
            out_action = self._window_to_nodes(action_sensors, num_sensors_in_window)

            # ---- Advance state ----
            done = bool(terminal_flag) or (step >= self._max_steps)
            if not terminal_flag:
                self._current_state = next_state
            self._time_delay = new_delay

            # ---- Logging ----
            self._logger.log_step(
                episode_id=self._episode_id,
                step=step,
                node_detections=list(node_detections),
                ground_truth=list(ground_truth),
                action=out_action,
                obj_detected=obj_detected,
                was_missing=was_missing,
                reward=float(reward),
                current_state=int(self._current_state),
                time_delay=int(self._time_delay),
            )

            if done:
                avg_sensors = self._ep_sensors / max(step, 1)
                self._logger.log_episode(
                    episode_id=self._episode_id,
                    ep_steps=step,
                    ep_found=self._ep_found,
                    ep_reward=self._ep_reward,
                    ep_sensors_avg=avg_sensors,
                )

            return {
                "action":        out_action,
                "reward":        float(reward),
                "done":          done,
                "obj_detected":  obj_detected,
                "was_missing":   bool(was_missing),
                "is_searching":  bool(was_missing),   # convenience alias for dashboard
                "current_state": int(self._current_state),
                "time_delay":    int(self._time_delay),
                "step":          step,
                "episode_id":    self._episode_id,
            }

    def close(self) -> None:
        """Shut down Ray. Call once when finished with all episodes."""
        import ray
        if ray.is_initialized():
            ray.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, was_missing: bool) -> Tuple:
        """Construct the (state_pos, state_time, action_history) tuple."""
        N          = self._cfg["N"]
        time_limit = self._time_limit

        action_vec = [1] * ACTION_HISTORY_LEN
        if self._actions_list:
            e_index = _AUGMENT_VEC[self._time_delay]
            action_vec[:e_index] = self._actions_list

        if not was_missing:
            state_pos  = self._current_state // (time_limit + 1)
            state_time = self._current_state %  (time_limit + 1)
        else:
            state_pos, state_time = N * N, 0

        return (state_pos, state_time, np.array(action_vec, dtype=np.int64))

    def _validate_array(self, arr, name: str) -> np.ndarray:
        """Validate a length-N_NODES binary input array."""
        arr = np.asarray(arr, dtype=int)
        if arr.shape != (N_NODES,):
            raise ValueError(
                f"{name} must have length {N_NODES}, got {arr.shape}"
            )
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"{name} must be binary (0 or 1), got {arr}")
        return arr

    def _window_to_nodes(
        self, action_sensors: np.ndarray, window_size: int
    ) -> List[int]:
        """
        Map the policy's window-sized action vector back to a length-N_NODES
        output vector so the field system always receives the same format.

        In the 4×4 grid environment, the sensor window is a subset of the grid;
        the mapping is approximate — we return the raw window padded/trimmed to
        N_NODES.  For the live interface this vector is what gets sent to the
        hardware nodes.
        """
        out = np.zeros(N_NODES, dtype=int)
        n   = min(window_size, N_NODES)
        out[:n] = action_sensors[:n]
        return out.tolist()


# ===========================================================================
# Episode runner
# ===========================================================================

def run_episode(
    model:      IoBTInference,
    data_source: Callable[[List[int]], Tuple[List[int], List[int]]],
    episode_id: int = 0,
    verbose:    bool = True,
) -> Dict[str, Any]:
    """
    Run one complete episode through the model.

    Parameters
    ----------
    model       : IoBTInference instance (already loaded)
    data_source : callable that accepts the current action (length-N_NODES list)
                  and returns (node_detections, ground_truth), both length-N_NODES.
                  For simulation, use make_sim_source().
                  For live use, wire this to your telemetry layer.
    episode_id  : identifier passed through to logger and returned in metrics
    verbose     : print per-step summary to stdout

    Returns
    -------
    dict with keys:
        episode_id, steps, found, tracking_accuracy,
        total_reward, avg_sensors_used
    """
    model.reset(episode_id=episode_id)

    # Prime the data source with a null action before the first real step
    null_action   = [0] * N_NODES
    node_det, gt  = data_source(null_action)

    ep_steps = ep_found = ep_sensors = 0
    ep_reward = 0.0

    while True:
        result = model.step(node_det, gt)

        ep_steps  += 1
        ep_reward += result["reward"]
        ep_sensors += int(np.sum(result["action"]))
        if not result["was_missing"] and result["obj_detected"]:
            ep_found += 1

        if verbose:
            state_str = "MISSING" if result["was_missing"] else str(result["current_state"])
            search_str = " [SEARCHING]" if result["is_searching"] else ""
            print(
                f"  ep {episode_id:3d}  step {result['step']:4d}  "
                f"state={state_str:<8s}  delay={result['time_delay']}  "
                f"detected={result['obj_detected']}  "
                f"reward={result['reward']:+.2f}{search_str}"
            )

        if result["done"]:
            break

        # Feed the action back to the data source for the next observation
        node_det, gt = data_source(result["action"])

    accuracy = ep_found / max(ep_steps, 1)
    if verbose:
        print(
            f"  --- Episode {episode_id} complete: "
            f"{ep_steps} steps, accuracy={accuracy:.3f}, "
            f"reward={ep_reward:+.2f} ---"
        )

    return {
        "episode_id":       episode_id,
        "steps":            ep_steps,
        "found":            ep_found,
        "tracking_accuracy": accuracy,
        "total_reward":     ep_reward,
        "avg_sensors_used": ep_sensors / max(ep_steps, 1),
    }


# ===========================================================================
# Simulation data source
# ===========================================================================

def make_sim_source(model: IoBTInference) -> Callable:
    """
    Return a data_source callable that reads directly from the model's
    internal iobt_env for simulation / replay use.

    In simulation, the environment drives object movement internally, so
    node_detections and ground_truth are derived from env state rather than
    real hardware.

    The returned callable ignores its action argument (the env already moved
    in model.step()) and just reads the current object position.
    """
    env = model._env

    def _sim_source(action: List[int]) -> Tuple[List[int], List[int]]:
        # ground_truth: one-hot at the current object node
        gt = [0] * N_NODES
        pos = env.object_pos
        if 0 <= pos < N_NODES:
            gt[pos] = 1

        # node_detections: which activated nodes actually see the object.
        # In the sim we don't have real sensor noise, so detection = (node active
        # and object is there).  In live mode this comes from hardware instead.
        det = [0] * N_NODES
        for i, active in enumerate(action):
            if active and gt[i]:
                det[i] = 1

        return det, gt

    return _sim_source


# ===========================================================================
# Convenience: run multiple episodes and print a summary
# ===========================================================================

def run_experiment(
    checkpoint_path: str,
    num_episodes:    int              = 10,
    test_id:         Optional[str]    = None,
    log_dir:         str              = "logs/",
    cfg:             Optional[Dict]   = None,
    data_source_factory: Optional[Callable[["IoBTInference"], Callable]] = None,
    verbose:         bool             = True,
) -> Dict[str, Any]:
    """
    High-level entry point: load model, run N episodes, log results, return summary.

    Parameters
    ----------
    checkpoint_path      : path to checkpoint_XXXXXX/ directory
    num_episodes         : number of episodes to run
    test_id              : log file identifier; auto-generated if None
                           format: YYYY-MM-DD_<num_episodes>ep_run<N>
    log_dir              : directory for .jsonl and .json log files
    cfg                  : config overrides (merged onto DEFAULTS)
    data_source_factory  : callable(model) -> data_source callable.
                           Defaults to make_sim_source (simulation mode).
    verbose              : print per-step and per-episode output

    Returns
    -------
    summary dict (same structure as IoBTLogger.close())
    """
    merged = {**DEFAULTS, **(cfg or {})}

    if test_id is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        test_id  = f"{date_str}_{num_episodes}ep_run{merged['run_number']}"

    logger = IoBTLogger(log_dir, test_id)
    model  = IoBTInference.from_checkpoint(checkpoint_path, cfg=merged, logger=logger)

    if data_source_factory is None:
        data_source_factory = make_sim_source

    data_source = data_source_factory(model)

    print("\n" + "=" * 60)
    print(f"  IoBT INFERENCE EXPERIMENT")
    print("=" * 60)
    print(f"  Test ID      : {test_id}")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Episodes     : {num_episodes}")
    print(f"  Log dir      : {log_dir}")
    print(f"  Max sensors  : {merged['max_sensors']}")
    print()

    episode_metrics = []
    for ep in range(num_episodes):
        metrics = run_episode(model, data_source, episode_id=ep, verbose=verbose)
        episode_metrics.append(metrics)

    summary = logger.close()

    print()
    print("=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Test ID             : {summary['test_id']}")
    print(f"  Total Episodes      : {summary['total_episodes']}")
    print(f"  Total Steps         : {summary['total_steps']}")
    print(f"  Tracking Accuracy   : {summary['tracking_accuracy']:.4f}"
          f"  ({summary['tracking_accuracy']*100:.2f}%)")
    print(f"  Mean Reward         : {summary['mean_reward']:+.3f}"
          f"  ± {summary['std_reward']:.3f}")
    print(f"  Mean Episode Length : {summary['mean_length']:.1f} steps")
    print(f"  Mean Sensors / Step : {summary['mean_sensors_used']:.2f}")
    print(f"  Step log            : {log_dir}{test_id}.jsonl")
    print(f"  Summary             : {log_dir}{test_id}.json")
    print("=" * 60)

    acc = summary["tracking_accuracy"]
    if acc >= 0.80:
        print("  Outstanding tracking performance.")
    elif acc >= 0.65:
        print("  Good tracking performance.")
    elif acc >= 0.50:
        print("  Moderate — consider longer training or tuning.")
    else:
        print("  Poor — check environment config or continue training.")
    print()

    model.close()
    return summary


# ===========================================================================
# CLI entry point
# ===========================================================================

def _find_latest_checkpoint(run_number: int, save_dir: Optional[str] = None) -> Optional[str]:
    """Locate the highest-numbered checkpoint for run_number."""
    candidates = []
    if save_dir:
        candidates.append(save_dir)
    else:
        runs = os.path.join(_PROJECT_ROOT, "runs")
        candidates += [
            os.path.join(runs, f"agent_run{run_number}_ppo"),
            os.path.join(runs, f"agent_iobt_run{run_number}_ppo"),
        ]

    for directory in candidates:
        if not os.path.isdir(directory):
            continue

        # Directory itself is a checkpoint
        if (os.path.isfile(os.path.join(directory, "rllib_checkpoint.json")) or
                os.path.isfile(os.path.join(directory, "algorithm_state.pkl"))):
            return directory

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
            return checkpoints[-1][1]

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run IoBT inference experiment (simulation mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Run 10 simulation episodes with auto-generated test ID
  python examples/iobt_inference.py

  # 50 episodes, specific checkpoint, custom log directory
  python examples/iobt_inference.py --episodes 50 \\
      --checkpoint runs/agent_run195_ppo/checkpoint_000100 \\
      --log-dir results/

  # Quiet mode (no per-step output)
  python examples/iobt_inference.py --episodes 20 --quiet
        """
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run",        type=int, default=DEFAULTS["run_number"])
    parser.add_argument("--save-dir",   type=str, default=None)
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--test-id",    type=str, default=None)
    parser.add_argument("--log-dir",    type=str, default="logs/")
    parser.add_argument("--max-sensors", type=int, default=None)
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt = _find_latest_checkpoint(args.run, save_dir=args.save_dir)

    if ckpt is None:
        print(f"[ERROR] No checkpoint found for run {args.run}.")
        print("  Train first, then re-run this script.")
        sys.exit(1)

    override_cfg = {}
    if args.max_sensors is not None:
        override_cfg["max_sensors"] = args.max_sensors

    run_experiment(
        checkpoint_path=ckpt,
        num_episodes=args.episodes,
        test_id=args.test_id,
        log_dir=args.log_dir,
        cfg=override_cfg,
        verbose=not args.quiet,
    )