"""
multiloop_iobt_env.py — synthetic multi-loop environment for the 10-node IoBT graph.

The object walks one of N fixed loops ("laps around the track").  At every
episode reset the env picks a loop uniformly at random and a random starting
position on it, then steps deterministically around that loop for the whole
episode.  Loops never terminate — the object circles forever.

Detection is ideal binary: activating the node the object is on always detects
it.  There is no audio, camera or GPS anywhere in this path, which is the point
— it isolates the multi-loop tracking problem from sensor noise.

Crucially the agent never observes which loop is active.  It sees only the
last known node, the staleness counter, and its own action history, so it has
to infer the trajectory from tracking state alone.

This module imports only numpy and the base grid env, so it stays cheap to
import and test without ray.
"""
from __future__ import annotations

import numpy as np

from .iobt_loops import IOBT_NUM_NODES, validate_loop
from .iobt_new_env import iobt_env

# Backing grid is 4x4 for the 10-node map (see iobt_new_env.py).
IOBT_N = 4

# IoBT-tuned rewards, matching RealIoBTEnv in examples/finetune_deterministic.py
# so synthetic numbers stay comparable with the live-data runs.
DEFAULT_TRACKING_REW              = 1.5
DEFAULT_TRACKING_MISS_REW         = -0.5
DEFAULT_TRACKING_REW_MISSING      = 0.5
DEFAULT_TRACKING_MISS_REW_MISSING = -1.0
DEFAULT_SENSOR_REW                = -0.25


class MultiLoopIoBTEnv(iobt_env):
    """
    iobt_env whose object walks a randomly chosen loop each episode.

    NOTE: every attribute reset_object_state() touches must be set BEFORE
    super().__init__(), because iobt_env.__init__() calls it at the end.
    """

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 loops, seed=None, no_moore_constraint=True,
                 sensor_rew=DEFAULT_SENSOR_REW):
        if not loops:
            raise ValueError("loops must be a non-empty list of node cycles")
        for i, loop in enumerate(loops):
            ok, msg = validate_loop(list(loop))
            if not ok:
                raise ValueError(f"loop {i} {list(loop)} is invalid: {msg}")

        self._loops        = [list(loop) for loop in loops]
        self._rng          = np.random.default_rng(seed)
        self._forced_loop  = None
        self._loop_idx     = 0
        self._pos_idx      = 0
        self._no_moore     = no_moore_constraint

        super().__init__(max_sensors, max_sensors_null, missing_state, time_limit)

        self.tracking_rew              = DEFAULT_TRACKING_REW
        self.tracking_miss_rew         = DEFAULT_TRACKING_MISS_REW
        self.tracking_rew_missing      = DEFAULT_TRACKING_REW_MISSING
        self.tracking_miss_rew_missing = DEFAULT_TRACKING_MISS_REW_MISSING
        self.sensor_rew                = float(sensor_rew)

    # ── loop access ────────────────────────────────────────────────────────

    @property
    def loops(self):
        """Copy of the loop set — mutating it must not affect the env."""
        return [list(loop) for loop in self._loops]

    @property
    def num_loops(self):
        return len(self._loops)

    @property
    def current_loop_idx(self):
        return self._loop_idx

    @property
    def current_loop(self):
        return list(self._loops[self._loop_idx])

    def force_loop(self, idx):
        """
        Pin episode resets to loop ``idx``, or pass None to resume uniform
        random selection.  Used by the per-loop evaluation.
        """
        if idx is not None:
            if not (0 <= idx < len(self._loops)):
                raise IndexError(
                    f"loop index {idx} out of range (0-{len(self._loops) - 1})"
                )
        self._forced_loop = idx

    # ── episode lifecycle ──────────────────────────────────────────────────

    def reset_object_state(self):
        """Pick a loop (uniformly, unless pinned) and a random start on it."""
        if self._forced_loop is not None:
            self._loop_idx = self._forced_loop
        else:
            self._loop_idx = int(self._rng.integers(len(self._loops)))
        loop            = self._loops[self._loop_idx]
        self._pos_idx   = int(self._rng.integers(len(loop)))
        self.object_pos = loop[self._pos_idx]

    def object_move(self):
        """Advance one node along the active loop.  Never terminates."""
        loop            = self._loops[self._loop_idx]
        self._pos_idx   = (self._pos_idx + 1) % len(loop)
        self.object_pos = loop[self._pos_idx]
        return 0

    # ── reward / transition ────────────────────────────────────────────────

    def get_reward_next_state(self, current_state, current_action, time_delay):
        """
        Ideal binary detection against the true node index.

        Mirrors the no_moore path of CircularIoBTEnv: the gym wrapper sends a
        flat n_grid-element action, so we index the node directly instead of
        going through the base env's (2t+3)^2 Moore window, which would shape-
        mismatch at time_delay >= 1.

        terminal_flag is always 0 — the loop never ends.
        """
        n_grid         = self.N * self.N          # 16, backing grid size
        obj_position   = self.object_pos
        action_sensors = np.asarray(current_action[:n_grid])

        # In missing state the tracker performs a full rescan, which must always
        # succeed — otherwise time_delay grows without bound.
        if current_state == self.missing_state:
            action_sensors = np.ones(n_grid, dtype=int)

        node_activated = int(action_sensors[obj_position]) == 1

        if node_activated:
            next_state       = obj_position * (self.time_limit + 1)
            time_delay_sense = 0
        else:
            time_delay_sense = time_delay + 1
            if current_state != self.missing_state:
                next_state = (self.missing_state
                              if time_delay_sense > self.time_limit
                              else current_state + 1)
            else:
                next_state = self.missing_state

        self.object_move()
        no_sensor_on = int(action_sensors.sum())

        if current_state != self.missing_state:
            reward = (float(node_activated)       * self.tracking_rew
                      + float(not node_activated) * self.tracking_miss_rew
                      + no_sensor_on              * self.sensor_rew)
        else:
            reward = (float(node_activated)       * self.tracking_rew_missing
                      + float(not node_activated) * self.tracking_miss_rew_missing
                      + n_grid                    * self.sensor_rew)

        return reward, next_state, 0, time_delay_sense


class MultiLoopIoBTLearner:
    """Wraps MultiLoopIoBTEnv to match the qobj interface the gym wrapper expects."""

    def __init__(self, run_number, num_trans, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, loops, seed=None,
                 no_moore_constraint=True, sensor_rew=DEFAULT_SENSOR_REW):
        self.run_number     = run_number
        # N must be the backing grid size so the gym wrapper's obs space is right.
        self.N              = IOBT_N
        self.n_cells        = IOBT_NUM_NODES
        self.num_trans      = num_trans
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = IOBT_N * IOBT_N * (time_limit_max + 1) + 1

        self.grid_env = MultiLoopIoBTEnv(
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            loops, seed=seed, no_moore_constraint=no_moore_constraint,
            sensor_rew=sensor_rew,
        )

        self.exploration_epsilon = 0.15
        self.total_actions       = self.grid_env.action_space_size
        self.total_actions_null  = self.grid_env.action_space_size_null
        self.current_state       = self.missing_state
        self.current_action      = 0
        self.next_state          = 0
        self.next_action         = 0
        self.time_delay          = 0
        self.max_sensors         = max_sensors
        self.sarsa_step_size     = 0.1
        self.gamma               = 1
        self.no_of_episodes      = 1
        self.episode_start       = 0
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


__all__ = ["MultiLoopIoBTEnv", "MultiLoopIoBTLearner"]
