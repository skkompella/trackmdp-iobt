"""
matrix_iobt_env.py — IoBT env whose object follows a transition matrix.

The loop envs move the object deterministically; this one samples each move
from a row of a 10x10 matrix over IOBT_MAP (stay + out-neighbours).  Several
matrices can be loaded and switched between with ``force_matrix``, which is
what the random-matrix switching experiment needs.  ``force_loop`` is an alias
so the existing online schedule code drives it unchanged.

Rewards, detection and the missing-state rescan are inherited from
MultiLoopIoBTEnv, so numbers are directly comparable with the loop runs.
"""
from __future__ import annotations

import numpy as np

from .iobt_loops import IOBT_MAP, IOBT_NUM_NODES
from .multiloop_iobt_env import IOBT_N, DEFAULT_REWARD_PRESET, MultiLoopIoBTEnv

# Passed only to satisfy the parent's loop validation; never walked.
_PLACEHOLDER_LOOP = [0, 4, 3, 6, 2, 1]


def validate_iobt_matrix(T) -> tuple[bool, str]:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (IOBT_NUM_NODES, IOBT_NUM_NODES):
        return False, f"shape {T.shape}"
    if (T < 0).any() or not np.allclose(T.sum(axis=1), 1.0):
        return False, "rows must be non-negative and sum to 1"
    for n in range(IOBT_NUM_NODES):
        allowed = set(IOBT_MAP[n]) | {n}
        if any(int(d) not in allowed for d in np.flatnonzero(T[n])):
            return False, f"row {n} puts mass outside stay + IOBT_MAP[{n}]"
    return True, "OK"


class MatrixIoBTEnv(MultiLoopIoBTEnv):
    """MultiLoopIoBTEnv whose object moves by sampling a transition matrix."""

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 matrices, seed=None, no_moore_constraint=True,
                 reward_preset=DEFAULT_REWARD_PRESET, sensor_rew=None):
        checked = []
        for i, T in enumerate(matrices):
            ok, msg = validate_iobt_matrix(T)
            if not ok:
                raise ValueError(f"matrix {i} is invalid: {msg}")
            checked.append(np.asarray(T, dtype=np.float64))
        # Set before super().__init__(), which calls reset_object_state().
        self._matrices   = checked
        self._matrix_idx = 0
        self._forced_m   = None
        super().__init__(max_sensors, max_sensors_null, missing_state,
                         time_limit, loops=[_PLACEHOLDER_LOOP], seed=seed,
                         no_moore_constraint=no_moore_constraint,
                         reward_preset=reward_preset, sensor_rew=sensor_rew)

    @property
    def matrices(self):
        return [T.copy() for T in self._matrices]

    @property
    def num_matrices(self):
        return len(self._matrices)

    @property
    def current_matrix_idx(self):
        return self._matrix_idx

    def force_matrix(self, idx):
        if idx is not None and not (0 <= idx < len(self._matrices)):
            raise IndexError(f"matrix index {idx} out of range")
        self._forced_m = idx
        if idx is not None:
            self._matrix_idx = idx

    force_loop = force_matrix   # lets the loop schedule code drive it

    def reset_object_state(self):
        if getattr(self, "_forced_m", None) is not None:
            self._matrix_idx = self._forced_m
        elif hasattr(self, "_matrices"):
            self._matrix_idx = int(self._rng.integers(len(self._matrices)))
        self._loop_idx = self._pos_idx = 0
        self.object_pos = int(self._rng.integers(IOBT_NUM_NODES))

    def object_move(self):
        row = self._matrices[self._matrix_idx][self.object_pos]
        self.object_pos = int(self._rng.choice(IOBT_NUM_NODES, p=row))
        return 0


class MatrixIoBTLearner:
    """qobj facade for the RLlib gym wrapper (mirrors MultiLoopIoBTLearner)."""

    def __init__(self, run_number, num_trans, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, matrices, seed=None,
                 no_moore_constraint=True,
                 reward_preset=DEFAULT_REWARD_PRESET, sensor_rew=None):
        self.run_number     = run_number
        self.N              = IOBT_N          # backing grid width the wrapper reads
        self.n_cells        = IOBT_NUM_NODES
        self.num_trans      = num_trans
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = IOBT_N * IOBT_N * (time_limit_max + 1) + 1

        self.grid_env = MatrixIoBTEnv(
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            matrices, seed=seed, no_moore_constraint=no_moore_constraint,
            reward_preset=reward_preset, sensor_rew=sensor_rew)

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
        self.grid_env.valid_q_indices_dict = \
            self.grid_env.get_valid_q_indices_dict()


__all__ = ["MatrixIoBTEnv", "MatrixIoBTLearner", "validate_iobt_matrix"]
