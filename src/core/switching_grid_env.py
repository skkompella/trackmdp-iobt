"""
switching_grid_env.py — a rectangular grid whose object moves under one of
several transition matrices, switched on a schedule.

This is the grid analogue of MultiLoopIoBTEnv.  The difference that matters:
the ten IoBT loops were all paths through one graph, so they shared transition
structure and a learner's knowledge carried across a switch.  Two opposed drift
matrices do not share that way, which is the point — it is the case the
multi-loop experiment left untested.

Built on grid_env_rect rather than TransitionMatrixGridEnv, because that class
has three problems for this use:

  1. It accepts no ``no_moore_constraint``.  The gym wrapper receives the flag
     but the env never does, so the wrapper emits a flat n_cells action while
     the env slices a Moore window off the end of it.  Silently wrong.
  2. ``object_move`` draws with the global numpy RNG, so runs are not
     reproducible and any other code calling np.random perturbs the trajectory.
  3. grid_env_rect hardcodes ``sensor_rew = -0.16`` with no way to override it.

All three are fixed here.  Note also that terminal is forced to 0: the
transition matrices have no terminal row, so the object circulates forever, as
it does in the loop env.
"""
from __future__ import annotations

import numpy as np

from .grid_env_rect import grid_env_rect
from .grid_transition_matrices import validate_transition_matrix


class SwitchingTransitionGridEnv(grid_env_rect):
    """
    grid_env_rect whose object_move() samples from the currently active matrix.

    Unlike MultiLoopIoBTEnv there is no ordering trap here: grid_env_rect's
    __init__ assigns object_pos directly rather than calling
    reset_object_state(), so the matrices can be installed after super().
    They are still installed first, defensively, in case that changes.
    """

    def __init__(self, nrows, ncols, max_sensors, max_sensors_null,
                 missing_state, time_limit, matrices, seed=None,
                 no_moore_constraint=True, sensor_rew=None,
                 num_trans=4, state_trans_cum_prob=None):
        matrices = list(matrices) if matrices is not None else []
        if not matrices:
            raise ValueError("matrices must be a non-empty list")

        n_cells = nrows * ncols
        checked = []
        for i, T in enumerate(matrices):
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (n_cells, n_cells):
                raise ValueError(
                    f"matrix {i} has shape {T.shape}, expected {(n_cells, n_cells)}")
            ok, msg = validate_transition_matrix(T, nrows, ncols)
            if not ok:
                raise ValueError(f"matrix {i} is invalid: {msg}")
            checked.append(T)

        self._matrices    = checked
        self._rng         = np.random.default_rng(seed)
        self._forced      = None
        self._matrix_idx  = 0

        if state_trans_cum_prob is None:
            state_trans_cum_prob = [round((i + 1) / num_trans, 4)
                                    for i in range(num_trans)]

        super().__init__(nrows, ncols, num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state,
                         time_limit, no_moore_constraint=no_moore_constraint)

        if sensor_rew is not None:
            self.sensor_rew = float(sensor_rew)

        self.reset_object_state()

    # ── matrix access ──────────────────────────────────────────────────────

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
        """Pin the active regime, or pass None for uniform random selection."""
        if idx is not None and not (0 <= idx < len(self._matrices)):
            raise IndexError(
                f"matrix index {idx} out of range (0-{len(self._matrices) - 1})")
        self._forced = idx
        if idx is not None:
            self._matrix_idx = idx

    # ── episode lifecycle ──────────────────────────────────────────────────

    def reset_object_state(self):
        """Pick the regime (unless pinned) and a uniform random start cell."""
        if self._forced is not None:
            self._matrix_idx = self._forced
        else:
            self._matrix_idx = int(self._rng.integers(len(self._matrices)))
        self.object_pos = int(self._rng.integers(self.n_cells))

    def object_move(self):
        """Sample the next cell from the active matrix. Never terminates."""
        row = self._matrices[self._matrix_idx][self.object_pos]
        self.object_pos = int(self._rng.choice(self.n_cells, p=row))
        return 0

    # ── reward / transition ────────────────────────────────────────────────

    def get_reward_next_state(self, current_state, current_action, time_delay):
        """
        Delegate to grid_env_rect, then force terminal to 0.

        The parent flags terminal when object_pos == n_cells, which its own
        equal-probability tables can produce via TERMINAL_PROB.  Our matrices
        have no terminal row, so that can never fire — the clamp is belt and
        braces, and documents the intent.
        """
        reward, next_state, _terminal, delay = super().get_reward_next_state(
            current_state, current_action, time_delay)
        return reward, next_state, 0, delay


class SwitchingGridLearner:
    """qobj façade matching what the gym wrappers expect."""

    def __init__(self, run_number, nrows, ncols, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, matrices, seed=None,
                 no_moore_constraint=True, sensor_rew=None, num_trans=4):
        self.run_number     = run_number
        self.nrows          = nrows
        self.ncols          = ncols
        self.N              = ncols                 # wrappers read this
        self.n_cells        = nrows * ncols
        self.num_trans      = num_trans
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = self.n_cells * (time_limit_max + 1) + 1

        self.grid_env = SwitchingTransitionGridEnv(
            nrows, ncols, max_sensors, max_sensors_null, self.missing_state,
            time_limit, matrices, seed=seed,
            no_moore_constraint=no_moore_constraint, sensor_rew=sensor_rew,
            num_trans=num_trans)

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


__all__ = ["SwitchingTransitionGridEnv", "SwitchingGridLearner"]
