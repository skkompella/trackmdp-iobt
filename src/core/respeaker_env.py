"""
respeaker_env.py — Environment variants that replay real respeaker power data.

RespeakerGridEnv replaces the stochastic object-movement model with a
deterministic replay of ground-truth cell positions derived from recorded
respeaker power readings (highest-power node = object location).

The position sequence loops indefinitely so training can run for many
iterations without exhausting the data.
"""

import numpy as np

from src.core.grid_env_rect import grid_env_rect
from src.core.grid_wrapper_rect import grid_environment_rect


class RespeakerGridEnv(grid_env_rect):
    """
    Drops in for grid_env_rect but moves the object through a fixed sequence
    of real positions rather than using stochastic transitions.

    Parameters
    ----------
    real_positions : array-like of int
        Ground-truth cell indices (0 .. n_cells-1), one per second.
        Looped indefinitely via object_move().
    All other parameters identical to grid_env_rect.
    """

    def __init__(self, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, missing_state, time_limit,
                 real_positions):
        super().__init__(nrows, ncols, num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state, time_limit)
        self._real_positions = np.asarray(real_positions, dtype=np.int32)
        self._n_real   = len(self._real_positions)
        self._pos_idx  = 0
        self.object_pos = int(self._real_positions[0])

    def object_move(self):
        """Advance one step in the real position sequence, looping at the end."""
        self._pos_idx  = (self._pos_idx + 1) % self._n_real
        self.object_pos = int(self._real_positions[self._pos_idx])
        return 0

    def reset_object_state(self):
        """Stay at the current sequence position (continuous replay across episodes)."""
        self.object_pos = int(self._real_positions[self._pos_idx])


class respeaker_grid_environment(grid_environment_rect):
    """
    Gymnasium wrapper for RespeakerGridEnv.

    Two additions over grid_environment_rect:
    1. reset_object_state() is a no-op — RespeakerGridEnv manages the
       position pointer so it is not randomised at episode boundaries.
    2. Episodes are truncated at max_ep_steps (from env_config, default 100)
       by returning done=True from step().  Without this, RespeakerGridEnv
       never fires a terminal signal (the real object never leaves the grid),
       so RLlib would see no completed episodes and report episode_reward_mean
       as NaN.
    """

    def __init__(self, env_config=None):
        self._max_ep_steps = (env_config or {}).get("max_ep_steps", 100)
        self._ep_steps = 0
        super().__init__(env_config)

    def reset(self, *, seed=None, options=None):
        self._ep_steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self._ep_steps += 1
        if self._ep_steps >= self._max_ep_steps:
            done = True
        return obs, reward, done, truncated, info

    def reset_object_state(self):
        pass  # RespeakerGridEnv manages its own position pointer
