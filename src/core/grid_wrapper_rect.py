"""
gym_wrapper_rect.py — Gymnasium wrapper for the rectangular M×N grid environment.

Drop-in replacement for gym_wrapper.py that works with grid_env_rect.
The only structural change is the observation-space Discrete size:
    original : Discrete(N²   + 1)   — square N×N grid
    rect     : Discrete(M*N  + 1)   — rectangular M×N grid, M*N cells

Everything else (action space, augment_vec, step logic) is unchanged.
"""

import gymnasium as gym
import numpy as np
import random as rnd
from gymnasium.utils import seeding

SEED = 1
rnd.seed(SEED)
np.random.seed(SEED)


class grid_environment_rect(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config=None):
        self.episode_counter = 0
        self.algo_counter    = 0

        self.qobj               = env_config["qobj"]
        self.time_limit_schedule = env_config["time_limit_schedule"]
        self.time_limit_max     = env_config["time_limit_max"]
        self.missing_state      = self.qobj.missing_state
        self.n_cells            = self.qobj.n_cells    # NROWS * NCOLS
        self._max_ep_steps      = env_config.get("max_ep_steps", 100)
        self._ep_step           = 0
        self.no_moore_constraint = env_config.get("no_moore_constraint", False)

        # Action space: full grid when no_moore_constraint, else (2t+3)² window
        if self.no_moore_constraint:
            action_space_sz = self.n_cells
        else:
            action_space_sz = (2 * self.time_limit_max + 3) ** 2
        self.action_space = gym.spaces.MultiDiscrete([2] * action_space_sz)

        # Augment vector
        if self.no_moore_constraint:
            self.augment_vec_1 = [self.n_cells] * (self.time_limit_max + 1)
        else:
            self.augment_vec_1 = [(2 * i + 3) ** 2 for i in range(self.time_limit_max + 1)]
        augment_vec = [0]
        for v in self.augment_vec_1:
            augment_vec.append(augment_vec[-1] + v)
        self.augment_vec = augment_vec

        self.actions_list = []

        # Observation space:
        #   state_pos  ∈ Discrete(n_cells + 1)   ← only change vs square wrapper
        #   state_time ∈ Discrete(time_limit_max + 1)
        #   action_history ∈ MultiDiscrete([2] * augment_vec[-1])
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(self.n_cells + 1),
            gym.spaces.Discrete(self.time_limit_max + 1),
            gym.spaces.MultiDiscrete([2] * augment_vec[-1]),
        ))

        self.current_state = self.missing_state
        self.state         = self.current_state

        _blank_action_history = np.array([1] * augment_vec[-1])
        self.tuple_augment_missing_state = (self.n_cells, 0, _blank_action_history)
        self.tuple_augment_state         = self.tuple_augment_missing_state

        self.time_delay = 0
        self.info       = {}
        self.reward     = 0
        self.done       = False
        self._ep_step   = 0

        self.seed(SEED)
        self.reset()

    # -----------------------------------------------------------------------
    # State helpers
    # -----------------------------------------------------------------------

    def to_tuple_augment_state(self, state, action_list):
        action_tp_len  = self.augment_vec[-1]
        action_vec     = [1] * action_tp_len
        if action_list:
            e_index = self.augment_vec[self.time_delay]
            action_vec[:e_index] = action_list

        state_pos  = state // (self.qobj.time_limit + 1)
        state_time = state %  (self.qobj.time_limit + 1)
        return (state_pos, state_time, np.array(action_vec))

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.reset_object_state()
        self.current_state       = self.missing_state
        self.state               = self.current_state
        self.tuple_augment_state = self.tuple_augment_missing_state
        self.actions_list        = []
        self.reward              = 0
        self.done                = False
        self._ep_step            = 0
        self.info                = {}
        return self.tuple_augment_state, {}

    def step(self, action):
        if self.current_state == self.missing_state:
            action_bool = [1] * (self.n_cells if self.no_moore_constraint
                                 else (2 * self.time_limit_max + 3) ** 2)
        else:
            action_bool = action

        reward, next_state, terminal, self.time_delay = \
            self.qobj.grid_env.get_reward_next_state(
                self.current_state, action_bool, self.time_delay
            )

        if self.time_delay == 0:
            self.actions_list = []
        else:
            td_ac = self.augment_vec_1[self.time_delay - 1]
            self.actions_list = self.actions_list + list(action_bool[-td_ac:])

        self.current_state = next_state
        if terminal:
            self.done = True
        if next_state != self.missing_state:
            self.state = self.current_state

        self._ep_step           += 1
        truncated                = (self._ep_step >= self._max_ep_steps)
        self.reward              = reward
        self.tuple_augment_state = self.to_tuple_augment_state(self.state, self.actions_list)
        return [self.tuple_augment_state, self.reward, self.done, truncated, self.info]

    def render(self, mode="human"):
        print(f"position: {self.state:2d}  reward: {self.reward:+.3f}  info: {self.info}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def reset_object_state(self):
        self.qobj.grid_env.object_pos = rnd.randrange(self.n_cells)