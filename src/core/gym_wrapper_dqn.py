"""
gym_wrapper_dqn.py — Gymnasium wrapper with a FLAT Discrete action space
for use with Rainbow DQN (and any other Discrete-action algorithm).

Why a separate wrapper?
-----------------------
PPO works with MultiDiscrete([2]*25): one binary decision per sensor cell.
DQN requires a single Discrete action space.  We enumerate all valid sensor
subsets (combinations of up to max_sensors cells in the max window) and map
each to an integer action index.

Action map
----------
    action_idx -> binary vector of length (2*time_limit_max+3)²

Built at __init__ time by enumerating C(window_size, k) for k = 1..max_sensors
in order of increasing k, then lexicographic within each k.

Action space size examples (window=25, time_limit_max=1):
    max_sensors=2:   325 actions
    max_sensors=3: 2,625 actions
    max_sensors=4: 15,275 actions
    max_sensors=6: 245,505 actions   (large — use max_sensors≤4 for DQN)

Observation space
-----------------
Identical to grid_environment_rect (Tuple with Discrete + MultiDiscrete).
RLlib's preprocessor flattens Tuple observations to a 1-D float vector
automatically before passing them to the DQN network.

Usage
-----
Pass this class as the environment to DQNConfig().environment(...).
"""

import gymnasium as gym
import numpy as np
import random as rnd
from itertools import combinations
from gymnasium.utils import seeding

SEED = 1
rnd.seed(SEED)
np.random.seed(SEED)


def build_action_map(max_window_sz: int, max_sensors: int) -> list[np.ndarray]:
    """
    Enumerate all binary sensor-activation vectors with 1..max_sensors active cells.

    Returns a list of uint8 arrays of length max_window_sz, ordered by:
        (number of active sensors, lexicographic cell index order).

    action_idx 0 always activates only the first cell.
    """
    actions = []
    for k in range(1, max_sensors + 1):
        for combo in combinations(range(max_window_sz), k):
            vec = np.zeros(max_window_sz, dtype=np.uint8)
            vec[list(combo)] = 1
            actions.append(vec)
    return actions


class grid_environment_dqn(gym.Env):
    """Rectangular M×N grid environment with Discrete action space for DQN."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config=None):
        self.episode_counter = 0
        self.algo_counter    = 0

        self.qobj                = env_config["qobj"]
        self.time_limit_schedule = env_config["time_limit_schedule"]
        self.time_limit_max      = env_config["time_limit_max"]
        self.missing_state       = self.qobj.missing_state
        self.n_cells             = self.qobj.n_cells

        # Max window size (at maximum time delay)
        self._max_window_sz = (2 * self.time_limit_max + 3) ** 2

        # Build action map: list index = action_idx, value = binary vec
        self._action_map = build_action_map(
            self._max_window_sz, self.qobj.max_sensors
        )
        n_actions = len(self._action_map)

        # Discrete action space
        self.action_space = gym.spaces.Discrete(n_actions)

        # Augment vector (same as PPO wrapper — depends only on time_limit_max)
        self.augment_vec_1 = [(2 * i + 3) ** 2 for i in range(self.time_limit_max + 1)]
        augment_vec = [0]
        for v in self.augment_vec_1:
            augment_vec.append(augment_vec[-1] + v)
        self.augment_vec = augment_vec

        self.actions_list = []

        # Observation space (same as PPO wrapper — RLlib flattens automatically)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(self.n_cells + 1),
            gym.spaces.Discrete(self.time_limit_max + 1),
            gym.spaces.MultiDiscrete([2] * augment_vec[-1]),
        ))

        self.current_state = self.missing_state
        self.state         = self.current_state
        self.time_delay    = 0

        _blank_history = np.array([1] * augment_vec[-1], dtype=np.int8)
        self.tuple_augment_missing_state = (self.n_cells, 0, _blank_history)
        self.tuple_augment_state         = self.tuple_augment_missing_state

        self.info   = {}
        self.reward = 0
        self.done   = False

        self.seed(SEED)
        self.reset()

    # -----------------------------------------------------------------------
    # State helpers
    # -----------------------------------------------------------------------

    def to_tuple_augment_state(self, state, action_list):
        action_tp_len = self.augment_vec[-1]
        action_vec    = [1] * action_tp_len
        if action_list:
            e_index = self.augment_vec[self.time_delay]
            action_vec[:e_index] = action_list

        state_pos  = state // (self.qobj.time_limit + 1)
        state_time = state %  (self.qobj.time_limit + 1)
        return (state_pos, state_time, np.array(action_vec, dtype=np.int8))

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.reset_object_state()
        self.current_state       = self.missing_state
        self.state               = self.current_state
        self.tuple_augment_state = self.tuple_augment_missing_state
        self.actions_list        = []
        self.time_delay          = 0
        self.reward              = 0
        self.done                = False
        self.info                = {}
        return self.tuple_augment_state, {}

    def step(self, action_idx: int):
        # Convert discrete action index to binary sensor vector
        if self.current_state == self.missing_state:
            # Broadcast all sensors to re-acquire
            action_bool = [1] * self._max_window_sz
        else:
            action_bool = self._action_map[int(action_idx)].tolist()

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

        self.reward              = reward
        self.tuple_augment_state = self.to_tuple_augment_state(
            self.state, self.actions_list
        )
        return [self.tuple_augment_state, self.reward, self.done, False, self.info]

    def render(self, mode="human"):
        print(f"pos={self.state}  reward={self.reward:+.3f}  delay={self.time_delay}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def reset_object_state(self):
        self.qobj.grid_env.object_pos = rnd.randrange(self.n_cells)

    # -----------------------------------------------------------------------
    # Utility: decode a discrete action to the binary vector (for eval)
    # -----------------------------------------------------------------------

    def decode_action(self, action_idx: int) -> np.ndarray:
        """Return the binary sensor vector for a given action index."""
        return self._action_map[int(action_idx)].copy()