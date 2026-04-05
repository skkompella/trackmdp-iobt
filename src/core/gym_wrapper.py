"""
Gymnasium Environment Wrapper for Object Tracking

This module provides a Gymnasium-compatible wrapper for the grid tracking environment,
enabling integration with Ray RLlib and other RL frameworks.
"""

import gymnasium as gym
import numpy as np
import random as rnd 
from gymnasium.utils import seeding

try:
    from ..visualization.renderer import TrackingRenderer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

SEED = 1
rnd.seed(SEED)
np.random.seed(SEED)


class grid_environment(gym.Env):
    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self, env_config=None):
        
        self.episode_counter = 0
        self.algo_counter = 0
        
        self.qobj = env_config['qobj']
        self.time_limit_schedule = env_config['time_limit_schedule']
        self.time_limit_max = env_config['time_limit_max']

        self.missing_state = self.qobj.missing_state
        
        action_space_sz = ((2*self.time_limit_max) + 3)**2
        
        self.action_space = gym.spaces.MultiDiscrete([2]*(action_space_sz))

        ##### Augment vec 
        augment_vec = [0]
        self.augment_vec_1 = [(2*i+ 3)**2 for i in range(self.time_limit_max+1)]
        for aval in self.augment_vec_1:
            augment_vec = augment_vec + [augment_vec[-1] + aval]
            
        self.augment_vec = augment_vec
        
        self.actions_list = []
        
        self.observation_space = gym.spaces.Tuple((
                                     gym.spaces.Discrete((self.qobj.N**2 +1)),
                                     gym.spaces.Discrete(self.qobj.time_limit_max+1), 
                                     gym.spaces.MultiDiscrete([2]*augment_vec[-1])   ))

        self.current_state = self.missing_state

        self.state = self.current_state
        
        self.tuple_augment_state = (self.qobj.N**2, 0, np.array([1]*augment_vec[-1])) 
        self.tuple_augment_missing_state = self.tuple_augment_state
        
        self.time_delay = 0
        self.info = {}
        self.reward = 0

        self.done = False

        # NB: change to guarantee the sequence of pseudorandom numbers
        # (e.g., for debugging)
        self.seed(SEED)

        self.reset()
        
    def to_binary_state(self, state):
        return np.array([int(i) for i in np.binary_repr(state, width=self.observation_space_binary_length)])
        
    def to_tuple_state(self, state):
        state_pos = state//(self.qobj.time_limit+1)
        state_time = state % (self.qobj.time_limit+1)
        return (state_pos, state_time)
    
    def to_tuple_augment_state(self, state, action_list):
        action_tp_len = self.augment_vec[-1]
        action_tp_ones = [1]*action_tp_len
        if len(action_list) > 0:
            action_vec = action_tp_ones
            e_index = self.augment_vec[self.time_delay-1+1]
            action_vec[:e_index] = action_list
        else:
            action_vec = action_tp_ones
            
        state_pos = state//(self.qobj.time_limit+1)
        state_time = state % (self.qobj.time_limit+1)
            
        return (state_pos, state_time, np.array(action_vec))

    def reset(self, *, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.reset_object_state()
        
        self.current_state = self.qobj.missing_state
        self.state = self.current_state 
        self.tuple_augment_state = self.tuple_augment_missing_state
        
        self.actions_list = []
        self.reward = 0
        self.done = False
        self.info = {}
        
        return self.tuple_augment_state, {}

    def step(self, action):

        if self.current_state == self.missing_state:
            action_bool = [1 for i in np.arange(((2*self.time_limit_max) + 3)**2)] 
        else:
            action_bool = action

        ###################
        reward, next_state, terminal_st_obj, self.time_delay = self.qobj.grid_env.get_reward_next_state(self.current_state, action_bool, self.time_delay)

        if self.time_delay == 0:
            self.actions_list = []
        else:
            td_ac = self.augment_vec_1[self.time_delay-1] 
            self.actions_list = self.actions_list + list(action_bool[-td_ac:])
        
        self.current_state = next_state
        
        if terminal_st_obj == 1:
            self.done = True

        if next_state != self.missing_state:
            self.state = self.current_state

        self.reward = reward
        self.truncated = bool(0)
        
        self.tuple_augment_state = self.to_tuple_augment_state(self.state, self.actions_list)
        
        return [self.tuple_augment_state, self.reward, self.done, self.truncated, self.info]

    def render(self, mode="human"):
        s = "position: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
    
    def reset_object_state(self):
        self.qobj.grid_env.object_pos = rnd.sample(list(np.arange(self.qobj.N**2)), 1)[0]


# For backward compatibility
TrackingEnv = grid_environment


def create_env(config):
    """
    Factory function to create grid_environment instance.
    
    Args:
        config (dict): Environment configuration
        
    Returns:
        grid_environment: Configured environment instance
    """
    return grid_environment(env_config=config)