import numpy as np
import random as rnd
from itertools import combinations
import torch
import math
import os

class iobt_env:
    """
    An environment tailored for the IoBT-MAX testbed at R2C2.
    It represents a K=10 fully connected graph rather than a spatial grid.
    The object can move from any of the 10 nodes to any other node, or exit (terminal state).
    """
    def __init__(self, num_nodes, state_trans_cum_prob, max_sensors, max_sensors_null, missing_state, time_limit):
        self.num_nodes = num_nodes # 10 nodes
        self.prob_list_cum = np.array(state_trans_cum_prob)
        
        # The transition matrix allows movement from any node to any node
        self.obj_trans_matrix = self.object_transition_matrix_fully_connected()
        
        # Object position is 0 to (num_nodes - 1). The terminal state is num_nodes (10)
        self.object_pos = rnd.sample(list(np.arange(self.num_nodes)), 1)[0]
        self.missing_state = missing_state
        
        self.time_limit = time_limit
        self.max_sensors = max_sensors
        self.max_sensors_null = max_sensors_null

        self.combination_dict = {}
        self.cum_comb_dict = {}
        self.combination_dict_null = {}
        self.total_cum_comb_dict = [0]
        self.cum_comb_dict_null = [0]
        
        self.action_space_size = {}
        self.action_space_size_null = 0
        
        # We don't need a complex valid_q_indices_dict because the graph isn't spatial.
        # Every sensor is "valid" at any time.
        self.valid_q_indices_dict = self.get_valid_q_indices_dict()
        
        ### Rewards (Tuned for IoBT-MAX)
        self.tracking_miss_rew = -0.5
        self.tracking_miss_rew_missing = -1.0
        self.sensor_rew = -0.25 # High power Orin NX penalty
        self.tracking_rew = 1.5
        self.tracking_rew_missing = 0.5

    def object_transition_matrix_fully_connected(self):
        """
        Creates a fully connected graph transition matrix.
        From any state, the object can transition to any of the 10 nodes.
        The last entry represents the terminal state.
        """
        obj_trans_matrix = []
        for i in range(self.num_nodes):
            # The object can move to any node (0-9)
            transition_list = list(range(self.num_nodes))
            obj_trans_matrix.append(transition_list)
        
        # Terminal state transitions to itself
        obj_trans_matrix.append(list(np.ones(self.num_nodes, dtype=int) * self.num_nodes))
        return obj_trans_matrix

    def reset_object_state(self):
        self.object_pos = rnd.sample(list(np.arange(self.num_nodes)), 1)[0]

    def object_move(self):
        """
        Moves the object. Since it's fully connected, it picks a random node
        or exits based on the cumulative probabilities.
        """
        states_mov = self.obj_trans_matrix[self.object_pos]
        
        # If already in terminal state, stay there
        if self.object_pos == self.num_nodes:
            return 0
            
        unval = np.random.uniform(0, 1)
        
        # The probability list should sum to 1.0. 
        # For a K10 graph, we assume uniform probability (0.1) for each node.
        # However, to allow a terminal exit, we use the passed prob_list_cum.
        sum_val = int(np.sum(self.prob_list_cum <= unval))
        
        if sum_val < len(states_mov): 
            next_state = states_mov[sum_val]
        else:
            # Exit to terminal state
            next_state = self.num_nodes

        self.object_pos = next_state
        return 0

    def generate_combination_lists_new(self):
        """
        Simplified combination generation for a non-spatial graph.
        We generate combinations of the 10 nodes.
        """
        for j in range(0, self.time_limit+1):
            self.combination_dict[j] = {}
            self.action_space_size[j] = 0
            self.cum_comb_dict[j] = [0]
            
            # Combinations from size 1 up to max_sensors out of the 10 nodes
            for i in range(1, self.max_sensors+1):
                self.combination_dict[j][i] = np.array(list(combinations(range(self.num_nodes), i)))
                v1 = len(self.combination_dict[j][i])
                self.action_space_size[j] += v1
                self.cum_comb_dict[j].append(self.cum_comb_dict[j][-1] + v1)
                
            self.cum_comb_dict[j] = np.array(self.cum_comb_dict[j])
            self.total_cum_comb_dict.append(self.total_cum_comb_dict[-1] + self.cum_comb_dict[j][-1])

        ### Null combinations
        for i in range(1, self.max_sensors_null+1):
            self.combination_dict_null[i] = np.array(list(combinations(range(self.num_nodes), i)))
            v1 = len(self.combination_dict_null[i])
            self.action_space_size_null += v1
            self.cum_comb_dict_null.append(self.cum_comb_dict_null[-1] + v1)

        self.total_cum_comb_dict = np.array(self.total_cum_comb_dict)
        self.cum_comb_dict_null = np.array(self.cum_comb_dict_null)

    def get_valid_q_indices_dict(self):
        """
        In a fully connected graph, all sensors are valid at all times.
        There is no 'spatial bounding box' to worry about.
        """
        valid_dict = {}
        for i in range(self.time_limit+1):
            valid_dict[i] = {}
            # state ranges up to num_nodes * (time_limit + 1)
            for j in range(0, (self.num_nodes) * (self.time_limit+1) + 2):
                valid_dict[i][j] = np.ones(self.num_nodes, dtype=int)
        return valid_dict

    def get_reward_next_state(self, current_state, current_action, time_delay):
        """
        Calculates reward based on whether the chosen action hit the current node.
        """
        obj_position = self.object_pos
        obj_found = 0
        
        # current_action is expected to be a binary array of length num_nodes [0,1,0,0...]
        # where 1 indicates the sensor is turned on.
        current_action_sensors = current_action[-self.num_nodes:] 
        
        if current_state != self.missing_state:
            # Check if the object is in the grid (not terminal) and the corresponding sensor is ON
            if (obj_position < self.num_nodes) and (current_action_sensors[obj_position] == 1):
                obj_found = 1
                next_state = obj_position * (self.time_limit + 1)
                time_delay_sense = 0
            else:
                time_delay_sense = time_delay + 1
                if time_delay_sense > self.time_limit: 
                    next_state = self.missing_state
                else:
                    next_state = current_state + 1 
        else:
            # If we are in the missing state, check if we found it
            if (obj_position < self.num_nodes) and (current_action_sensors[obj_position] == 1):
                obj_found = 1
                next_state = obj_position * (self.time_limit + 1)
                time_delay_sense = 0
            else:
                time_delay_sense = time_delay + 1
                next_state = self.missing_state

        # Move object for the next step
        self.object_move()
        
        # Calculate Reward
        no_sensor_on = np.sum(current_action_sensors)
        
        if current_state != self.missing_state:
            reward = (obj_found * self.tracking_rew) + \
                     ((1 - obj_found) * self.tracking_miss_rew) + \
                     (no_sensor_on * self.sensor_rew)
        else: 
            reward = (obj_found * self.tracking_rew_missing) + \
                     ((1 - obj_found) * self.tracking_miss_rew_missing) + \
                     (no_sensor_on * self.sensor_rew)

        terminal_st_obj = 1 if self.object_pos == self.num_nodes else 0

        return reward, next_state, terminal_st_obj, time_delay_sense


class learning_iobt_sarsa:
    """
    Wrapper for the IoBT environment, mirroring the learning_grid_sarsa_0 structure.
    """
    def __init__(self, run_number, num_nodes, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit, time_limit_max):
        self.run_number = run_number
        self.num_nodes = num_nodes
        self.prob_list_cum = state_trans_cum_prob
        self.time_limit = time_limit
        self.time_limit_max = time_limit_max
        
        # Missing state is placed after all possible node/time combinations
        self.missing_state = (self.num_nodes * (self.time_limit_max + 1) + 1)
        
        self.grid_env = iobt_env(num_nodes, state_trans_cum_prob, max_sensors, max_sensors_null, self.missing_state, time_limit)
        
        # Generate the action space
        self.grid_env.generate_combination_lists_new()
        
        self.exploration_epsilon = 0.1
        self.total_actions = self.grid_env.action_space_size[0] # taking baseline
        self.total_actions_null = self.grid_env.action_space_size_null

        self.current_state = self.missing_state
        self.current_action = 0
        self.next_state = 0 
        self.next_action = 0 
        self.time_delay = 0
        
        self.max_sensors = max_sensors
        self.sarsa_step_size = 0.1
        self.gamma = 1
        self.no_of_episodes = 1