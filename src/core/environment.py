"""
Grid Environment for Object Tracking

This module implements a grid-based environment where an object moves stochastically
and the agent must track it using sensor activations.
"""

import numpy as np
import random as rnd
from itertools import combinations
import torch
import math
import os


class grid_env:
    """
    A grid environment for object tracking with stochastic object movement.
    
    Attributes:
        N (int): Grid size (NxN grid)
        num_trans (int): Number of possible movements
        prob_list_cum (np.ndarray): Cumulative transition probabilities
        obj_trans_matrix (list): Object transition matrix
        object_pos (int): Current object position (0 to N²-1 or terminal state)
        missing_state (int): Missing state value
        time_limit (int): Maximum time delay before entering missing state
        max_sensors (int): Maximum number of sensors
        max_sensors_null (int): Maximum number of sensors for null state
        combination_dict (dict): Dictionary of sensor combinations
        cum_comb_dict (dict): Cumulative combination counts
        combination_dict_null (dict): Dictionary of null sensor combinations
        total_cum_comb_dict (list): Total cumulative combination counts
        cum_comb_dict_null (list): Cumulative null combination counts
        action_space_size (dict): Action space sizes
        action_space_size_null (int): Null action space size
        valid_q_indices_dict (dict): Valid sensor indices for each state/time
    """
    
    def __init__(self, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, missing_state, time_limit):
        """
        Initialize the grid environment.
        
        Args:
            N (int): Grid size (NxN grid)
            num_trans (int): Number of possible movements
            state_trans_cum_prob (list): Cumulative transition probabilities
            max_sensors (int): Maximum number of sensors
            max_sensors_null (int): Maximum number of sensors for null state
            missing_state (int): Missing state value
            time_limit (int): Maximum time delay
        """
        self.N = N
        self.num_trans = num_trans
        self.prob_list_cum = np.array(state_trans_cum_prob)
        self.obj_trans_matrix = self.object_transition_matrix_random_3x3()
        self.object_pos = rnd.sample(list(np.arange(N*N)), 1)[0]
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
        self.valid_q_indices_dict = self.get_valid_q_indices_dict()
        
        ### Rewards
        self.tracking_miss_rew = 0
        self.tracking_miss_rew_missing = 0
        self.sensor_rew = -0.16
        self.tracking_rew = 1
        self.tracking_rew_missing = 0
    
    #### For object
    def grid_to_val(self, x_val, y_val):
        pos = y_val * self.N + x_val
        return pos

    def val_to_grid(self, val):
        grid_y = (val//self.N) 
        grid_x = val%self.N
        return grid_x, grid_y

    def get_transition_list(self, list_vals_x, list_vals_y):
        transition_list = []
        for x, y in zip(list_vals_x, list_vals_y):
            transition_list += [self.grid_to_val(x, y)]
        return transition_list

    def object_transition_matrix(self):
        # [left , right, up , down]
        N = self.N
        obj_trans_matrix = []
        for i in range(N*N):
            grid_x, grid_y = self.val_to_grid(i)
            list_vals = []
            if grid_x == 0:
                list_vals_x = [grid_x+1, grid_x+1, grid_x, grid_x]
            elif grid_x == N-1:
                 list_vals_x = [grid_x-1, grid_x-1, grid_x, grid_x]
            
            else:
                list_vals_x = [grid_x-1, grid_x+1, grid_x, grid_x]
            
            if grid_y == 0:
                list_vals_y = [grid_y, grid_y, grid_y+1, grid_y+1]

            elif grid_y == N-1:
                list_vals_y = [grid_y, grid_y, grid_y-1, grid_y-1]

            else:
                list_vals_y = [grid_y, grid_y, grid_y+1, grid_y-1]

            transition_list = self.get_transition_list(list_vals_x, list_vals_y)

            obj_trans_matrix += [transition_list]
        obj_trans_matrix += [list(np.ones(self.num_trans, dtype=int)*(N*N))]

        return obj_trans_matrix
    
    def object_transition_matrix_random_3x3(self):
        N = self.N
        obj_trans_matrix = []
    
        for i in range(N*N):
            grid_x, grid_y = self.val_to_grid(i)
            
            # Determine valid positions in 3x3 grid around current position
            # to prevent wall collisions at boundaries
            valid_positions = []
            for rel_y in range(-1, 2):  # -1, 0, 1
                for rel_x in range(-1, 2):  # -1, 0, 1
                    new_x = grid_x + rel_x
                    new_y = grid_y + rel_y
                    
                    # Check if the new position is within grid bounds
                    if 0 <= new_x < N and 0 <= new_y < N:
                        pos_index = rel_y * 3 + rel_x + 4  # Convert to 0-8 index in 3x3 grid
                        valid_positions.append(pos_index)
            
            # Select random positions only from valid ones
            if len(valid_positions) >= self.num_trans:
                random_indices = np.random.choice(valid_positions, size=self.num_trans, replace=True)
            else:
                # If we have fewer valid positions than needed, use them with replacement
                random_indices = np.random.choice(valid_positions, size=self.num_trans, replace=True)
            
            list_vals_x = []
            list_vals_y = []
            for j in random_indices: 
                pos_rel_x, pos_rel_y = (j%3) - 1, (j//3) - 1
                new_x = grid_x + pos_rel_x
                new_y = grid_y + pos_rel_y
                
                # Since we pre-filtered for valid positions, these should always be in bounds
                list_vals_x.append(new_x)
                list_vals_y.append(new_y)

            transition_list = self.get_transition_list(list_vals_x, list_vals_y)

            obj_trans_matrix += [transition_list]
        obj_trans_matrix += [list(np.ones(self.num_trans, dtype=int)*(N*N))]

        return obj_trans_matrix
        

    def reset_object_state(self):
        self.object_pos = rnd.sample(list(np.arange(self.N*self.N)), 1)[0]

    def object_move(self):
        N = self.N
        states_mov = self.obj_trans_matrix[self.object_pos]
        unval = np.random.uniform(0, 1)
        sum_val = int(np.sum(self.prob_list_cum <= unval))
        if sum_val < self.num_trans: 
            next_state = states_mov[sum_val]
        else:
            next_state = N*N

        self.object_pos = next_state
        return 0

    ### For sensing
    def generate_combination_lists(self):
        N = self.N
        for i in range(1, self.max_sensors+1):
            self.combination_dict[i] = np.array(list(combinations(range(N*N), i)))
            v1 = len(self.combination_dict[i])
            self.action_space_size += v1
            self.cum_comb_dict += [self.cum_comb_dict[-1]+v1]
        for i in range(1, self.max_sensors_null+1):
            self.combination_dict_null[i] = np.array(list(combinations(range(N*N), i)))
            v1 = len(self.combination_dict_null[i])
            self.action_space_size_null += v1
            self.cum_comb_dict_null += [self.cum_comb_dict_null[-1]+v1]

        self.cum_comb_dict = np.array(self.cum_comb_dict)
        self.cum_comb_dict_null = np.array(self.cum_comb_dict_null)

    def generate_combination_lists_new(self):
        N = self.N

        for j in range(0, self.time_limit+1):
            # 2*j+3
            self.combination_dict[j] = {}
            self.action_space_size[j] = 0
            self.cum_comb_dict[j] = [0]
            for i in range(1, self.max_sensors+1):
                self.combination_dict[j][i] = np.array(list(combinations(range((2*j+3)*(2*j+3)), i)))
                v1 = len(self.combination_dict[j][i])
                self.action_space_size[j] += v1
                self.cum_comb_dict[j] += [self.cum_comb_dict[j][-1]+v1]
            self.cum_comb_dict[j] = np.array(self.cum_comb_dict[j])
            self.total_cum_comb_dict += [self.total_cum_comb_dict[-1] + self.cum_comb_dict[j][-1]]

        ### Just for non null
        for i in range(1, self.max_sensors_null+1):
            self.combination_dict_null[i] = np.array(list(combinations(range(N*N), i)))
            v1 = len(self.combination_dict_null[i])
            self.action_space_size_null += v1
            self.cum_comb_dict_null += [self.cum_comb_dict_null[-1]+v1]

        self.total_cum_comb_dict = np.array(self.total_cum_comb_dict)
        self.cum_comb_dict_null = np.array(self.cum_comb_dict_null)

    def check_valid_sensor(self, time_value, b_l, b_r, b_d, b_u):
        grid_sz = (2*(time_value) + 3)
        s_x_rel, s_y_rel = grid_sz//2, grid_sz//2
        valid_ind = []
        for i in range(grid_sz**2):
            d_x, d_y = i%grid_sz - s_x_rel, i//grid_sz - s_y_rel
            if (d_x >= b_l) and (d_x <= b_r) and (d_y >= b_d) and (d_y <= b_u):
                valid_ind += [1]
            else:
                valid_ind += [0]
        
        return valid_ind

    def get_valid_q_indices(self, state, time_value):
        
        state_gp = state//(self.time_limit + 1)
        state_grid_x, state_grid_y = self.val_to_grid(state_gp)
        b_l, b_r = 0 - state_grid_x, (self.N - 1) - state_grid_x
        b_d, b_u = 0 - state_grid_y, (self.N - 1) - state_grid_y
        
        valid_sensors = self.check_valid_sensor(time_value, b_l, b_r, b_d, b_u)
        
        return np.array(valid_sensors)
    
    def get_valid_q_indices_dict(self):
        valid_dict = {}
        for i in range(self.time_limit+1):
            valid_dict[i] = {}
            for j in range(0, (self.N**2)*(self.time_limit+1)):
                valid_dict[i][j] = self.get_valid_q_indices(j, i)
            
        return valid_dict
        
    def realign_obj(self, object_pos, current_state, time_delay):
        
        current_state_gp = current_state//(self.time_limit+1)
        grid_rad = ((2*time_delay) + 3)//2
        current_state_x, current_state_y = self.val_to_grid(current_state_gp)
        object_pos_x, object_pos_y = self.val_to_grid(object_pos)
        diff_x, diff_y = (object_pos_x - current_state_x), (object_pos_y - current_state_y)
        obj_in_grid = 1
        if (np.abs(diff_x) > grid_rad) or (np.abs(diff_y) > grid_rad):
            obj_in_grid = 0
        
        current_state_rel_pt = (grid_rad*(2*grid_rad +1)) + grid_rad
        obj_rel_pos = ((diff_y* (2*grid_rad +1)) + diff_x) + current_state_rel_pt

        return obj_rel_pos, obj_in_grid
        
    def get_reward_next_state(self, current_state, current_action, time_delay):
        obj_position = self.object_pos
        obj_found = 0
        
        no_of_time_sensors = (2*(time_delay)+3)**2
        if current_state != self.missing_state:

            no_of_time_sensors = (2*(time_delay)+3)**2
            current_action_clip = current_action[-no_of_time_sensors:] 
    
            current_action_sensors = np.multiply(current_action_clip, self.valid_q_indices_dict[time_delay][current_state])

            obj_rel_pos, obj_in_grid = self.realign_obj(self.object_pos, current_state, time_delay)
            
        else:
            obj_rel_pos, obj_in_grid = 0, 1
            current_action_sensors = [1]*no_of_time_sensors
        
        if (obj_in_grid == 1) and (current_action_sensors[int(obj_rel_pos)] == 1):
            obj_found = 1
            next_state = obj_position*(self.time_limit+1)
            time_delay_sense = 0

        else:
            time_delay_sense = time_delay + 1
            if current_state != self.missing_state:
                if time_delay_sense > self.time_limit: 
                    next_state = self.missing_state
                else:
                    next_state = current_state +1 
            else:
                next_state = self.missing_state

        self.object_move()
        no_sensor_on = np.sum(current_action_sensors)
        
        if current_state != self.missing_state:
            reward = obj_found*self.tracking_rew + \
                 (1-obj_found)*self.tracking_miss_rew + no_sensor_on*self.sensor_rew
        else: 
            reward = obj_found*self.tracking_rew_missing + \
                 (1-obj_found)*self.tracking_miss_rew_missing + (self.N**2)*self.sensor_rew

        terminal_st_obj = 0
        if self.object_pos == self.N*self.N:
            terminal_st_obj = 1

        return reward, next_state, terminal_st_obj, time_delay_sense


class learning_grid_sarsa_0:
    def __init__(self, run_number, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit, time_limit_max):
        self.run_number = run_number
        self.N = N
        self.num_trans = num_trans
        self.prob_list_cum = state_trans_cum_prob
        self.time_limit = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state = ((self.N*self.N)*(self.time_limit_max+1) + 1)
        self.grid_env = grid_env(N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, self.missing_state, time_limit)
        self.exploration_epsilon = 0.1
        self.total_actions = self.grid_env.action_space_size
        self.total_actions_null = self.grid_env.action_space_size_null

        self.current_state = self.missing_state # start with absolutely no knowledge state
        self.current_action = 0
        self.next_state = 0 
        self.next_action = 0 
        self.time_delay = 0
        
        self.max_sensors = max_sensors
        self.sarsa_step_size = 0.1
        self.exploration_epsilon = 0.15
        self.gamma = 1
        self.no_of_episodes = 1
        
        ### Save file variables
        self.episode_start = 0
        self.file_save_directory = "/home/ma10/documents/rl_sensor/qsave"
        self.save_directory = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# For backward compatibility, create aliases
GridEnvironment = grid_env
TrackingLearner = learning_grid_sarsa_0


def demo_environment():
    """Demonstration of the grid environment."""
    run_number = 9999
    N, num_trans = 10, 2
    terminal_st_prob = 0.005
    state_prob_run = 0.15
    state_trans_cum_prob = [i*(1-terminal_st_prob-state_prob_run )/float(num_trans-1) for i in range(1, num_trans)]
    state_trans_cum_prob += [state_trans_cum_prob[-1]+ state_prob_run] 
    max_sensors, max_sensors_null = 6, 6
    time_limit_start = 1 
    time_limit_max = 1
    
    qobj = learning_grid_sarsa_0(run_number, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit_start, time_limit_max)
    
    print("Grid Environment Demo")
    print(f"Grid Size: {qobj.N}x{qobj.N}")
    print(f"Initial Object Position: {qobj.grid_env.object_pos}")
    
    for step in range(10):
        if qobj.grid_env.object_pos == qobj.N*qobj.N:
            print("\nObject reached terminal state. Resetting...")
            qobj.grid_env.reset_object_state()
        
        print(f"\nStep {step + 1}: Object at position {qobj.grid_env.object_pos}")
        row, col = qobj.grid_env.val_to_grid(qobj.grid_env.object_pos)
        
        # Visualize grid
        grid = np.zeros((qobj.N, qobj.N))
        if qobj.grid_env.object_pos < qobj.N*qobj.N:
            grid[row, col] = 1
        print(grid)
        
        qobj.grid_env.object_move()
        input("Press Enter to continue...")


if __name__ == '__main__':
    demo_environment()