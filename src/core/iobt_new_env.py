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


import numpy as np
import random as rnd
from .environment import grid_env

class iobt_env(grid_env):
    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit):
        # 1. Define the physical coordinate layout of the 10 nodes
        # Keys = node indices (0 to 9), Values = (x, y) coordinates on a 4x4 grid.
        # You should tweak these coordinates to reflect the actual GPS/spacing of the facility.
        self.node_coords = {
            0: (0, 1),  # Node 1
            1: (1, 1),  # Node 2
            2: (2, 1),  # Node 3
            3: (3, 1),  # Node 4
            4: (1, 2),  # Node 5
            5: (2, 2),  # Node 6
            6: (2, 0),  # Node 7
            7: (0, 0),  # Node 8
            8: (1, 0),  # Node 9
            9: (0, 2)   # Node 10
        }
        
        # Setup constants
        self.N = 4
        self.num_trans = 6
        state_trans_cum_prob = [round((i+1)/self.num_trans, 4) 
                                for i in range(self.num_trans)]
        
        # Initialize parent (which will automatically call our overridden val_to_grid)
        super().__init__(self.N, self.num_trans, state_trans_cum_prob, 
                         max_sensors, max_sensors_null, missing_state, time_limit)
        
        # Override the transition matrix and initial spawn point
        self.obj_trans_matrix = self.object_transition_matrix_iobt()
        self.reset_object_state()

    def val_to_grid(self, val):
        """Override to place specific nodes at designated physical coordinates."""
        if val in self.node_coords:
            return self.node_coords[val]
        elif val == self.N * self.N:
            return -1, -1  # Terminal state
        else:
            # Fallback for the 6 "dead" cells (10-15) if queried internally
            return val % self.N, val // self.N

    def get_valid_q_indices(self, state, time_value):
        """Override to mask out any coordinates that do not contain an actual node."""
        state_gp = state // (self.time_limit + 1)
        state_grid_x, state_grid_y = self.val_to_grid(state_gp)
        
        # Calculate grid boundaries
        b_l, b_r = 0 - state_grid_x, (self.N - 1) - state_grid_x
        b_d, b_u = 0 - state_grid_y, (self.N - 1) - state_grid_y
        
        # Get base valid sensors (checking against the global 4x4 boundaries)
        valid_sensors = self.check_valid_sensor(time_value, b_l, b_r, b_d, b_u)
        
        # Spatial logic twist: Mask out any cell that isn't a functional node
        grid_sz = (2 * time_value) + 3
        s_x_rel, s_y_rel = grid_sz // 2, grid_sz // 2
        valid_coords = set(self.node_coords.values())
        
        for i in range(grid_sz**2):
            if valid_sensors[i] == 1:
                d_x, d_y = i % grid_sz - s_x_rel, i // grid_sz - s_y_rel
                global_x = state_grid_x + d_x
                global_y = state_grid_y + d_y
                
                # If the coordinate doesn't belong to a node, disable sensor placement
                if (global_x, global_y) not in valid_coords:
                    valid_sensors[i] = 0 
                    
        return np.array(valid_sensors)

    def reset_object_state(self):
        """Ensure the object only spawns on valid functional nodes."""
        self.object_pos = rnd.sample(list(self.node_coords.keys()), 1)[0]

    def object_transition_matrix_iobt(self):
        # Your custom map connections
        iobt_map = {
            0: [1, 7, 4],       # Node 1
            1: [0, 7, 6, 2],    # Node 2
            2: [1, 6, 3],       # Node 3
            3: [2, 5, 6],       # Node 4
            4: [5, 2, 3, 0, 1], # Node 5
            5: [4, 3],          # Node 6
            6: [1, 2, 8],       # Node 7
            7: [9, 8, 0, 1],    # Node 8
            8: [9, 6],          # Node 9
            9: [8, 7]           # Node 10
        }
        
        N_sq = self.N * self.N
        obj_trans_matrix = []

        for i in range(N_sq):
            if i in iobt_map:
                moves = [i] + iobt_map[i]
                # Pad to match num_trans (6)
                while len(moves) < self.num_trans:
                    moves.append(i)
                transition_list = moves[:self.num_trans]
            else:
                transition_list = [i] * self.num_trans
            
            obj_trans_matrix.append(transition_list)

        # Terminal state transition
        obj_trans_matrix.append([N_sq] * self.num_trans)
        return obj_trans_matrix


class learning_grid_sarsa_0:
    def __init__(self, run_number, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit, time_limit_max):
        self.run_number = run_number
        self.N = N
        self.num_trans = num_trans
        self.prob_list_cum = state_trans_cum_prob
        self.time_limit = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state = ((self.N*self.N)*(self.time_limit_max+1) + 1)
        self.grid_env = iobt_env(max_sensors, max_sensors_null, self.missing_state, time_limit)
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