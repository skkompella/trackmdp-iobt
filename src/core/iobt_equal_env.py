"""
Grid Environment for Object Tracking — IoBT variant

Identical to the original iobt_new_env.py except for one targeted change:

    object_move() is overridden to give every destination (self + all
    graph neighbours) exactly equal probability, with a fixed terminal
    probability reserved above the cumulative table.

The original code padded shorter adjacency lists to a fixed length of 6
and used a single global prob_list_cum, which caused destinations to
receive unequal probability mass depending on padding count.  This
override computes a per-node cumulative probability table at __init__
time and uses it in object_move(), leaving every other method unchanged.
"""
import numpy as np
import random as rnd
from .environment import grid_env


# ---------------------------------------------------------------------------
# IoBT graph topology (0-indexed).  1-indexed labels shown in comments.
# ---------------------------------------------------------------------------
IOBT_MAP = {
    0: [1, 7, 4],        # node 1  -> 2, 8, 5
    1: [0, 7, 6, 2],     # node 2  -> 1, 8, 7, 3
    2: [1, 6, 3],        # node 3  -> 2, 7, 4
    3: [2, 5, 6],        # node 4  -> 3, 6, 7
    4: [5, 2, 3, 0, 1],  # node 5  -> 6, 3, 4, 1, 2
    5: [4, 3],           # node 6  -> 5, 4
    6: [1, 2, 8],        # node 7  -> 2, 3, 9
    7: [9, 8, 0, 1],     # node 8  -> 10, 9, 1, 2
    8: [9, 6],           # node 9  -> 10, 7
    9: [8, 7],           # node 10 -> 9, 8
}

NUM_NODES = len(IOBT_MAP)   # 10

# Terminal probability applied uniformly to every node.
TERMINAL_PROB = 0.005


class iobt_env(grid_env):
    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit):
        # Physical (x, y) coordinates of each node on the 4x4 backing grid.
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
            9: (0, 2),  # Node 10
        }

        self.N = 4
        self.num_trans = 6
        state_trans_cum_prob = [round((i + 1) / self.num_trans, 4)
                                for i in range(self.num_trans)]

        super().__init__(self.N, self.num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state, time_limit)

        # Build the per-node transition tables and override the parent matrix.
        self._node_dests, self._node_cum_probs = self._build_equal_prob_tables()
        self.obj_trans_matrix = self.object_transition_matrix_iobt()
        self.reset_object_state()

    # -----------------------------------------------------------------------
    # Core change: per-node equal-probability tables
    # -----------------------------------------------------------------------

    def _build_equal_prob_tables(self):
        """
        For each node build:

            destinations  — ordered list [self, nbr1, nbr2, ...]
            cum_probs     — cumulative thresholds of length len(destinations)

        Every destination receives probability (1 - TERMINAL_PROB) / n_dests,
        so P(terminal) = TERMINAL_PROB exactly, and all reachable nodes
        (including staying put) are equally likely.

        object_move() draws u ~ Uniform(0, 1):
          - slot = number of thresholds <= u  =>  index into destinations
          - if slot >= len(destinations)  =>  terminal (u > cum_probs[-1])
        """
        node_dests     = {}
        node_cum_probs = {}

        for node, nbrs in IOBT_MAP.items():
            dests   = [node] + nbrs          # stay + all neighbours
            n       = len(dests)
            p_each  = (1.0 - TERMINAL_PROB) / n
            cum     = [p_each * (i + 1) for i in range(n)]
            node_dests[node]     = dests
            node_cum_probs[node] = np.array(cum)

        return node_dests, node_cum_probs

    # -----------------------------------------------------------------------
    # Override: use per-node tables instead of the global prob_list_cum
    # -----------------------------------------------------------------------

    def object_move(self):
        """
        Move the object one step using equal per-destination probabilities.

        Replaces the parent's object_move(), which uses a single global
        prob_list_cum and a fixed-width padded transition matrix — causing
        unequal probability mass whenever a node has fewer than num_trans
        unique neighbours.
        """
        pos = self.object_pos

        # Terminal state is absorbing.
        if pos == self.N * self.N:
            return 0

        # Nodes outside IOBT_MAP (dead grid cells) should never be occupied,
        # but guard defensively: go terminal.
        if pos not in self._node_cum_probs:
            self.object_pos = self.N * self.N
            return 0

        cum   = self._node_cum_probs[pos]
        dests = self._node_dests[pos]

        u    = np.random.uniform(0, 1)
        slot = int(np.sum(cum <= u))  # number of thresholds not exceeded

        if slot < len(dests):
            self.object_pos = dests[slot]
        else:
            self.object_pos = self.N * self.N  # terminal

        return 0

    # -----------------------------------------------------------------------
    # Remaining overrides — unchanged from original
    # -----------------------------------------------------------------------

    def val_to_grid(self, val):
        if val in self.node_coords:
            return self.node_coords[val]
        elif val == self.N * self.N:
            return -1, -1
        else:
            return val % self.N, val // self.N

    def get_valid_q_indices(self, state, time_value):
        state_gp = state // (self.time_limit + 1)
        state_grid_x, state_grid_y = self.val_to_grid(state_gp)

        b_l = 0 - state_grid_x
        b_r = (self.N - 1) - state_grid_x
        b_d = 0 - state_grid_y
        b_u = (self.N - 1) - state_grid_y

        valid_sensors = self.check_valid_sensor(time_value, b_l, b_r, b_d, b_u)

        grid_sz  = (2 * time_value) + 3
        s_x_rel  = grid_sz // 2
        s_y_rel  = grid_sz // 2
        valid_coords = set(self.node_coords.values())

        for i in range(grid_sz ** 2):
            if valid_sensors[i] == 1:
                d_x      = i % grid_sz - s_x_rel
                d_y      = i // grid_sz - s_y_rel
                global_x = state_grid_x + d_x
                global_y = state_grid_y + d_y
                if (global_x, global_y) not in valid_coords:
                    valid_sensors[i] = 0

        return np.array(valid_sensors)

    def reset_object_state(self):
        self.object_pos = rnd.sample(list(self.node_coords.keys()), 1)[0]

    def object_transition_matrix_iobt(self):
        """
        Build the padded transition matrix expected by the parent class.

        This matrix is only used by the parent's object_move(); since we
        override object_move() entirely, this serves as documentation of
        topology and is kept for API compatibility (e.g. anything that
        inspects obj_trans_matrix directly).
        """
        N_sq = self.N * self.N
        obj_trans_matrix = []

        for i in range(N_sq):
            if i in IOBT_MAP:
                moves = [i] + IOBT_MAP[i]
                while len(moves) < self.num_trans:
                    moves.append(i)
                obj_trans_matrix.append(moves[:self.num_trans])
            else:
                obj_trans_matrix.append([i] * self.num_trans)

        obj_trans_matrix.append([N_sq] * self.num_trans)  # terminal row
        return obj_trans_matrix


# ---------------------------------------------------------------------------
# Wrapper — unchanged from original
# ---------------------------------------------------------------------------

class learning_grid_sarsa_0:
    def __init__(self, run_number, N, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max):
        self.run_number       = run_number
        self.N                = N
        self.num_trans        = num_trans
        self.prob_list_cum    = state_trans_cum_prob
        self.time_limit       = time_limit
        self.time_limit_max   = time_limit_max
        self.missing_state    = (self.N * self.N) * (self.time_limit_max + 1) + 1
        self.grid_env         = iobt_env(max_sensors, max_sensors_null,
                                         self.missing_state, time_limit)
        self.exploration_epsilon  = 0.1
        self.total_actions        = self.grid_env.action_space_size
        self.total_actions_null   = self.grid_env.action_space_size_null

        self.current_state  = self.missing_state
        self.current_action = 0
        self.next_state     = 0
        self.next_action    = 0
        self.time_delay     = 0

        self.max_sensors          = max_sensors
        self.sarsa_step_size      = 0.1
        self.exploration_epsilon  = 0.15
        self.gamma                = 1
        self.no_of_episodes       = 1

        self.episode_start        = 0
        self.file_save_directory  = "/home/ma10/documents/rl_sensor/qsave"
        self.save_directory       = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


GridEnvironment  = grid_env
TrackingLearner  = learning_grid_sarsa_0