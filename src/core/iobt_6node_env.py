"""
iobt_6node_env.py — IoBT environment variant for a fully-connected 6-node graph.

Every node is connected to every other node.  The object at any node moves
to each of the 6 destinations (stay-in-place + 5 neighbours) with equal
probability, with a fixed terminal probability reserved above the cumulative
table.

    P(move to any single destination) = (1 - TERMINAL_PROB) / 6  ≈ 0.1658
    P(terminal)                        = TERMINAL_PROB             = 0.005

Backing grid: 3×3 (N=3).  The 6 nodes occupy the bottom two rows:
    node 0 = (0,0)   node 1 = (1,0)   node 2 = (2,0)
    node 3 = (0,1)   node 4 = (1,1)   node 5 = (2,1)

The top row (y=2) contains no active nodes; the sensor-masking logic in
get_valid_q_indices() removes those cells from the action space automatically.

State encoding (inherited from grid_env):
    state = node_idx * (time_limit + 1) + time_component
    missing_state = N*N * (time_limit_max + 1) + 1  =  9 * 2 + 1  =  19
"""

import numpy as np
import random as rnd
from .environment import grid_env


# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------

# Fully-connected: every node connects to all other 5 nodes.
IOBT_6_MAP = {
    0: [1, 2, 3, 4, 5],   # node 1
    1: [0, 2, 3, 4, 5],   # node 2
    2: [0, 1, 3, 4, 5],   # node 3
    3: [0, 1, 2, 4, 5],   # node 4
    4: [0, 1, 2, 3, 5],   # node 5
    5: [0, 1, 2, 3, 4],   # node 6
}

NUM_NODES     = len(IOBT_6_MAP)   # 6
TERMINAL_PROB = 0.005


class iobt_6node_env(grid_env):
    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit):

        # Physical (x, y) coordinates on the 3×3 backing grid.
        # Row-major layout: node i -> (i % 3, i // 3), so coords are exact.
        self.node_coords = {
            0: (0, 0),   # node 1
            1: (1, 0),   # node 2
            2: (2, 0),   # node 3
            3: (0, 1),   # node 4
            4: (1, 1),   # node 5
            5: (2, 1),   # node 6
        }

        self.N         = 3                # 3×3 backing grid
        self.num_trans = 6                # self + 5 neighbours — no padding needed

        # Uniform cumulative probs for the parent's default object_move
        # (overridden below, but required by super().__init__).
        state_trans_cum_prob = [round((i + 1) / self.num_trans, 4)
                                for i in range(self.num_trans)]

        super().__init__(self.N, self.num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state, time_limit)

        # Per-node equal-probability tables (override parent's global cum prob).
        self._node_dests, self._node_cum_probs = self._build_equal_prob_tables()
        self.obj_trans_matrix = self._build_transition_matrix()
        self.reset_object_state()

    # -----------------------------------------------------------------------
    # Equal-probability transition tables
    # -----------------------------------------------------------------------

    def _build_equal_prob_tables(self):
        """
        For each node build:
            destinations  = [self, nbr1, nbr2, nbr3, nbr4, nbr5]   (length 6)
            cum_probs     = cumulative thresholds, length 6

        Every destination receives (1 - TERMINAL_PROB) / 6 ≈ 0.1658.
        A uniform draw above cum_probs[-1] triggers the terminal transition.
        """
        node_dests     = {}
        node_cum_probs = {}

        for node, nbrs in IOBT_6_MAP.items():
            dests   = [node] + nbrs          # 1 self + 5 neighbours = 6
            n       = len(dests)             # always 6
            p_each  = (1.0 - TERMINAL_PROB) / n
            cum     = [p_each * (i + 1) for i in range(n)]
            node_dests[node]     = dests
            node_cum_probs[node] = np.array(cum)

        return node_dests, node_cum_probs

    # -----------------------------------------------------------------------
    # Override: equal-probability object_move
    # -----------------------------------------------------------------------

    def object_move(self):
        """
        Move the object one step using per-node equal-probability tables.

        Each of the 6 destinations (stay + 5 neighbours) is equally likely.
        A draw above the last cumulative threshold triggers the terminal state.
        """
        pos = self.object_pos

        if pos == self.N * self.N:          # already terminal — absorbing
            return 0

        if pos not in self._node_cum_probs: # dead grid cell — go terminal
            self.object_pos = self.N * self.N
            return 0

        cum   = self._node_cum_probs[pos]
        dests = self._node_dests[pos]

        u    = np.random.uniform(0, 1)
        slot = int(np.sum(cum <= u))

        if slot < len(dests):
            self.object_pos = dests[slot]
        else:
            self.object_pos = self.N * self.N   # terminal

        return 0

    # -----------------------------------------------------------------------
    # Remaining overrides
    # -----------------------------------------------------------------------

    def val_to_grid(self, val):
        if val in self.node_coords:
            return self.node_coords[val]
        elif val == self.N * self.N:
            return -1, -1                    # terminal
        else:
            return val % self.N, val // self.N

    def get_valid_q_indices(self, state, time_value):
        """Mask sensor positions that do not correspond to an active node."""
        state_gp = state // (self.time_limit + 1)
        state_grid_x, state_grid_y = self.val_to_grid(state_gp)

        b_l = 0 - state_grid_x
        b_r = (self.N - 1) - state_grid_x
        b_d = 0 - state_grid_y
        b_u = (self.N - 1) - state_grid_y

        valid_sensors = self.check_valid_sensor(time_value, b_l, b_r, b_d, b_u)

        grid_sz      = (2 * time_value) + 3
        s_x_rel      = grid_sz // 2
        s_y_rel      = grid_sz // 2
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
        """Spawn the object at a random active node."""
        self.object_pos = rnd.choice(list(self.node_coords.keys()))

    def _build_transition_matrix(self):
        """
        Build the padded transition matrix for API compatibility.

        The parent's object_move() is never called (we override it), so this
        matrix is documentation only.  Every active-node row is exactly
        [self, nbr1, nbr2, nbr3, nbr4, nbr5] — no padding required since
        num_trans == 6 == len(dests) for every node.
        """
        N_sq = self.N * self.N
        matrix = []

        for i in range(N_sq):
            if i in IOBT_6_MAP:
                row = [i] + IOBT_6_MAP[i]  # length 6, exactly num_trans
            else:
                row = [i] * self.num_trans  # dead cell — stays put
            matrix.append(row)

        matrix.append([N_sq] * self.num_trans)   # terminal row
        return matrix


# ---------------------------------------------------------------------------
# Wrapper (mirrors learning_grid_sarsa_0 from iobt_equal_env.py)
# ---------------------------------------------------------------------------

class learning_grid_sarsa_0:
    def __init__(self, run_number, N, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max):
        self.run_number      = run_number
        self.N               = N
        self.num_trans       = num_trans
        self.prob_list_cum   = state_trans_cum_prob
        self.time_limit      = time_limit
        self.time_limit_max  = time_limit_max
        self.missing_state   = (self.N * self.N) * (self.time_limit_max + 1) + 1
        self.grid_env        = iobt_6node_env(
                                   max_sensors, max_sensors_null,
                                   self.missing_state, time_limit)

        self.exploration_epsilon = 0.1
        self.total_actions       = self.grid_env.action_space_size
        self.total_actions_null  = self.grid_env.action_space_size_null

        self.current_state  = self.missing_state
        self.current_action = 0
        self.next_state     = 0
        self.next_action    = 0
        self.time_delay     = 0

        self.max_sensors         = max_sensors
        self.sarsa_step_size     = 0.1
        self.exploration_epsilon = 0.15
        self.gamma               = 1
        self.no_of_episodes      = 1

        self.episode_start       = 0
        self.file_save_directory = "/home/ma10/documents/rl_sensor/qsave"
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


GridEnvironment = grid_env
TrackingLearner = learning_grid_sarsa_0