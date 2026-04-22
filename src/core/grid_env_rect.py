# """
# grid_env_rect.py — Grid environment for object tracking on a rectangular M×N grid.

# Drop-in replacement for environment.py / grid_env, supporting non-square grids.
# The interface is identical except:
#   - __init__ takes nrows and ncols instead of a single N.
#   - learning_grid_sarsa_0 takes nrows and ncols (N is kept as an alias for ncols
#     so gym_wrapper.py continues to work without modification).

# Default grid: 2 rows × 3 cols.

# State encoding (unchanged from original):
#     state = cell_idx * (time_limit + 1) + time_component
#     cell_idx ∈ [0, NROWS*NCOLS - 1]  (row-major: idx = row*NCOLS + col)
#     terminal state = NROWS*NCOLS
#     missing_state  = NROWS*NCOLS*(time_limit_max+1) + 1
# """

# import numpy as np
# import random as rnd
# from itertools import combinations


# class grid_env_rect:
#     """
#     Rectangular M×N grid environment for object tracking.

#     Parameters
#     ----------
#     nrows : int   — number of rows    (M)
#     ncols : int   — number of columns (N)
#     num_trans, state_trans_cum_prob, max_sensors, max_sensors_null,
#     missing_state, time_limit — same as original grid_env
#     """

#     def __init__(self, nrows, ncols, num_trans, state_trans_cum_prob,
#                  max_sensors, max_sensors_null, missing_state, time_limit):
#         self.nrows = nrows
#         self.ncols = ncols
#         self.N     = ncols          # gym_wrapper reads self.N for obs-space size
#         self.n_cells = nrows * ncols

#         self.num_trans      = num_trans
#         self.prob_list_cum  = np.array(state_trans_cum_prob)
#         self.missing_state  = missing_state
#         self.time_limit     = time_limit
#         self.max_sensors    = max_sensors
#         self.max_sensors_null = max_sensors_null

#         self.obj_trans_matrix = self._build_transition_matrix()
#         self.object_pos = rnd.randrange(self.n_cells)

#         self.combination_dict      = {}
#         self.cum_comb_dict         = {}
#         self.combination_dict_null = {}
#         self.total_cum_comb_dict   = [0]
#         self.cum_comb_dict_null    = [0]

#         self.action_space_size      = {}
#         self.action_space_size_null = 0

#         self.valid_q_indices_dict = self.get_valid_q_indices_dict()

#         # Rewards (identical to original)
#         self.tracking_miss_rew         = 0
#         self.tracking_miss_rew_missing = 0
#         self.sensor_rew                = -0.16
#         self.tracking_rew              = 1
#         self.tracking_rew_missing      = 0

#     # -----------------------------------------------------------------------
#     # Coordinate helpers
#     # -----------------------------------------------------------------------

#     def grid_to_val(self, col, row):
#         """(col, row) -> flat cell index  (col = x, row = y)."""
#         return row * self.ncols + col

#     def val_to_grid(self, val):
#         """Flat cell index -> (col, row)  (col = x, row = y)."""
#         return val % self.ncols, val // self.ncols

#     # -----------------------------------------------------------------------
#     # Transition matrix
#     # -----------------------------------------------------------------------

#     def _build_transition_matrix(self):
#         """
#         Random 3×3-neighbourhood transition matrix, respecting rectangular bounds.

#         For each cell, num_trans destinations are sampled (with replacement) from
#         the valid Moore-neighbourhood cells (those within the grid).
#         """
#         matrix = []
#         for idx in range(self.n_cells):
#             col, row = self.val_to_grid(idx)

#             # Collect valid neighbour indices (Moore neighbourhood + self)
#             valid = []
#             for dr in range(-1, 2):        # row offset
#                 for dc in range(-1, 2):    # col offset
#                     nc, nr = col + dc, row + dr
#                     if 0 <= nc < self.ncols and 0 <= nr < self.nrows:
#                         valid.append(self.grid_to_val(nc, nr))

#             chosen = np.random.choice(valid, size=self.num_trans, replace=True)
#             matrix.append(chosen.tolist())

#         # Terminal state: absorbing
#         matrix.append([self.n_cells] * self.num_trans)
#         return matrix

#     # -----------------------------------------------------------------------
#     # Object movement
#     # -----------------------------------------------------------------------

#     def reset_object_state(self):
#         self.object_pos = rnd.randrange(self.n_cells)

#     def object_move(self):
#         moves  = self.obj_trans_matrix[self.object_pos]
#         u      = np.random.uniform(0, 1)
#         slot   = int(np.sum(self.prob_list_cum <= u))
#         if slot < self.num_trans:
#             self.object_pos = moves[slot]
#         else:
#             self.object_pos = self.n_cells   # terminal
#         return 0

#     # -----------------------------------------------------------------------
#     # Sensor validity
#     # -----------------------------------------------------------------------

#     def check_valid_sensor(self, time_value, b_l, b_r, b_d, b_u):
#         """Return a flat list of 1/0 for each cell in the (2t+3)² window."""
#         grid_sz = 2 * time_value + 3
#         cx, cy  = grid_sz // 2, grid_sz // 2
#         valid   = []
#         for i in range(grid_sz * grid_sz):
#             dx, dy = i % grid_sz - cx, i // grid_sz - cy
#             if b_l <= dx <= b_r and b_d <= dy <= b_u:
#                 valid.append(1)
#             else:
#                 valid.append(0)
#         return valid

#     def get_valid_q_indices(self, state, time_value):
#         """Valid sensor positions for a given Track-MDP state and time delay."""
#         state_cell    = state // (self.time_limit + 1)
#         col, row      = self.val_to_grid(state_cell)
#         b_l = 0   - col;  b_r = (self.ncols - 1) - col
#         b_d = 0   - row;  b_u = (self.nrows - 1) - row
#         return np.array(self.check_valid_sensor(time_value, b_l, b_r, b_d, b_u))

#     def get_valid_q_indices_dict(self):
#         valid_dict = {}
#         for t in range(self.time_limit + 1):
#             valid_dict[t] = {}
#             for s in range(self.n_cells * (self.time_limit + 1)):
#                 valid_dict[t][s] = self.get_valid_q_indices(s, t)
#         return valid_dict

#     # -----------------------------------------------------------------------
#     # Object alignment
#     # -----------------------------------------------------------------------

#     def realign_obj(self, object_pos, current_state, time_delay):
#         """Return (relative_position_in_window, in_window_flag)."""
#         state_cell = current_state // (self.time_limit + 1)
#         grid_rad   = (2 * time_delay + 3) // 2

#         s_col, s_row = self.val_to_grid(state_cell)
#         o_col, o_row = self.val_to_grid(object_pos)
#         dx, dy       = o_col - s_col, o_row - s_row

#         if abs(dx) > grid_rad or abs(dy) > grid_rad:
#             in_window = 0
#         else:
#             in_window = 1

#         window_w   = 2 * grid_rad + 1
#         centre_idx = grid_rad * window_w + grid_rad
#         rel_pos    = centre_idx + dy * window_w + dx
#         return rel_pos, in_window

#     # -----------------------------------------------------------------------
#     # Reward / next state
#     # -----------------------------------------------------------------------

#     def get_reward_next_state(self, current_state, current_action, time_delay):
#         obj_position      = self.object_pos
#         obj_found         = 0
#         no_of_time_sensors = (2 * time_delay + 3) ** 2

#         if current_state != self.missing_state:
#             action_clip    = current_action[-no_of_time_sensors:]
#             action_sensors = np.multiply(
#                 action_clip,
#                 self.valid_q_indices_dict[time_delay][current_state]
#             )
#             obj_rel_pos, obj_in_grid = self.realign_obj(
#                 self.object_pos, current_state, time_delay
#             )
#         else:
#             obj_rel_pos, obj_in_grid = 0, 1
#             action_sensors = [1] * no_of_time_sensors

#         if obj_in_grid == 1 and action_sensors[int(obj_rel_pos)] == 1:
#             obj_found        = 1
#             next_state       = obj_position * (self.time_limit + 1)
#             time_delay_sense = 0
#         else:
#             time_delay_sense = time_delay + 1
#             if current_state != self.missing_state:
#                 if time_delay_sense > self.time_limit:
#                     next_state = self.missing_state
#                 else:
#                     next_state = current_state + 1
#             else:
#                 next_state = self.missing_state

#         self.object_move()
#         no_sensor_on = np.sum(action_sensors)

#         if current_state != self.missing_state:
#             reward = (obj_found * self.tracking_rew
#                       + (1 - obj_found) * self.tracking_miss_rew
#                       + no_sensor_on * self.sensor_rew)
#         else:
#             reward = (obj_found * self.tracking_rew_missing
#                       + (1 - obj_found) * self.tracking_miss_rew_missing
#                       + self.n_cells * self.sensor_rew)

#         terminal = 1 if self.object_pos == self.n_cells else 0
#         return reward, next_state, terminal, time_delay_sense


# class learning_grid_sarsa_0:
#     """
#     Wrapper matching the original interface.

#     Parameters
#     ----------
#     nrows, ncols : grid dimensions  (replaces the single N parameter)
#     All other parameters identical to the original learning_grid_sarsa_0.
#     """

#     def __init__(self, run_number, nrows, ncols, num_trans, state_trans_cum_prob,
#                  max_sensors, max_sensors_null, time_limit, time_limit_max):
#         self.run_number     = run_number
#         self.nrows          = nrows
#         self.ncols          = ncols
#         self.N              = ncols         # keep for gym_wrapper compatibility
#         self.n_cells        = nrows * ncols
#         self.num_trans      = num_trans
#         self.prob_list_cum  = state_trans_cum_prob
#         self.time_limit     = time_limit
#         self.time_limit_max = time_limit_max
#         self.missing_state  = self.n_cells * (self.time_limit_max + 1) + 1

#         self.grid_env = grid_env_rect(
#             nrows, ncols, num_trans, state_trans_cum_prob,
#             max_sensors, max_sensors_null, self.missing_state, time_limit
#         )

#         self.exploration_epsilon = 0.15
#         self.total_actions       = self.grid_env.action_space_size
#         self.total_actions_null  = self.grid_env.action_space_size_null
#         self.current_state       = self.missing_state
#         self.current_action      = 0
#         self.next_state          = 0
#         self.next_action         = 0
#         self.time_delay          = 0
#         self.max_sensors         = max_sensors
#         self.sarsa_step_size     = 0.1
#         self.gamma               = 1
#         self.no_of_episodes      = 1
#         self.episode_start       = 0
#         self.file_save_directory = "/home/ma10/documents/rl_sensor/qsave"
#         self.save_directory      = None

#     def update_time_limit(self, new_time_limit):
#         self.time_limit = new_time_limit
#         self.grid_env.time_limit = new_time_limit
#         self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()



"""
grid_env_rect.py — Grid environment for object tracking on a rectangular M×N grid.

Drop-in replacement for environment.py / grid_env, supporting non-square grids.
The interface is identical except:
  - __init__ takes nrows and ncols instead of a single N.
  - learning_grid_sarsa_0 takes nrows and ncols (N is kept as an alias for ncols
    so gym_wrapper.py continues to work without modification).

Default grid: 2 rows × 3 cols.

Object movement — equal-probability 4-connected (axis-aligned):
    At each step the object can stay, move up, down, left, or right.
    Only moves that remain within the grid are available.  All available
    destinations receive equal probability:
        p_each = (1 - TERMINAL_PROB) / n_available_dests
    Corner cells have 3 destinations, edge cells 4, interior cells 5.
    TERMINAL_PROB = 0.005 is reserved above the last cumulative threshold.

    This is implemented via per-cell cumulative probability tables built
    at __init__ time (same pattern as iobt_equal_env).  The padded
    obj_trans_matrix is retained for API compatibility but is NOT used
    by object_move().

State encoding (unchanged from original):
    state = cell_idx * (time_limit + 1) + time_component
    cell_idx ∈ [0, NROWS*NCOLS - 1]  (row-major: idx = row*NCOLS + col)
    terminal state = NROWS*NCOLS
    missing_state  = NROWS*NCOLS*(time_limit_max+1) + 1
"""

import numpy as np
import random as rnd
from itertools import combinations

# Probability reserved for the terminal (exit) transition at every cell.
TERMINAL_PROB = 0.005


class grid_env_rect:
    """
    Rectangular M×N grid environment for object tracking.

    Parameters
    ----------
    nrows : int   — number of rows    (M)
    ncols : int   — number of columns (N)
    num_trans, state_trans_cum_prob, max_sensors, max_sensors_null,
    missing_state, time_limit — same as original grid_env
    """

    def __init__(self, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, missing_state, time_limit):
        self.nrows = nrows
        self.ncols = ncols
        self.N     = ncols          # gym_wrapper reads self.N for obs-space size
        self.n_cells = nrows * ncols

        self.num_trans      = num_trans
        self.prob_list_cum  = np.array(state_trans_cum_prob)
        self.missing_state  = missing_state
        self.time_limit     = time_limit
        self.max_sensors    = max_sensors
        self.max_sensors_null = max_sensors_null

        self.obj_trans_matrix = self._build_transition_matrix()
        # Per-cell equal-probability tables for object_move()
        self._cell_dests, self._cell_cum_probs = self._build_equal_prob_tables()
        self.object_pos = rnd.randrange(self.n_cells)

        self.combination_dict      = {}
        self.cum_comb_dict         = {}
        self.combination_dict_null = {}
        self.total_cum_comb_dict   = [0]
        self.cum_comb_dict_null    = [0]

        self.action_space_size      = {}
        self.action_space_size_null = 0

        self.valid_q_indices_dict = self.get_valid_q_indices_dict()

        # Rewards (identical to original)
        self.tracking_miss_rew         = 0
        self.tracking_miss_rew_missing = 0
        self.sensor_rew                = -0.16
        self.tracking_rew              = 1
        self.tracking_rew_missing      = 0

    # -----------------------------------------------------------------------
    # Coordinate helpers
    # -----------------------------------------------------------------------

    def grid_to_val(self, col, row):
        """(col, row) -> flat cell index  (col = x, row = y)."""
        return row * self.ncols + col

    def val_to_grid(self, val):
        """Flat cell index -> (col, row)  (col = x, row = y)."""
        return val % self.ncols, val // self.ncols

    # -----------------------------------------------------------------------
    # Transition matrix
    # -----------------------------------------------------------------------

    def _build_transition_matrix(self):
        """
        Build the padded transition matrix for API compatibility.

        object_move() does NOT use this matrix — it uses the per-cell
        equal-probability tables built by _build_equal_prob_tables().
        This matrix is kept so any code that inspects obj_trans_matrix
        directly continues to work.  Each active-cell row lists the
        4-connected neighbours (stay + cardinal directions) padded to
        num_trans columns with repeats of the cell itself.
        """
        matrix = []
        for cell in range(self.n_cells):
            dests = self._4connected_dests(cell)
            row   = list(dests)
            while len(row) < self.num_trans:
                row.append(cell)          # pad with stay-in-place
            matrix.append(row[:self.num_trans])
        matrix.append([self.n_cells] * self.num_trans)   # terminal row
        return matrix

    def _4connected_dests(self, cell):
        """Return [stay, up, down, left, right] filtered to grid bounds."""
        col, row = cell % self.ncols, cell // self.ncols
        dests = [cell]                                               # stay
        if row > 0:               dests.append((row-1)*self.ncols + col)  # up
        if row < self.nrows - 1:  dests.append((row+1)*self.ncols + col)  # down
        if col > 0:               dests.append(row*self.ncols + (col-1))  # left
        if col < self.ncols - 1:  dests.append(row*self.ncols + (col+1))  # right
        return dests

    def _build_equal_prob_tables(self):
        """
        Build per-cell cumulative probability tables for equal-probability
        4-connected movement with a fixed terminal probability.

        For each cell:
            destinations = [stay, up, down, left, right]  (within-bounds only)
            p_each       = (1 - TERMINAL_PROB) / len(destinations)
            cum_probs    = [p_each, 2*p_each, ..., len(dests)*p_each]

        object_move() draws u ~ Uniform(0,1):
            slot = #{thresholds <= u}
            slot < len(dests)  ->  move to dests[slot]
            slot >= len(dests) ->  terminal  (u > cum_probs[-1])
        """
        cell_dests     = {}
        cell_cum_probs = {}
        for cell in range(self.n_cells):
            dests  = self._4connected_dests(cell)
            n      = len(dests)
            p_each = (1.0 - TERMINAL_PROB) / n
            cum    = np.array([p_each * (i + 1) for i in range(n)])
            cell_dests[cell]     = dests
            cell_cum_probs[cell] = cum
        return cell_dests, cell_cum_probs

    # -----------------------------------------------------------------------
    # Object movement — equal-probability 4-connected
    # -----------------------------------------------------------------------

    def object_move(self):
        """
        Move the object one step using per-cell equal-probability tables.

        Every available destination (stay + valid cardinal directions) is
        equally likely.  A draw above the last cumulative threshold triggers
        the terminal transition.  Replaces the parent's slot-based method
        which used a global prob_list_cum over a randomly-padded matrix.
        """
        pos = self.object_pos

        if pos == self.n_cells:          # already terminal — absorbing
            return 0

        if pos not in self._cell_cum_probs:   # defensive guard
            self.object_pos = self.n_cells
            return 0

        cum   = self._cell_cum_probs[pos]
        dests = self._cell_dests[pos]
        u     = np.random.uniform(0, 1)
        slot  = int(np.sum(cum <= u))

        if slot < len(dests):
            self.object_pos = dests[slot]
        else:
            self.object_pos = self.n_cells   # terminal

        return 0

    def reset_object_state(self):
        self.object_pos = rnd.randrange(self.n_cells)

    # -----------------------------------------------------------------------
    # Sensor validity
    # -----------------------------------------------------------------------

    def check_valid_sensor(self, time_value, b_l, b_r, b_d, b_u):
        """Return a flat list of 1/0 for each cell in the (2t+3)² window."""
        grid_sz = 2 * time_value + 3
        cx, cy  = grid_sz // 2, grid_sz // 2
        valid   = []
        for i in range(grid_sz * grid_sz):
            dx, dy = i % grid_sz - cx, i // grid_sz - cy
            if b_l <= dx <= b_r and b_d <= dy <= b_u:
                valid.append(1)
            else:
                valid.append(0)
        return valid

    def get_valid_q_indices(self, state, time_value):
        """Valid sensor positions for a given Track-MDP state and time delay."""
        state_cell    = state // (self.time_limit + 1)
        col, row      = self.val_to_grid(state_cell)
        b_l = 0   - col;  b_r = (self.ncols - 1) - col
        b_d = 0   - row;  b_u = (self.nrows - 1) - row
        return np.array(self.check_valid_sensor(time_value, b_l, b_r, b_d, b_u))

    def get_valid_q_indices_dict(self):
        valid_dict = {}
        for t in range(self.time_limit + 1):
            valid_dict[t] = {}
            for s in range(self.n_cells * (self.time_limit + 1)):
                valid_dict[t][s] = self.get_valid_q_indices(s, t)
        return valid_dict

    # -----------------------------------------------------------------------
    # Object alignment
    # -----------------------------------------------------------------------

    def realign_obj(self, object_pos, current_state, time_delay):
        """Return (relative_position_in_window, in_window_flag)."""
        state_cell = current_state // (self.time_limit + 1)
        grid_rad   = (2 * time_delay + 3) // 2

        s_col, s_row = self.val_to_grid(state_cell)
        o_col, o_row = self.val_to_grid(object_pos)
        dx, dy       = o_col - s_col, o_row - s_row

        if abs(dx) > grid_rad or abs(dy) > grid_rad:
            in_window = 0
        else:
            in_window = 1

        window_w   = 2 * grid_rad + 1
        centre_idx = grid_rad * window_w + grid_rad
        rel_pos    = centre_idx + dy * window_w + dx
        return rel_pos, in_window

    # -----------------------------------------------------------------------
    # Reward / next state
    # -----------------------------------------------------------------------

    def get_reward_next_state(self, current_state, current_action, time_delay):
        obj_position      = self.object_pos
        obj_found         = 0
        no_of_time_sensors = (2 * time_delay + 3) ** 2

        if current_state != self.missing_state:
            action_clip    = current_action[-no_of_time_sensors:]
            action_sensors = np.multiply(
                action_clip,
                self.valid_q_indices_dict[time_delay][current_state]
            )
            obj_rel_pos, obj_in_grid = self.realign_obj(
                self.object_pos, current_state, time_delay
            )
        else:
            obj_rel_pos, obj_in_grid = 0, 1
            action_sensors = [1] * no_of_time_sensors

        if obj_in_grid == 1 and action_sensors[int(obj_rel_pos)] == 1:
            obj_found        = 1
            next_state       = obj_position * (self.time_limit + 1)
            time_delay_sense = 0
        else:
            time_delay_sense = time_delay + 1
            if current_state != self.missing_state:
                if time_delay_sense > self.time_limit:
                    next_state = self.missing_state
                else:
                    next_state = current_state + 1
            else:
                next_state = self.missing_state

        self.object_move()
        no_sensor_on = np.sum(action_sensors)

        if current_state != self.missing_state:
            reward = (obj_found * self.tracking_rew
                      + (1 - obj_found) * self.tracking_miss_rew
                      + no_sensor_on * self.sensor_rew)
        else:
            reward = (obj_found * self.tracking_rew_missing
                      + (1 - obj_found) * self.tracking_miss_rew_missing
                      + self.n_cells * self.sensor_rew)

        terminal = 1 if self.object_pos == self.n_cells else 0
        return reward, next_state, terminal, time_delay_sense


class learning_grid_sarsa_0:
    """
    Wrapper matching the original interface.

    Parameters
    ----------
    nrows, ncols : grid dimensions  (replaces the single N parameter)
    All other parameters identical to the original learning_grid_sarsa_0.
    """

    def __init__(self, run_number, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max):
        self.run_number     = run_number
        self.nrows          = nrows
        self.ncols          = ncols
        self.N              = ncols         # keep for gym_wrapper compatibility
        self.n_cells        = nrows * ncols
        self.num_trans      = num_trans
        self.prob_list_cum  = state_trans_cum_prob
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = self.n_cells * (self.time_limit_max + 1) + 1

        self.grid_env = grid_env_rect(
            nrows, ncols, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null, self.missing_state, time_limit
        )

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
        self.file_save_directory = "/home/ma10/documents/rl_sensor/qsave"
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()