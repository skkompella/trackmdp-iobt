#!/usr/bin/env python3
"""
finetune_deterministic.py — Fine-tune a pretrained PPO checkpoint on a
deterministic circular path through either a rectangular M×N grid or the
10-node Camp Buckner IoBT graph.

The target object moves deterministically along a fixed cycle of node/cell
indices, looping forever with no terminal probability.  The PPO policy
(loaded from a prior checkpoint) is trained on this simple pattern with
aggressive hyperparameters designed for fast convergence on a single, known
trajectory.

Hyperparameter rationale
------------------------
  lr = 5e-4           : 5× higher than default — forces large updates from fresh data
  train_batch_size=512: tiny vs default 4000 — only fresh circular experience in buffer
  num_sgd_iter=20     : more gradient passes per batch — squeezes every bit of signal
  entropy_coeff=0.001 : near-zero — suppresses exploration, policy collapses onto circle
  clip_param=0.3      : slightly relaxed PPO clip — allows bigger policy steps
  rollout_fragment_length=64: short worker rollouts — batch stays fresh
  max_ep_steps=100    : short episodes — tight feedback loop on the circle
  eval_interval=1     : eval every single training iteration — see convergence live

WARNING: these settings will overfit the policy to the circle.  Reloading this
checkpoint in a stochastic environment will likely degrade general performance.
That is expected — the goal is fast demonstration of Track-MDP re-acquisition
on a fixed path.

Defining the circle
-------------------
Edit RECT_CIRCLE_PATH (grid mode) or IOBT_CIRCLE_PATH (iobt mode) at the top
of this file, or pass --circle as a comma-separated list on the command line.

  Default rect 2×3 perimeter:  [0, 1, 2, 5, 4, 3]  (clockwise)
  Default IoBT 10-node cycle:  [0, 4, 3, 6, 2, 1]  (valid 6-node outer loop)

  IoBT adjacency (Camp Buckner IOBT_MAP):
    node 0 → [1, 7, 4]        node 5 → [4, 3]
    node 1 → [0, 7, 6, 2]     node 6 → [1, 2, 8]
    node 2 → [1, 6, 3]        node 7 → [9, 8, 0, 1]
    node 3 → [2, 5, 6]        node 8 → [9, 6]
    node 4 → [5, 2, 3, 0, 1]  node 9 → [8, 7]

Usage
-----
    # Rectangular grid (default)
    python examples/finetune_deterministic.py --run 200
    python examples/finetune_deterministic.py --run 200 --new-run 201 --iterations 300
    python examples/finetune_deterministic.py --eval-only --checkpoint runs/agent_run201_ppo

    # IoBT 10-node graph
    python examples/finetune_deterministic.py --env iobt --run 195
    python examples/finetune_deterministic.py --env iobt --run 195 --new-run 210 --iterations 500
    python examples/finetune_deterministic.py --env iobt --circle 0,1,2,6,3 --run 195
"""

import argparse
import os
import sys
import numpy as np

# NumPy 2.0 removed np.float_; Ray RLlib's space_utils still checks for it.
# Patch it back as an alias so compute_single_action doesn't crash.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig

# Rectangular grid
from src.core.grid_env_rect     import learning_grid_sarsa_0 as rect_learner_cls, grid_env_rect
from src.core.grid_wrapper_rect import grid_environment_rect

# IoBT graph
from src.core.iobt_new_env      import iobt_env
from src.core.gym_wrapper       import grid_environment


# ===========================================================================
# ─── CIRCLE PATHS ───────────────────────────────────────────────────────────
#
#   RECT_CIRCLE_PATH : cell indices (row-major) for the 2×3 default grid.
#   IOBT_CIRCLE_PATH : node indices for the 10-node Camp Buckner IoBT graph.
#
#   2×3 grid layout:
#     col:   0    1    2
#     row 0: [0]  [1]  [2]
#     row 1: [3]  [4]  [5]
#
#   IoBT path [0, 4, 3, 6, 2, 1] traces the outer 6-node loop:
#     0→4 (in IOBT_MAP[0]=[1,7,4])  4→3 (in IOBT_MAP[4]=[5,2,3,0,1])
#     3→6 (in IOBT_MAP[3]=[2,5,6])  6→2 (in IOBT_MAP[6]=[1,2,8])
#     2→1 (in IOBT_MAP[2]=[1,6,3])  1→0 (in IOBT_MAP[1]=[0,7,6,2])
#
# ===========================================================================

RECT_CIRCLE_PATH: list[int] = [0, 1, 2, 5, 4, 3]
IOBT_CIRCLE_PATH: list[int] = [0, 4, 3, 6, 2, 1]


# ===========================================================================
# ─── IOBT MAP (Camp Buckner 10-node topology) ────────────────────────────────
# ===========================================================================

IOBT_MAP = {
    0: [1, 7, 4],        # node 1
    1: [0, 7, 6, 2],     # node 2
    2: [1, 6, 3],        # node 3
    3: [2, 5, 6],        # node 4
    4: [5, 2, 3, 0, 1],  # node 5
    5: [4, 3],           # node 6
    6: [1, 2, 8],        # node 7
    7: [9, 8, 0, 1],     # node 8
    8: [9, 6],           # node 9
    9: [8, 7],           # node 10
}
IOBT_NUM_NODES = len(IOBT_MAP)   # 10
IOBT_N         = 4               # 4×4 backing grid for the 10-node IoBT map


# ===========================================================================
# Defaults
# ===========================================================================

RECT_DEFAULTS = {
    # Environment
    "env":             "rect",
    "run_number":      200,
    "new_run":         301,
    "nrows":           2,
    "ncols":           3,
    "num_trans":       4,
    "max_sensors":     6,
    "max_sensors_null": 6,
    "time_limit":      1,
    "time_limit_max":  1,
    # Fast-convergence PPO hyperparameters
    "lr":                      5e-4,
    "train_batch_size":        512,
    "num_sgd_iter":            20,
    "sgd_minibatch_size":      128,
    "clip_param":              0.3,
    "entropy_coeff":           0.001,
    "vf_loss_coeff":           1.0,
    "grad_clip":               10.0,
    "num_workers":             2,
    "rollout_fragment_length": 64,
    # Training schedule
    "training_iterations": 200,
    "eval_interval":       1,
    "eval_episodes":       20,
    "max_ep_steps":        100, # was 100
}

IOBT_DEFAULTS = dict(RECT_DEFAULTS)
IOBT_DEFAULTS.update({
    "env":         "iobt",
    "run_number":  195,
    "num_trans":   6,
    "max_sensors": 6,
    "max_sensors_null": 6,
    # nrows/ncols not used for IoBT, but kept for shared code paths
    "nrows": IOBT_N,
    "ncols": IOBT_N,
})

# Slow-convergence hyperparameters matching src/training/trainer.py
ADVANCED_HPARAMS = {
    "lr":                      1e-4,
    "train_batch_size":        4000,
    "num_sgd_iter":            10,
    "sgd_minibatch_size":      128,
    "clip_param":              0.2,
    "entropy_coeff":           0.01,
    "vf_loss_coeff":           1.0,
    "grad_clip":               30.0,
    "num_workers":             4,
    "rollout_fragment_length": 1000,
}


# Augment vecs (must match gym_wrapper)
def _build_augment_vecs(tlm):
    v1 = [(2 * i + 3) ** 2 for i in range(tlm + 1)]
    v  = [0]
    for x in v1:
        v.append(v[-1] + x)
    return v1, v


def _build_augment_vecs_no_moore(tlm, n_cells):
    v1 = [n_cells] * (tlm + 1)
    v  = [0]
    for x in v1:
        v.append(v[-1] + x)
    return v1, v

_AV1, _AV        = _build_augment_vecs(RECT_DEFAULTS["time_limit_max"])
ACTION_HISTORY_LEN = _AV[-1]           # 34
MAX_ACTION_SZ      = (2 * RECT_DEFAULTS["time_limit_max"] + 3) ** 2  # 25


# ===========================================================================
# Rectangular grid — circular environment
# ===========================================================================

class CircularGridEnv(grid_env_rect):
    """
    grid_env_rect with object_move() overridden to step deterministically
    along a circle path.  No terminal transitions — the object loops forever.
    """

    def __init__(self, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, missing_state, time_limit,
                 circle_path):
        super().__init__(nrows, ncols, num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state, time_limit)
        self._circle     = circle_path
        self._circle_len = len(circle_path)
        for cell in circle_path:
            assert 0 <= cell < self.n_cells, \
                f"Circle cell {cell} out of range for {nrows}×{ncols} grid"
        self.reset_object_state()

    def reset_object_state(self):
        import random
        self.object_pos  = random.choice(self._circle)
        self._circle_idx = self._circle.index(self.object_pos)

    def object_move(self):
        self._circle_idx = (self._circle_idx + 1) % self._circle_len
        self.object_pos  = self._circle[self._circle_idx]
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        reward, next_state, _, new_delay = super().get_reward_next_state(
            current_state, current_action, time_delay
        )
        return reward, next_state, 0, new_delay   # terminal always 0


class CircularLearner:
    """Wraps CircularGridEnv to match the qobj interface expected by the wrappers."""

    def __init__(self, run_number, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max,
                 circle_path):
        self.run_number     = run_number
        self.nrows          = nrows
        self.ncols          = ncols
        self.N              = ncols
        self.n_cells        = nrows * ncols
        self.num_trans      = num_trans
        self.prob_list_cum  = state_trans_cum_prob
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = self.n_cells * (time_limit_max + 1) + 1

        self.grid_env = CircularGridEnv(
            nrows, ncols, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            circle_path
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# Rectangular grid — transition-matrix environment
# ===========================================================================

class TransitionMatrixGridEnv(grid_env_rect):
    """
    grid_env_rect with object_move() overridden to sample the next cell from
    an empirical transition matrix T[i, j] = P(move to j | currently at i).
    Terminal is always 0; the object never leaves the grid.
    """

    def __init__(self, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, missing_state, time_limit,
                 transition_matrix):
        super().__init__(nrows, ncols, num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state, time_limit)
        T = np.asarray(transition_matrix, dtype=np.float64)
        assert T.shape == (self.n_cells, self.n_cells), \
            f"Transition matrix must be {self.n_cells}×{self.n_cells}, got {T.shape}"
        self._T = T

    def reset_object_state(self):
        self.object_pos = int(np.random.randint(self.n_cells))

    def object_move(self):
        row = self._T[self.object_pos].copy()
        s = row.sum()
        if s == 0:
            # Cell never observed as source — uniform over all other cells
            row[:] = 1.0
            row[self.object_pos] = 0.0
            s = row.sum()
        row /= s
        self.object_pos = int(np.random.choice(self.n_cells, p=row))
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        reward, next_state, _, new_delay = super().get_reward_next_state(
            current_state, current_action, time_delay
        )
        return reward, next_state, 0, new_delay


class TransitionMatrixLearner:
    """Wraps TransitionMatrixGridEnv to match the qobj interface."""

    def __init__(self, run_number, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max,
                 transition_matrix):
        self.run_number     = run_number
        self.nrows          = nrows
        self.ncols          = ncols
        self.N              = ncols
        self.n_cells        = nrows * ncols
        self.num_trans      = num_trans
        self.prob_list_cum  = state_trans_cum_prob
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = self.n_cells * (time_limit_max + 1) + 1

        self.grid_env = TransitionMatrixGridEnv(
            nrows, ncols, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            transition_matrix,
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# Real-data replay — 2×3 rect grid driven by a precomputed object sequence
# ===========================================================================

class RealDataGridEnv(grid_env_rect):
    """
    grid_env_rect whose object steps deterministically through a precomputed
    (T,) sequence of cell indices (0–5, corresponding to nodes 11–16).
    Sequence is derived from GPS ground truth (training) or RSSI posterior (eval).
    No terminal transitions — the sequence loops.
    """

    def __init__(self, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, missing_state, time_limit,
                 object_sequence: np.ndarray, no_moore_constraint=False,
                 binary_detections=None):
        # Must be assigned BEFORE super().__init__() — it calls reset_object_state()
        self._object_sequence = np.asarray(object_sequence, dtype=np.int32)
        self._seq_t = 0
        self._binary_detections = (np.asarray(binary_detections, dtype=bool)
                                   if binary_detections is not None else None)
        super().__init__(nrows, ncols, num_trans, state_trans_cum_prob,
                         max_sensors, max_sensors_null, missing_state, time_limit,
                         no_moore_constraint=no_moore_constraint)

    def reset_object_state(self):
        self._seq_t = int(np.random.randint(len(self._object_sequence)))
        self.object_pos = int(self._object_sequence[self._seq_t])

    def object_move(self):
        self._seq_t = (self._seq_t + 1) % len(self._object_sequence)
        self.object_pos = int(self._object_sequence[self._seq_t])
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        obj_position       = self.object_pos
        obj_found          = 0
        no_of_time_sensors = (2 * time_delay + 3) ** 2

        if current_state != self.missing_state:
            if self.no_moore_constraint:
                action_sensors = np.asarray(current_action[:self.n_cells])
                obj_rel_pos, obj_in_grid = self.object_pos, 1
            else:
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
            action_sensors = [1] * (self.n_cells if self.no_moore_constraint
                                    else no_of_time_sensors)

        node_activated = action_sensors[int(obj_rel_pos)] == 1
        # Binary gating only in tracked state: missing-state is a full-grid rescan
        # and must always succeed (otherwise time_delay can overflow av1's bounds).
        clf_says_yes   = (self._binary_detections is None or
                          current_state == self.missing_state or
                          bool(self._binary_detections[self._seq_t, int(obj_rel_pos)]))

        if obj_in_grid == 1 and node_activated and clf_says_yes:
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

        # terminal forced to 0 — sequence loops indefinitely
        return reward, next_state, 0, time_delay_sense


class RealDataLearner:
    """Wraps RealDataGridEnv to match the qobj interface (forces 2×3 grid)."""

    def __init__(self, run_number, nrows, ncols, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max,
                 object_sequence: np.ndarray, no_moore_constraint=False,
                 binary_detections=None):
        self.run_number     = run_number
        self.nrows          = nrows
        self.ncols          = ncols
        self.N              = ncols
        self.n_cells        = nrows * ncols
        self.num_trans      = num_trans
        self.prob_list_cum  = state_trans_cum_prob
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = self.n_cells * (time_limit_max + 1) + 1

        self.grid_env = RealDataGridEnv(
            nrows, ncols, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            object_sequence, no_moore_constraint=no_moore_constraint,
            binary_detections=binary_detections,
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# IoBT 10-node graph — circular environment
# ===========================================================================

class CircularIoBTEnv(iobt_env):
    """
    iobt_env with object_move() overridden to step deterministically along
    IOBT_CIRCLE_PATH (or any valid cycle of IoBT node indices).

    No terminal transitions — the object loops forever; terminal_flag is
    always returned as 0 from get_reward_next_state().

    NOTE: _circle must be set before calling super().__init__() because
    iobt_env.__init__() calls self.reset_object_state() at the end, which
    is overridden here to use self._circle.
    """

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 circle_path, no_moore_constraint=False):
        # Set _circle BEFORE super().__init__() — iobt_env.__init__ ends with
        # self.reset_object_state(), which our override requires.
        for node in circle_path:
            assert 0 <= node < IOBT_NUM_NODES, \
                f"Circle node {node} out of range (0–{IOBT_NUM_NODES-1})"
        self._circle     = circle_path
        self._circle_len = len(circle_path)
        self._no_moore   = no_moore_constraint
        super().__init__(max_sensors, max_sensors_null, missing_state, time_limit)

    def reset_object_state(self):
        import random
        self.object_pos  = random.choice(self._circle)
        self._circle_idx = self._circle.index(self.object_pos)

    def object_move(self):
        """Advance one step along the IoBT circle — deterministic, no terminal."""
        self._circle_idx = (self._circle_idx + 1) % self._circle_len
        self.object_pos  = self._circle[self._circle_idx]
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        """Same as parent but with terminal_flag forced to 0.

        When no_moore_constraint=True the gym wrapper sends a flat n_grid-element
        action.  The base grid_env clips to a (2t+3)² Moore window, causing a
        shape mismatch at time_delay=1 (16 vs 25 elements).  We bypass that
        path entirely and use a direct IoBT-node lookup instead.
        """
        if not self._no_moore:
            reward, next_state, _, new_delay = super().get_reward_next_state(
                current_state, current_action, time_delay
            )
            return reward, next_state, 0, new_delay

        # ── no_moore path: direct node-index detection ─────────────────────
        n_grid         = self.N * self.N          # 16 — backing grid size
        obj_position   = self.object_pos
        action_sensors = np.asarray(current_action[:n_grid])

        # In missing state, treat all nodes as scanned (mirrors base env behaviour)
        if current_state == self.missing_state:
            action_sensors = np.ones(n_grid, dtype=int)

        node_activated = int(action_sensors[obj_position]) == 1

        if node_activated:
            next_state       = obj_position * (self.time_limit + 1)
            time_delay_sense = 0
        else:
            time_delay_sense = time_delay + 1
            if current_state != self.missing_state:
                next_state = (self.missing_state
                              if time_delay_sense > self.time_limit
                              else current_state + 1)
            else:
                next_state = self.missing_state

        self.object_move()
        no_sensor_on = int(action_sensors.sum())

        if current_state != self.missing_state:
            reward = (float(node_activated)       * self.tracking_rew
                      + float(not node_activated) * self.tracking_miss_rew
                      + no_sensor_on              * self.sensor_rew)
        else:
            reward = (float(node_activated)       * self.tracking_rew_missing
                      + float(not node_activated) * self.tracking_miss_rew_missing
                      + n_grid                    * self.sensor_rew)

        return reward, next_state, 0, time_delay_sense


class CircularIoBTLearner:
    """Wraps CircularIoBTEnv to match the qobj interface expected by the wrappers."""

    def __init__(self, run_number, num_trans, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, circle_path,
                 no_moore_constraint=False):
        self.run_number     = run_number
        # N must be the backing grid size so gym_wrapper obs-space is correct.
        self.N              = IOBT_N
        self.n_cells        = IOBT_NUM_NODES
        self.num_trans      = num_trans
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        # missing_state follows the N×N backing grid formula (matching iobt_new_env.py)
        self.missing_state  = IOBT_N * IOBT_N * (time_limit_max + 1) + 1

        self.grid_env = CircularIoBTEnv(
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            circle_path, no_moore_constraint=no_moore_constraint
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# IoBT 10-node graph — topology-driven random-walk environment
# ===========================================================================

class TopoIoBTEnv(iobt_env):
    """
    iobt_env with object_move() overridden to perform a random walk on the
    IOBT_MAP topology.  At every step the object stays in its current node
    OR moves to one of its immediate neighbours, each with equal probability
    1 / (1 + len(neighbours)).

    Fully compatible with no_moore_constraint=True (same obs/action space as
    RealIoBTEnv: obs=83, action=16).  get_reward_next_state() reuses the same
    no_moore path as CircularIoBTEnv so the two environments are drop-in
    interchangeable for training.
    """

    # Build the self-inclusive neighbour lists once at class level.
    _TOPO: dict[int, list[int]] = {
        node: [node] + neighbours
        for node, neighbours in IOBT_MAP.items()
    }

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 no_moore_constraint=False):
        self._no_moore = no_moore_constraint
        super().__init__(max_sensors, max_sensors_null, missing_state, time_limit)

    def reset_object_state(self):
        import random
        self.object_pos = random.randrange(IOBT_NUM_NODES)

    def object_move(self):
        """Step to a uniformly-random neighbour (or stay) on the IOBT graph."""
        import random
        self.object_pos = random.choice(self._TOPO[self.object_pos])
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        """Identical no_moore path as CircularIoBTEnv.get_reward_next_state."""
        if not self._no_moore:
            reward, next_state, _, new_delay = super().get_reward_next_state(
                current_state, current_action, time_delay
            )
            return reward, next_state, 0, new_delay

        n_grid         = self.N * self.N
        obj_position   = self.object_pos
        action_sensors = np.asarray(current_action[:n_grid])

        if current_state == self.missing_state:
            action_sensors = np.ones(n_grid, dtype=int)

        node_activated = int(action_sensors[obj_position]) == 1

        if node_activated:
            next_state       = obj_position * (self.time_limit + 1)
            time_delay_sense = 0
        else:
            time_delay_sense = time_delay + 1
            if current_state != self.missing_state:
                next_state = (self.missing_state
                              if time_delay_sense > self.time_limit
                              else current_state + 1)
            else:
                next_state = self.missing_state

        self.object_move()
        no_sensor_on = int(action_sensors.sum())

        if current_state != self.missing_state:
            reward = (float(node_activated)       * self.tracking_rew
                      + float(not node_activated) * self.tracking_miss_rew
                      + no_sensor_on              * self.sensor_rew)
        else:
            reward = (float(node_activated)       * self.tracking_rew_missing
                      + float(not node_activated) * self.tracking_miss_rew_missing
                      + n_grid                    * self.sensor_rew)

        return reward, next_state, 0, time_delay_sense


class TopoIoBTLearner:
    """Wraps TopoIoBTEnv to match the qobj interface expected by the wrappers."""

    def __init__(self, run_number, num_trans, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, no_moore_constraint=True):
        self.run_number     = run_number
        self.N              = IOBT_N
        self.n_cells        = IOBT_NUM_NODES
        self.num_trans      = num_trans
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = IOBT_N * IOBT_N * (time_limit_max + 1) + 1

        self.grid_env = TopoIoBTEnv(
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            no_moore_constraint=no_moore_constraint,
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# IoBT 10-node graph — GPS-driven real-data environment with soft reward
# ===========================================================================

class RealIoBTEnv(iobt_env):
    """
    iobt_env whose object replays a GPS-derived (T_gps,) node-index sequence
    (0-based).  Reward is soft (probability-weighted); state transitions use a
    binary threshold.

    p_audio: (T_gps, 10) float32 — calibrated classifier probability at each
             GPS-covered step for each of the 10 nodes.
             If None, falls back to hard binary (node activated = detected).

    NOTE: all pre-super() attributes must be set before super().__init__()
    because iobt_env.__init__() calls reset_object_state() at the end.
    """

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 object_sequence: np.ndarray,
                 p_audio=None,
                 soft_threshold: float = 0.5,
                 soft_scale: float = 1.0,
                 sensor_rew=None):
        self._object_sequence = np.asarray(object_sequence, dtype=np.int32)
        self._seq_t           = 0
        self._p_audio         = (np.asarray(p_audio, dtype=np.float32)
                                  if p_audio is not None else None)
        self._soft_threshold  = soft_threshold
        self._soft_scale      = soft_scale
        super().__init__(max_sensors, max_sensors_null, missing_state, time_limit)
        # Override with IoBT-tuned reward params (iobt_environment.py values)
        self.tracking_rew              = 1.5
        self.tracking_miss_rew         = -0.5
        self.tracking_rew_missing      = 0.5
        self.tracking_miss_rew_missing = -1.0
        self.sensor_rew                = (-0.25 if sensor_rew is None
                                          else float(sensor_rew))

    def reset_object_state(self):
        self._seq_t     = int(np.random.randint(len(self._object_sequence)))
        self.object_pos = int(self._object_sequence[self._seq_t])

    def object_move(self):
        self._seq_t     = (self._seq_t + 1) % len(self._object_sequence)
        self.object_pos = int(self._object_sequence[self._seq_t])
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        n_grid = self.N * self.N   # backing grid size (16); matches grid_env convention
        obj_position   = self.object_pos
        action_sensors = np.asarray(current_action[:n_grid])
        obj_rel_pos    = obj_position   # no_moore_constraint → direct mapping
        obj_in_grid    = 1

        # When in missing state, mirror base-env behaviour: treat ALL sensors as
        # active for the re-acquisition scan (clipping from evaluate_policy must
        # not prevent the agent from finding the object in the unknown state).
        if current_state == self.missing_state:
            action_sensors = np.ones(n_grid, dtype=int)

        # Agent action mask: P_audio contributes ONLY when agent activated this node.
        node_activated = int(action_sensors[obj_rel_pos]) == 1

        # ── Soft detection ─────────────────────────────────────────────────────
        if (self._p_audio is not None and
                current_state != self.missing_state and
                self._seq_t < len(self._p_audio)):
            soft_det = (float(node_activated)
                        * float(self._p_audio[self._seq_t, obj_rel_pos])
                        * self._soft_scale)
            obj_found_binary = soft_det >= self._soft_threshold
        else:
            soft_det         = float(node_activated)
            obj_found_binary = bool(node_activated)

        # ── State transition (binary) ──────────────────────────────────────────
        if obj_in_grid and obj_found_binary:
            next_state       = obj_position * (self.time_limit + 1)
            time_delay_sense = 0
        else:
            time_delay_sense = time_delay + 1
            if current_state != self.missing_state:
                next_state = (self.missing_state
                              if time_delay_sense > self.time_limit
                              else current_state + 1)
            else:
                next_state = self.missing_state

        self.object_move()
        no_sensor_on = int(action_sensors.sum())

        # ── Reward ─────────────────────────────────────────────────────────────
        # Tracked state: soft interpolation between tracking_rew / tracking_miss_rew.
        # Missing state: binary found/missed (full-grid rescan must always succeed).
        if current_state != self.missing_state:
            reward = (soft_det * self.tracking_rew
                      + (1.0 - soft_det) * self.tracking_miss_rew
                      + no_sensor_on * self.sensor_rew)
        else:
            reward = (float(obj_found_binary) * self.tracking_rew_missing
                      + float(not obj_found_binary) * self.tracking_miss_rew_missing
                      + n_grid * self.sensor_rew)

        # terminal forced to 0 — sequence loops indefinitely
        return reward, next_state, 0, time_delay_sense


class RealIoBTLearner:
    """Wraps RealIoBTEnv to match the qobj interface expected by gym_wrapper."""

    def __init__(self, run_number, max_sensors, max_sensors_null,
                 time_limit, time_limit_max,
                 object_sequence: np.ndarray,
                 p_audio=None,
                 soft_threshold: float = 0.5,
                 soft_scale: float = 1.0,
                 sensor_rew=None):
        self.run_number     = run_number
        self.N              = IOBT_N
        self.n_cells        = IOBT_NUM_NODES
        self.num_trans      = 6
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = IOBT_N * IOBT_N * (time_limit_max + 1) + 1

        self.grid_env = RealIoBTEnv(
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            object_sequence,
            p_audio        = p_audio,
            soft_threshold = soft_threshold,
            soft_scale     = soft_scale,
            sensor_rew     = sensor_rew,
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# IoBT 10-node graph — transition-matrix environment
# ===========================================================================

class TransitionMatrixIoBTEnv(iobt_env):
    """
    iobt_env with object_move() overridden to sample from an empirical
    transition matrix.  Terminal is always 0.

    NOTE: _T must be set BEFORE super().__init__() because iobt_env.__init__
    calls reset_object_state() at the end.
    """

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 transition_matrix):
        T = np.asarray(transition_matrix, dtype=np.float64)
        assert T.shape == (IOBT_NUM_NODES, IOBT_NUM_NODES), \
            f"IoBT transition matrix must be {IOBT_NUM_NODES}×{IOBT_NUM_NODES}, got {T.shape}"
        self._T = T
        super().__init__(max_sensors, max_sensors_null, missing_state, time_limit)

    def reset_object_state(self):
        self.object_pos = int(np.random.randint(IOBT_NUM_NODES))

    def object_move(self):
        row = self._T[self.object_pos].copy()
        s = row.sum()
        if s == 0:
            row[:] = 1.0
            row[self.object_pos] = 0.0
            s = row.sum()
        row /= s
        self.object_pos = int(np.random.choice(IOBT_NUM_NODES, p=row))
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        reward, next_state, _, new_delay = super().get_reward_next_state(
            current_state, current_action, time_delay
        )
        return reward, next_state, 0, new_delay


class TransitionMatrixIoBTLearner:
    """Wraps TransitionMatrixIoBTEnv to match the qobj interface."""

    def __init__(self, run_number, num_trans, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, transition_matrix):
        self.run_number     = run_number
        self.N              = IOBT_N
        self.n_cells        = IOBT_NUM_NODES
        self.num_trans      = num_trans
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max
        self.missing_state  = IOBT_N * IOBT_N * (time_limit_max + 1) + 1

        self.grid_env = TransitionMatrixIoBTEnv(
            max_sensors, max_sensors_null, self.missing_state, time_limit,
            transition_matrix,
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
        self.file_save_directory = ""
        self.save_directory      = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.grid_env.time_limit = new_time_limit
        self.grid_env.valid_q_indices_dict = self.grid_env.get_valid_q_indices_dict()


# ===========================================================================
# Transition matrix helpers
# ===========================================================================

def make_random_transition_matrix(n_cells):
    """
    Generate a random row-stochastic n_cells×n_cells transition matrix with
    no self-loops: T[i, i] = 0, each row sums to 1.
    Off-diagonal entries drawn from Dirichlet(1, ..., 1).
    """
    T = np.zeros((n_cells, n_cells), dtype=np.float64)
    for i in range(n_cells):
        off = np.random.dirichlet(np.ones(n_cells - 1))
        cols = [j for j in range(n_cells) if j != i]
        T[i, cols] = off
    return T


# Physical node IDs for the 6-node IOBT experiment (nodes 11-16)
# Column order in intensity.npy matches this list.
INTENSITY_NODE_IDS = [11, 12, 13, 14, 15, 16]

# Theoretical loop for the 20260416 experiment (physical node IDs):
# 15→11→16→14→13→12→11→15
# With cell mapping node_id → col_index (node 11=col0, 12=col1, ..., 16=col5):
#   15→11 = (4,0), 11→16 = (0,5), 16→14 = (5,3), 14→13 = (3,2),
#   13→12 = (2,1), 12→11 = (1,0), 11→15 = (0,4)  ← wrap
INTENSITY_THEORETICAL_TRANSITIONS = [
    (4, 0), (0, 5), (5, 3), (3, 2), (2, 1), (1, 0), (0, 4),
]


def intensity_to_transition(intensity_path: str, alpha: float = 0.85,
                             smooth_window: int = 10) -> np.ndarray:
    """
    Convert a (T, 6) intensity.npy into a 6×6 row-stochastic transition matrix.

    Column order in intensity.npy: [node11, node12, node13, node14, node15, node16]
    → cell indices [0, 1, 2, 3, 4, 5].

    Steps
    -----
    1. Z-score normalise each column (per-node baseline removal).
    2. Apply a running-average smoother (uniform window of smooth_window steps).
    3. Take argmax per timestep → dominant cell sequence.
    4. Build empirical 6×6 count matrix from consecutive-pair transitions.
    5. Blend: T = alpha * T_theory + (1-alpha) * T_empirical
       where T_theory encodes the known loop 15→11→16→14→13→12→11→15.
    6. Row-normalise so each row sums to 1.

    Returns float64 (6, 6) transition matrix.
    """
    from scipy.ndimage import uniform_filter1d

    intensity = np.load(intensity_path).astype(np.float64)
    T_steps, n_nodes = intensity.shape
    assert n_nodes == 6, f"Expected 6-node intensity, got shape {intensity.shape}"

    # ── 1. Z-score normalise per column ────────────────────────────────────
    mu  = intensity.mean(axis=0)
    sig = intensity.std(axis=0) + 1e-9
    norm = (intensity - mu) / sig

    # ── 2. Smooth ───────────────────────────────────────────────────────────
    smoothed = uniform_filter1d(norm, size=smooth_window, axis=0)

    # ── 3. Dominant cell per timestep ───────────────────────────────────────
    dom = np.argmax(smoothed, axis=1)   # (T,) values in 0–5

    # ── 4. Empirical count matrix ───────────────────────────────────────────
    counts = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for t in range(len(dom) - 1):
        counts[dom[t], dom[t + 1]] += 1

    # Normalise rows (uniform fallback for unobserved rows)
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).ravel()
    counts[zero_rows] = 1.0                # uniform fallback
    T_empirical = counts / counts.sum(axis=1, keepdims=True)

    # ── 5. Theoretical matrix ───────────────────────────────────────────────
    T_theory = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for (src, dst) in INTENSITY_THEORETICAL_TRANSITIONS:
        T_theory[src, dst] += 1.0
    # Normalise rows
    row_sums = T_theory.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).ravel()
    T_theory[zero_rows] = 1.0 / n_nodes
    T_theory /= T_theory.sum(axis=1, keepdims=True)

    # ── 6. Blend and normalise ──────────────────────────────────────────────
    T_blend = alpha * T_theory + (1.0 - alpha) * T_empirical
    T_blend /= T_blend.sum(axis=1, keepdims=True)

    # Report dominant-sequence summary
    from itertools import groupby
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(dom.tolist())]
    phys   = {i: n for i, n in enumerate(INTENSITY_NODE_IDS)}
    print("[intensity] Z-score+smoothed dominant node groups "
          f"(smooth_window={smooth_window}):")
    print("  " + "  ".join(f"node{phys[k]}×{c}" for k, c in groups))
    print(f"[intensity] Blending: {alpha:.0%} theory + {1-alpha:.0%} empirical")

    return T_blend


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _is_checkpoint(path):
    return (os.path.isfile(os.path.join(path, "rllib_checkpoint.json")) or
            os.path.isfile(os.path.join(path, "algorithm_state.pkl")))


def find_latest_checkpoint(run_number, save_dir=None):
    candidates = []
    if save_dir:
        candidates.append(save_dir)
    else:
        runs = os.path.join(project_root, "runs")
        candidates += [
            os.path.join(runs, f"agent_run{run_number}_ppo"),
            f"./agent_run{run_number}_ppo",
        ]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        if _is_checkpoint(d):
            return d, d
        ckpts = []
        for item in os.listdir(d):
            full = os.path.join(d, item)
            if os.path.isdir(full) and item.startswith("checkpoint_"):
                try:
                    ckpts.append((int(item.split("_")[1]), full))
                except (ValueError, IndexError):
                    pass
        if ckpts:
            ckpts.sort(key=lambda x: x[0])
            return ckpts[-1][1], d
    return None, candidates[0] if candidates else f"./agent_run{run_number}_ppo"


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_policy(algo, env, cfg):
    """
    Greedy rollout on the circular env.  Works for both rect and IoBT modes.

    cfg must contain:
        n_cells       — number used as state_pos in the missing-state obs
                        (n_cells = nrows*ncols for rect; N*N for IoBT)
        missing_state — the sentinel value (n_cells*(T+1)+1 or N*N*(T+1)+1)
        time_limit, time_limit_max, max_sensors, eval_episodes, max_ep_steps
    """
    n_cells        = cfg["n_cells"]
    missing_state  = cfg["missing_state"]
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    max_sensors    = cfg["max_sensors"]
    no_moore       = cfg.get("no_moore_constraint", False)

    if no_moore:
        av1, av           = _build_augment_vecs_no_moore(time_limit_max, n_cells)
        action_history_len = av[-1]
        max_action_sz      = n_cells
    else:
        av1, av           = _build_augment_vecs(time_limit_max)
        action_history_len = av[-1]
        max_action_sz      = (2 * time_limit_max + 3) ** 2

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(cfg["eval_episodes"]):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < cfg["max_ep_steps"]:
            num_sensors = n_cells if no_moore else (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            action_vec = [1] * action_history_len
            if actions_list:
                action_vec[:av[time_delay]] = actions_list

            if not was_missing:
                state_pos  = current_state // (time_limit + 1)
                state_time = current_state %  (time_limit + 1)
            else:
                state_pos, state_time = n_cells, 0

            obs         = (state_pos, state_time, np.array(action_vec, dtype=np.int64))
            action_full = algo.compute_single_action(obs, explore=False)

            if not was_missing:
                if no_moore:
                    action_sensors = np.array(action_full[:n_cells], dtype=int)
                    obj_rel_pos, obj_in_window = env.object_pos, 1
                else:
                    action_clip    = np.array(action_full[-num_sensors:], dtype=int)
                    action_sensors = np.multiply(
                        action_clip,
                        env.valid_q_indices_dict[time_delay][current_state]
                    )
                    obj_rel_pos, obj_in_window = env.realign_obj(
                        env.object_pos, current_state, time_delay
                    )
            else:
                action_sensors = np.ones(num_sensors, dtype=int)
                obj_rel_pos, obj_in_window = 0, 1

            if action_sensors.sum() > max_sensors:
                active = np.where(action_sensors == 1)[0][:max_sensors]
                action_sensors = np.zeros(num_sensors, dtype=int)
                action_sensors[active] = 1

            obj_detected = int(obj_in_window == 1 and
                               action_sensors[int(obj_rel_pos)] == 1)
            if not was_missing and obj_detected:
                total_found += 1

            reward, next_state, terminal_flag, time_delay = \
                env.get_reward_next_state(current_state, action_sensors, time_delay)

            ep_reward  += reward
            ep_steps   += 1
            ep_sensors += int(action_sensors.sum())
            total_steps += 1

            if time_delay == 0:
                actions_list = []
            else:
                td_idx  = min(int(time_delay) - 1, len(av1) - 1)
                td_ac       = av1[td_idx]
                action_bool = [1] * max_action_sz if was_missing else list(action_full)
                actions_list = actions_list + list(np.array(action_bool)[-td_ac:])

            if terminal_flag:
                break
            current_state = next_state

        total_rewards.append(ep_reward)
        ep_lengths.append(ep_steps)
        sensors_ep.append(ep_sensors / max(ep_steps, 1))

    return {
        "tracking_accuracy": total_found / max(total_steps, 1),
        "mean_reward":       float(np.mean(total_rewards)),
        "std_reward":        float(np.std(total_rewards)),
        "mean_length":       float(np.mean(ep_lengths)),
        "mean_sensors_used": float(np.mean(sensors_ep)),
    }


# ===========================================================================
# Real-data sequence builders + real-data evaluator
# ===========================================================================

def build_gps_sequence(nodes_txt, gps_csv) -> np.ndarray:
    """
    Build a (M,) int32 sequence of cell indices 0–5 from GPS ground truth.
    For each GPS sample the nearest of the 6 sensor nodes is chosen.
    Used ONLY for the training environment — GPS is never touched at eval time.
    """
    from pathlib import Path as _Path
    from collection.flac_to_trackmdp import (
        load_gps_track, load_node_positions_ordered,
    )

    node_ids = INTENSITY_NODE_IDS          # [11, 12, 13, 14, 15, 16]
    node_xy  = load_node_positions_ordered(_Path(nodes_txt), node_ids)
    gps_df   = load_gps_track(_Path(gps_csv))

    sequence = np.zeros(len(gps_df), dtype=np.int32)
    for idx, row in gps_df.iterrows():
        dx = row['x'] - node_xy[:, 0]
        dy = row['y'] - node_xy[:, 1]
        sequence[idx] = int(np.argmin(dx ** 2 + dy ** 2))

    cells_seen = sorted(set(sequence.tolist()))
    print(f"[realdata] GPS  sequence: {len(sequence)} steps, "
          f"cells {cells_seen}  "
          f"(nodes {[INTENSITY_NODE_IDS[c] for c in cells_seen]})")
    return sequence


def build_rssi_sequence(nodes_txt, flac_dir, path_loss_json,
                        gps_csv=None, session="20260416_154037",
                        step_s=0.5) -> np.ndarray:
    """
    Build a (T,) int32 sequence of cell indices 0–5 from the Bayesian RSSI
    posterior.  No GPS is used.  Auto-trains path-loss params if JSON absent.
    Used for evaluation only.
    """
    from pathlib import Path as _Path
    from collection.flac_to_trackmdp import (
        discover_flac_files, compute_power_per_step,
        load_node_positions_ordered, load_path_loss,
        build_hypothesis_grid, compute_posterior_top1,
    )

    nodes_txt_p      = _Path(nodes_txt)
    flac_dir_p       = _Path(flac_dir)
    path_loss_json_p = _Path(path_loss_json)

    # Auto-train path-loss params if the JSON is missing
    if not path_loss_json_p.exists():
        if gps_csv is None:
            raise FileNotFoundError(
                f"{path_loss_json} not found and no --gps-csv provided for auto-training."
            )
        print(f"[realdata] {path_loss_json} not found — running path-loss training …")
        from collection.flac_to_trackmdp import run_train
        run_train(flac_dir_p, nodes_txt_p, _Path(gps_csv),
                  step_s, session, path_loss_json_p)

    # Discover FLAC files for this session
    file_map = discover_flac_files(flac_dir_p, session=session)
    if not file_map:
        raise FileNotFoundError(
            f"No FLAC files for session '{session}' in {flac_dir}"
        )
    node_ids = sorted(file_map.keys())

    # Load node positions + path-loss params
    node_xy             = load_node_positions_ordered(nodes_txt_p, node_ids)
    P0, ETA, SIG2, MASK = load_path_loss(path_loss_json_p, node_ids)

    # Build Bayesian hypothesis grid
    hypos, dist, nearest_idx, _, _ = build_hypothesis_grid(node_xy)

    # Compute audio power per step for every node
    power_arrays: dict = {}
    for nid, fpath in file_map.items():
        power_arrays[nid] = compute_power_per_step(fpath, step_s)
    T = min(len(a) for a in power_arrays.values())

    # Run posterior at each step → cell index
    sequence  = np.zeros(T, dtype=np.int32)
    prev_cell = 0
    for t in range(T):
        rss_map = {
            nid: 10.0 * float(np.log10(max(float(power_arrays[nid][t]), 1e-12)))
            for nid in node_ids
        }
        top1 = compute_posterior_top1(
            rss_map, node_ids, dist, nearest_idx, P0, ETA, SIG2, MASK, hypos
        )
        if top1 is not None and top1 in INTENSITY_NODE_IDS:
            prev_cell = INTENSITY_NODE_IDS.index(top1)
        sequence[t] = prev_cell

    cells_seen = sorted(set(sequence.tolist()))
    print(f"[realdata] RSSI sequence: {T} steps, "
          f"cells {cells_seen}  "
          f"(nodes {[INTENSITY_NODE_IDS[c] for c in cells_seen]})")
    return sequence


def build_rf_sequence(clf_pkl, flac_dir, nodes_txt, session,
                      rssi_window=5, step_s=0.5) -> np.ndarray:
    """
    Build a (T,) int32 sequence of cell indices 0–5 using the trained
    sklearn classifier (RF/SVM/MLP) saved by supervised_transition.py.
    No GPS is used — the RF model predicts node from RSSI features alone.
    """
    import joblib
    from pathlib import Path as _Path
    from collection.supervised_transition import (
        build_rssi_matrix, apply_sklearn_classifier,
    )
    from collection.flac_to_trackmdp import (
        discover_flac_files, compute_power_per_step,
    )

    node_ids = INTENSITY_NODE_IDS          # [11, 12, 13, 14, 15, 16]
    file_map = discover_flac_files(_Path(flac_dir), session=session)
    if not file_map:
        raise FileNotFoundError(
            f"No FLAC files for session '{session}' in {flac_dir}"
        )

    power_arrays = {}
    for nid, fpath in file_map.items():
        power_arrays[nid] = compute_power_per_step(fpath, step_s)
    T = min(len(a) for a in power_arrays.values())

    X = build_rssi_matrix(power_arrays, node_ids, T, rssi_window=rssi_window)

    clf = joblib.load(clf_pkl)
    rankings = apply_sklearn_classifier(X, clf, node_ids)  # (T,) node IDs 11–16

    id_to_cell = {nid: i for i, nid in enumerate(node_ids)}
    sequence = np.array([id_to_cell.get(int(r), 0) for r in rankings], dtype=np.int32)

    cells_seen = sorted(set(sequence.tolist()))
    print(f"[clf-pkl] RF sequence : {T} steps, "
          f"cells {cells_seen}  "
          f"(nodes {[node_ids[c] for c in cells_seen]})")
    return sequence


def build_binary_detection_matrix(node_clfs_dir, flac_dir, session,
                                  rssi_window=5, step_s=0.5) -> np.ndarray:
    """
    Returns (T, 6) bool array. B[t, k] = 1 if node k's classifier
    predicts the person is at node k at timestep t.

    Supports two classifier versions, auto-detected by directory contents:
      v1 — 2 features (rolling mean+std of dB power); no scaler pkl
      v3 — 20 features (amplitude+spectral); paired node_scaler_N*.pkl required
    """
    import joblib
    import glob
    import pandas as _pd
    from pathlib import Path as _Path
    from collection.flac_to_trackmdp import (
        discover_flac_files, compute_power_per_step,
    )

    node_ids = INTENSITY_NODE_IDS   # [11, 12, 13, 14, 15, 16]
    file_map = discover_flac_files(_Path(flac_dir), session=session)
    if not file_map:
        raise FileNotFoundError(
            f"No FLAC files for session '{session}' in {flac_dir}"
        )

    T = min(
        len(compute_power_per_step(fp, step_s))
        for fp in file_map.values()
    )

    # Auto-detect v3 by presence of a scaler pkl for the first node
    first_nid = node_ids[0]
    scaler_pattern = os.path.join(node_clfs_dir, f"node_scaler_N{first_nid}_*.pkl")
    is_v3 = bool(sorted(glob.glob(scaler_pattern)))

    if is_v3:
        from collection.train_node_classifiers_pcen import (
            build_node_features as _build_v3,
            aggregate_rolling,
            FEATURE_NAMES,
            WINDOW_STEPS,
        )
    else:
        from collection.train_node_classifiers import build_node_features as _build_v1

    B = np.zeros((T, len(node_ids)), dtype=bool)
    for col_idx, nid in enumerate(node_ids):
        clf_matches = sorted(glob.glob(
            os.path.join(node_clfs_dir, f"node_clf_N{nid}_*.pkl")
        ))
        if not clf_matches:
            raise FileNotFoundError(
                f"No node classifier found for node {nid}: {node_clfs_dir}"
            )
        clf = joblib.load(clf_matches[-1])

        if is_v3:
            scaler_matches = sorted(glob.glob(
                os.path.join(node_clfs_dir, f"node_scaler_N{nid}_*.pkl")
            ))
            if not scaler_matches:
                raise FileNotFoundError(
                    f"No scaler pkl for node {nid}: {node_clfs_dir}"
                )
            scaler = joblib.load(scaler_matches[-1])
            raw, _ = _build_v3(file_map[nid], step_s)
            X = aggregate_rolling(scaler.transform(raw[:T]), WINDOW_STEPS)
            B[:, col_idx] = clf.predict(
                _pd.DataFrame(X, columns=FEATURE_NAMES)
            ).astype(bool)
        else:
            power = compute_power_per_step(file_map[nid], step_s)
            X = _build_v1(power[:T], rssi_window)   # (T, 2)
            B[:, col_idx] = clf.predict(X).astype(bool)

        print(f"[node-clfs] Node {nid}: {os.path.basename(clf_matches[-1])} "
              f"({'v3' if is_v3 else 'v1'}), "
              f"positive rate = {B[:, col_idx].mean():.3f}")

    print(f"[node-clfs] Binary detection matrix: shape={B.shape}, "
          f"overall positive rate = {B.mean():.3f}")
    return B


def build_iobt_gt_sequence(session_dir: str,
                           step_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Build GPS ground-truth node sequence for the 10-node IoBT session.

    Parameters
    ----------
    session_dir : path to a single session directory (contains meta_data.json,
                  gt_tracks.csv, and node*_respeaker.flac files)

    Returns
    -------
    gt_seq   : (T_gps,) int32  — 0-based node indices; GPS-covered steps only
    gps_mask : (T_full,) bool  — boolean mask into the full FLAC-aligned step array
    """
    import soundfile as _sf
    from pathlib import Path as _Path
    from collection.train_node_classifiers_10 import (
        load_node_positions_from_meta,
        load_gps_track_10,
        build_ground_truth_sequence_10,
        _discover_session_flac_10,
    )

    session_dir    = _Path(session_dir)
    node_positions = load_node_positions_from_meta(session_dir / "meta_data.json")
    gt_df          = load_gps_track_10(session_dir / "gt_tracks.csv")
    file_map       = _discover_session_flac_10(session_dir)

    # Determine T_full from FLAC durations
    T_full = min(
        int(_sf.info(str(fp)).frames / (_sf.info(str(fp)).samplerate * step_s))
        for fp in file_map.values()
    )

    gt_nid_full = build_ground_truth_sequence_10(
        gt_df, node_positions, step_s, T_full
    )  # (T_full,) int — node IDs 1-10, 0 = no GPS coverage

    gps_mask = gt_nid_full > 0                                   # (T_full,) bool
    gt_seq   = (gt_nid_full[gps_mask] - 1).astype(np.int32)     # 0-based node idx

    cells_seen = sorted(set(gt_seq.tolist()))
    print(f"[iobt-gt] T_full={T_full}  GPS-covered={gps_mask.sum()}  "
          f"0-based nodes visited: {cells_seen}")
    return gt_seq, gps_mask


def build_p_audio_iobt_10(
    pooled_clf_path: str,
    session_dir:     str,
    node_ids:        list[int],
    calibrators=None,
    step_s:          float = 0.5,
) -> np.ndarray:
    """
    Build (T_full, N) float32 calibrated audio probability matrix using the
    10-node pooled LightGBM classifier.

    Parameters
    ----------
    pooled_clf_path : path to pooled_clf_{ts}.pkl (scaler discovered alongside it)
    session_dir     : path to session directory with FLAC files
    node_ids        : list of 1-based node IDs (e.g. list(range(1, 11)))
    calibrators     : dict {node_id: IsotonicRegression} or None

    Returns full (T_full, N) array aligned to FLAC bins.
    Use gps_mask from build_iobt_gt_sequence to slice to GPS-covered steps.
    """
    import joblib
    from pathlib import Path as _Path
    from collection.train_node_classifiers_pooled import build_p_audio_pooled
    from collection.train_node_classifiers_10 import WINDOW_STEPS

    clf_path    = _Path(pooled_clf_path)
    ts          = clf_path.stem.replace("pooled_clf_", "")
    scaler_path = clf_path.parent / f"pooled_scaler_{ts}.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    clf    = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)

    P_raw = build_p_audio_pooled(
        clf, scaler, _Path(session_dir), node_ids,
        step_s=step_s, window=WINDOW_STEPS,
    )  # (T_full, N) float32

    if calibrators is None:
        return P_raw.astype(np.float32)

    P_cal = P_raw.copy()
    for node_idx, nid in enumerate(node_ids):
        ir = calibrators.get(nid)
        if ir is not None:
            P_cal[:, node_idx] = ir.predict(
                P_raw[:, node_idx].astype(np.float64)
            ).astype(np.float32)
    print(f"[p-audio] P_audio shape={P_cal.shape}  "
          f"mean={P_cal.mean():.4f}  max={P_cal.max():.4f}  "
          f"calibrated={calibrators is not None}")
    return P_cal.astype(np.float32)


def build_binary_detection_matrix_10(
    node_clfs_dir: str,
    session_dir:   str,
    node_ids:      list[int],
    step_s:        float = 0.5,
) -> np.ndarray:
    """
    Build (T_full, N) bool binary detection matrix for the 10-node IoBT
    per-node v3 classifiers (results/v3_10node/node_clf_N{nid}_multisession_v3_*.pkl),
    using the same 20-feature amplitude+spectral pipeline they were trained with.

    Returns full (T_full, N) array aligned to FLAC bins, same convention as
    build_p_audio_iobt_10 / build_p_cam_10 — slice with gps_mask before use.
    """
    import glob
    import joblib
    import pandas as _pd
    from pathlib import Path as _Path
    from collection.train_node_classifiers_10 import (
        build_node_features, aggregate_rolling, FEATURE_NAMES, WINDOW_STEPS,
        _discover_session_flac_10,
    )

    session_dir = _Path(session_dir)
    file_map    = _discover_session_flac_10(session_dir)
    missing     = [nid for nid in node_ids if nid not in file_map]
    if missing:
        raise FileNotFoundError(
            f"No FLAC file for node(s) {missing} in {session_dir}"
        )

    raw_and_T = {nid: build_node_features(file_map[nid], step_s) for nid in node_ids}
    T_full    = min(t for _, t in raw_and_T.values())

    B = np.zeros((T_full, len(node_ids)), dtype=bool)
    for col_idx, nid in enumerate(node_ids):
        clf_matches = sorted(glob.glob(
            os.path.join(node_clfs_dir, f"node_clf_N{nid}_*.pkl")
        ))
        scaler_matches = sorted(glob.glob(
            os.path.join(node_clfs_dir, f"node_scaler_N{nid}_*.pkl")
        ))
        if not clf_matches:
            raise FileNotFoundError(f"No node classifier found for node {nid}: {node_clfs_dir}")
        if not scaler_matches:
            raise FileNotFoundError(f"No scaler pkl for node {nid}: {node_clfs_dir}")
        clf    = joblib.load(clf_matches[-1])
        scaler = joblib.load(scaler_matches[-1])

        raw, _ = raw_and_T[nid]
        X = aggregate_rolling(scaler.transform(raw[:T_full]), WINDOW_STEPS)
        B[:, col_idx] = clf.predict(_pd.DataFrame(X, columns=FEATURE_NAMES)).astype(bool)

        print(f"[node-clfs-10] Node {nid}: {os.path.basename(clf_matches[-1])}, "
              f"positive rate = {B[:, col_idx].mean():.3f}")

    print(f"[node-clfs-10] Binary detection matrix: shape={B.shape}, "
          f"overall positive rate = {B.mean():.3f}")
    return B


def evaluate_realdata_policy(algo, cfg, rssi_sequence) -> dict:
    """
    Evaluate algo on real RSSI-posterior object positions.
    Builds a temporary RealDataGridEnv from rssi_sequence and delegates to
    evaluate_policy().  GPS is never used here.
    """
    terminal_prob  = 0.005
    state_prob_run = 0.15
    num_trans      = cfg.get("num_trans", 4)
    cum_prob = [
        i * (1 - terminal_prob - state_prob_run) / float(num_trans - 1)
        for i in range(1, num_trans)
    ]
    cum_prob += [cum_prob[-1] + state_prob_run]

    nrows          = 2
    ncols          = 3
    n_cells        = nrows * ncols
    time_limit     = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    missing_state  = n_cells * (time_limit_max + 1) + 1

    rssi_env = RealDataGridEnv(
        nrows, ncols, num_trans, cum_prob,
        cfg["max_sensors"], cfg["max_sensors_null"],
        missing_state, time_limit,
        rssi_sequence,
        no_moore_constraint=cfg.get("no_moore_constraint", False),
        binary_detections=cfg.get("binary_detections", None),
    )
    eval_cfg = dict(cfg, n_cells=n_cells, missing_state=missing_state)
    return evaluate_policy(algo, rssi_env, eval_cfg)


# ===========================================================================
# Circle validation
# ===========================================================================

def validate_circle_rect(circle_path, nrows, ncols):
    """Check the circle is valid for a rectangular grid: bounds + Moore adjacency."""
    n_cells = nrows * ncols
    for c in circle_path:
        if not (0 <= c < n_cells):
            return False, f"cell {c} out of range for {nrows}×{ncols} grid"
    for i in range(len(circle_path)):
        a  = circle_path[i]
        b  = circle_path[(i + 1) % len(circle_path)]
        ac = a % ncols;  ar = a // ncols
        bc = b % ncols;  br = b // ncols
        if abs(ac - bc) > 1 or abs(ar - br) > 1:
            return False, f"cells {a} and {b} are not Moore-neighbours"
    return True, "OK"


def validate_circle_iobt(circle_path):
    """Check the circle is valid for the 10-node IoBT graph topology."""
    for c in circle_path:
        if not (0 <= c < IOBT_NUM_NODES):
            return False, f"node {c} out of range (0–{IOBT_NUM_NODES-1})"
    for i in range(len(circle_path)):
        a = circle_path[i]
        b = circle_path[(i + 1) % len(circle_path)]
        if b not in IOBT_MAP.get(a, []):
            return False, (f"nodes {a} and {b} are not adjacent in IOBT_MAP "
                           f"(IOBT_MAP[{a}]={IOBT_MAP.get(a, [])})")
    return True, "OK"


# ===========================================================================
# Fine-tune
# ===========================================================================

def finetune(cfg, source_checkpoint, new_run, circle_path=None,
             transition_matrix=None, eval_only=False,
             output_dir=None, eval_finetuned=False,
             realdata=False, train_sequence=None, eval_sequence=None,
             no_moore=False, binary_detections=None, gps_eval=False,
             iobt_gt_sequence=None, iobt_p_audio=None,
             iobt_gt_sequence_eval=None, iobt_p_audio_eval=None,
             soft_threshold=0.5, soft_scale=1.0,
             iobt_prior="circular", sensor_rew=None):
    env_mode      = cfg["env"]
    time_limit    = cfg["time_limit"]
    time_limit_max = cfg["time_limit_max"]
    save_dir      = output_dir or os.path.join(project_root, "runs",
                                                f"agent_run{new_run}_ppo")

    terminal_prob  = 0.005
    state_prob_run = 0.15
    cum_prob = [
        i * (1 - terminal_prob - state_prob_run) / float(cfg["num_trans"] - 1)
        for i in range(1, cfg["num_trans"])
    ]
    cum_prob += [cum_prob[-1] + state_prob_run]

    use_realdata      = realdata and train_sequence is not None
    use_transition    = transition_matrix is not None and not use_realdata
    use_soft_iobt     = iobt_gt_sequence is not None
    iobt_eval_env     = None

    # ── Build env + wrapper depending on mode and movement model ─────────────
    cfg["binary_detections"] = binary_detections

    if use_realdata:
        # GPS-driven training env (2×3 rect, object replays GPS-nearest-node sequence)
        qobj = RealDataLearner(
            cfg["run_number"], 2, 3,
            cfg["num_trans"], cum_prob,
            cfg["max_sensors"], cfg["max_sensors_null"],
            time_limit, time_limit_max,
            train_sequence,
            no_moore_constraint=no_moore,
            binary_detections=binary_detections,
        )
        cfg["n_cells"]       = 6
        cfg["missing_state"] = 6 * (time_limit_max + 1) + 1
        env_wrapper_cls      = grid_environment_rect
    elif use_soft_iobt:
        # 10-node GPS-driven IoBT with soft probability reward
        qobj = RealIoBTLearner(
            cfg["run_number"],
            cfg["max_sensors"], cfg["max_sensors_null"],
            time_limit, time_limit_max,
            iobt_gt_sequence,
            p_audio        = iobt_p_audio,
            soft_threshold = soft_threshold,
            soft_scale     = soft_scale,
            sensor_rew     = sensor_rew,
        )
        cfg["n_cells"]            = IOBT_N * IOBT_N
        cfg["missing_state"]      = IOBT_N * IOBT_N * (time_limit_max + 1) + 1
        cfg["no_moore_constraint"] = True
        env_wrapper_cls           = grid_environment

        # If a distinct eval-only sequence was supplied (multi-session training,
        # where training data is a concatenation of sessions but eval must stay
        # on a single held-out-comparable session), build a separate eval env so
        # the reported accuracy is measured only on that primary session.
        iobt_eval_env = None
        if (iobt_gt_sequence_eval is not None
                and (len(iobt_gt_sequence_eval) != len(iobt_gt_sequence)
                     or not np.array_equal(iobt_gt_sequence_eval, iobt_gt_sequence))):
            iobt_eval_env = RealIoBTEnv(
                cfg["max_sensors"], cfg["max_sensors_null"], cfg["missing_state"],
                time_limit,
                iobt_gt_sequence_eval,
                p_audio        = iobt_p_audio_eval,
                soft_threshold = soft_threshold,
                soft_scale     = soft_scale,
                sensor_rew     = sensor_rew,
            )
    elif env_mode == "iobt":
        if use_transition:
            qobj = TransitionMatrixIoBTLearner(
                cfg["run_number"], cfg["num_trans"],
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                transition_matrix,
            )
        elif iobt_prior == "topo":
            qobj = TopoIoBTLearner(
                cfg["run_number"], cfg["num_trans"],
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                no_moore_constraint=no_moore,
            )
        else:
            qobj = CircularIoBTLearner(
                cfg["run_number"], cfg["num_trans"],
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                circle_path,
                no_moore_constraint=no_moore,
            )
        # n_cells for obs uses N*N (backing grid), matching gym_wrapper.py
        cfg["n_cells"]       = IOBT_N * IOBT_N
        cfg["missing_state"] = IOBT_N * IOBT_N * (time_limit_max + 1) + 1
        env_wrapper_cls = grid_environment
    else:
        if use_transition:
            qobj = TransitionMatrixLearner(
                cfg["run_number"], cfg["nrows"], cfg["ncols"],
                cfg["num_trans"], cum_prob,
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                transition_matrix,
            )
        else:
            qobj = CircularLearner(
                cfg["run_number"], cfg["nrows"], cfg["ncols"],
                cfg["num_trans"], cum_prob,
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                circle_path,
            )
        cfg["n_cells"]       = cfg["nrows"] * cfg["ncols"]
        cfg["missing_state"] = cfg["n_cells"] * (time_limit_max + 1) + 1
        env_wrapper_cls = grid_environment_rect

    env_config = {
        "qobj":                 qobj,
        "time_limit_schedule":  [2000],
        "time_limit_max":       time_limit_max,
        "max_ep_steps":         cfg["max_ep_steps"],
        "no_moore_constraint":  no_moore,
    }

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
            # This sandbox blocks self-connections to the machine's external-facing
            # IP (Ray's auto-detected node IP), which hangs GCS startup. "127.0.0.1"
            # doesn't work here: Ray's services.resolve_ip_for_localhost() special-cases
            # the literal string "127.0.0.1"/"localhost"/"::1" and silently rewrites it
            # back to the auto-detected external IP. Use another loopback address
            # (127.0.0.0/8 is all loopback on Linux) that isn't special-cased, so it
            # survives resolution and the GCS server actually binds/connects locally.
            _node_ip_address="127.0.0.2",
        )

    def _eval(algo_, eval_cfg_):
        if use_realdata:
            return evaluate_realdata_policy(algo_, eval_cfg_, eval_sequence)
        eval_env_ = iobt_eval_env if iobt_eval_env is not None else qobj.grid_env
        return evaluate_policy(algo_, eval_env_, eval_cfg_)

    if eval_only:
        print(f"Loading checkpoint: {source_checkpoint}")
        algo = PPO.from_checkpoint(source_checkpoint)
        metrics = _eval(algo, cfg)
        _print_metrics("EVAL (source)", metrics)
        return

    if eval_finetuned:
        ft_ckpt, _ = find_latest_checkpoint(new_run, save_dir=save_dir)
        if ft_ckpt is None:
            print(f"[ERROR] No fine-tuned checkpoint found in {save_dir}")
            print("        Run fine-tuning first, then use --eval-finetuned.")
            return
        print(f"Loading fine-tuned checkpoint: {ft_ckpt}")
        algo     = PPO.from_checkpoint(ft_ckpt)
        eval_cfg = dict(cfg, eval_episodes=1)
        metrics  = _eval(algo, eval_cfg)
        _print_metrics("EVAL (fine-tuned, 1 episode)", metrics)
        return

    # ── Build fast-convergence PPO config ─────────────────────────────────
    # Use inspect to pick the right parameter names for this RLlib version:
    #   sgd_minibatch_size (old) → minibatch_size (new)
    #   num_sgd_iter       (old) → num_epochs      (new)
    #   clip_param         (old) → clip_epsilon     (new)
    import inspect
    try:
        _accepted = set(inspect.signature(PPOConfig.training).parameters.keys())
    except Exception:
        _accepted = set()

    def _pick(aliases):
        for name, val in aliases:
            if not _accepted or name in _accepted:
                return {name: val}
        return {}

    training_kwargs = dict(
        lr               = cfg["lr"],
        train_batch_size = cfg["train_batch_size"],
        entropy_coeff    = cfg["entropy_coeff"],
        vf_loss_coeff    = cfg["vf_loss_coeff"],
        grad_clip        = cfg["grad_clip"],
    )
    training_kwargs.update(_pick([("num_sgd_iter",       cfg["num_sgd_iter"]),
                                   ("num_epochs",          cfg["num_sgd_iter"])]))
    training_kwargs.update(_pick([("sgd_minibatch_size", cfg["sgd_minibatch_size"]),
                                   ("minibatch_size",      cfg["sgd_minibatch_size"])]))
    training_kwargs.update(_pick([("clip_param",          cfg["clip_param"]),
                                   ("clip_epsilon",         cfg["clip_param"])]))

    ppo_cfg = (
        PPOConfig()
        .environment(env_wrapper_cls, env_config=env_config)
        .framework("torch")
        .training(**training_kwargs)
        .env_runners(
            num_env_runners         = cfg["num_workers"],
            rollout_fragment_length = cfg["rollout_fragment_length"],
        )
        .resources(num_gpus=0)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    ppo_cfg.normalize_actions = False

    algo = ppo_cfg.build()
    if source_checkpoint is not None:
        print(f"Building PPO and restoring from: {source_checkpoint}")
        algo.restore(source_checkpoint)
        print("Checkpoint restored.\n")
    else:
        print("Building PPO from scratch (no checkpoint).\n")

    os.makedirs(save_dir, exist_ok=True)

    # ── Baseline eval before any new training ─────────────────────────────
    if use_soft_iobt:
        print("Baseline (prior policy on IoBT GPS sequence + soft audio reward) ...")
    elif use_realdata:
        print("Baseline (prior policy on real RSSI data) ...")
    else:
        print("Baseline (prior policy on circle) ...")
    baseline = _eval(algo, cfg)
    _print_metrics("BASELINE", baseline)
    best_accuracy = baseline["tracking_accuracy"]
    best_sensors  = baseline["mean_sensors_used"]
    best_ckpt     = source_checkpoint

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"Fine-tuning for {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")
    print(f"  {'iter':>5}  {'reward':>10}  {'accuracy':>10}  {'sensors':>8}  {'ep_len':>8}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    for i in range(1, cfg["training_iterations"] + 1):
        result      = algo.train()
        reward_mean = (result.get("episode_reward_mean")
                       or result.get("env_runners", {}).get("episode_reward_mean")
                       or float("nan"))

        if i % cfg["eval_interval"] == 0 or i == cfg["training_iterations"]:
            metrics = _eval(algo, cfg)
            print(
                f"  {i:5d}  {reward_mean:+10.3f}  "
                f"{metrics['tracking_accuracy']:10.4f}  "
                f"{metrics['mean_sensors_used']:8.2f}  "
                f"{metrics['mean_length']:8.1f}"
            )

            if metrics["tracking_accuracy"] > best_accuracy:
                best_accuracy = metrics["tracking_accuracy"]
                best_sensors  = metrics["mean_sensors_used"]
                best_ckpt     = algo.save(save_dir)
                print(f"         ★ New best {best_accuracy:.4f} — saved")
        else:
            print(f"  {i:5d}  {reward_mean:+10.3f}")

    final_ckpt = algo.save(save_dir)

    print(f"\n{'='*65}")
    print("  FINE-TUNE COMPLETE")
    print(f"{'='*65}")
    print(f"  Mode              : {env_mode.upper()}")
    print(f"  Source checkpoint : {source_checkpoint}")
    print(f"  New checkpoints   : {save_dir}")
    if use_soft_iobt:
        print(f"  Movement model    : IoBT GPS sequence (soft probability reward)")
        print(f"  Sequence length   : {len(iobt_gt_sequence)}")
        print(f"  P_audio shape     : {iobt_p_audio.shape if iobt_p_audio is not None else 'None'}")
        print(f"  Soft scale        : {soft_scale}")
        print(f"  Soft threshold    : {soft_threshold}")
    elif use_realdata:
        if gps_eval:
            print(f"  Movement model    : RF classifier (train) / GPS ground truth (eval)")
        else:
            print(f"  Movement model    : Real data (GPS train / RSSI eval)")
    elif use_transition:
        n = transition_matrix.shape[0]
        print(f"  Movement model    : Transition matrix ({n}×{n})")
    else:
        print(f"  Circle path       : {circle_path}")
    print(f"  Baseline accuracy : {baseline['tracking_accuracy']:.4f}")
    print(f"  Baseline sensors  : {baseline['mean_sensors_used']:.2f}")
    print(f"  Best accuracy     : {best_accuracy:.4f}  ({best_accuracy*100:.2f}%)")
    print(f"  Best sensors      : {best_sensors:.2f}")
    print(f"  Best checkpoint   : {best_ckpt}")
    print(f"  Final checkpoint  : {final_ckpt}")
    delta = best_accuracy - baseline["tracking_accuracy"]
    print(f"  Improvement       : {delta:+.4f}")
    print(f"{'='*65}")

    algo.stop()


def _print_metrics(label, m):
    print(f"\n  ── {label} ──")
    print(f"     Accuracy    : {m['tracking_accuracy']:.4f}  ({m['tracking_accuracy']*100:.2f}%)")
    print(f"     Reward      : {m['mean_reward']:+.3f} ± {m['std_reward']:.3f}")
    print(f"     Ep length   : {m['mean_length']:.1f} steps")
    print(f"     Sensors/step: {m['mean_sensors_used']:.2f}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a PPO checkpoint on a deterministic circular path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment modes
-----------------
  --env rect  (default)
    Rectangular M×N grid.  Circle path is cell indices (row-major).
    Default 2×3 perimeter: [0, 1, 2, 5, 4, 3]

  --env iobt
    10-node Camp Buckner IoBT graph.  Circle path is node indices (0-9).
    Default outer 6-node loop: [0, 4, 3, 6, 2, 1]
    IoBT adjacency: IOBT_MAP at top of file.

Examples
--------
  python examples/finetune_deterministic.py --run 200
  python examples/finetune_deterministic.py --run 200 --new-run 201 --iterations 300
  python examples/finetune_deterministic.py --env iobt --run 195
  python examples/finetune_deterministic.py --env iobt --run 195 --new-run 210
  python examples/finetune_deterministic.py --env iobt --circle 0,1,2,6,3 --run 195
  python examples/finetune_deterministic.py --eval-only --checkpoint runs/agent_run201_ppo
        """
    )
    parser.add_argument("--env",         type=str,   default="rect",
                        choices=["rect", "iobt"],
                        help="Environment mode: 'rect' (default) or 'iobt'")
    parser.add_argument("--run",         type=int,   default=None,
                        help="Source run number to load checkpoint from")
    parser.add_argument("--new-run",     type=int,   default=None,
                        help="Output run number (default: --run + 1)")
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Explicit source checkpoint path (overrides --run)")
    parser.add_argument("--scratch",     action="store_true", default=False,
                        help="Train from scratch: build a fresh PPO policy without "
                             "restoring any checkpoint. --run and --checkpoint are "
                             "ignored when this flag is set.")
    parser.add_argument("--save-dir",    type=str,   default=None)
    parser.add_argument("--time-limit",  type=int,   default=None,
                        help="Override time_limit AND time_limit_max (default 1). "
                             "The tracker tolerates this many consecutive failed "
                             "confirmations before dropping to missing state. "
                             "NOTE: obs/action space depends on time_limit_max, so "
                             "checkpoints trained with a different value are "
                             "INCOMPATIBLE — use --scratch to train a new base.")
    parser.add_argument("--circle",      type=str,   default=None,
                        help="Comma-separated circle path, e.g. '0,4,3,6,2,1'")
    parser.add_argument("--transition",  type=str,   default=None,
                        help="Path to .npy transition matrix file (n_cells×n_cells, row-stochastic)")
    parser.add_argument("--intensity",   type=str,   default=None,
                        help="Path to intensity .npy file (T×6, nodes 11-16). "
                             "Derives a 6×6 transition matrix via z-score + smoothing + "
                             "theoretical-loop blending. Forces --env rect --nrows 2 --ncols 3.")
    parser.add_argument("--intensity-alpha", type=float, default=0.85,
                        help="Theory-blend weight for --intensity (default 0.85; 1=pure theory)")
    parser.add_argument("--random-transition", action="store_true",
                        help="Use a randomly generated transition matrix instead of a circle path")
    parser.add_argument("--nrows",       type=int,   default=None)
    parser.add_argument("--ncols",       type=int,   default=None)
    parser.add_argument("--iterations",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--train-batch", type=int,   default=None)
    parser.add_argument("--entropy",     type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--max-ep-steps", type=int,  default=None)
    parser.add_argument("--eval-only",       action="store_true",
                        help="Evaluate the source checkpoint without training")
    parser.add_argument("--eval-finetuned",  action="store_true",
                        help="Load the saved fine-tuned checkpoint and run 1-episode eval")
    parser.add_argument("--output",          type=str, default=None,
                        help="Output directory for the fine-tuned checkpoint "
                             "(default: {project_root}/runs/agent_run{new_run}_ppo)")

    # ── RF classifier mode ───────────────────────────────────────────────
    parser.add_argument("--clf-pkl",     type=str, default=None,
                        help="Path to trained sklearn .pkl (from supervised_transition.py). "
                             "Uses RF/SVM/MLP predictions as object-position oracle for "
                             "both training and eval. Replaces Bayesian posterior. "
                             "Forces --env rect --nrows 2 --ncols 3.")
    parser.add_argument("--rssi-window", type=int, default=5,
                        help="[--clf-pkl] Rolling-mean window used when building RSSI "
                             "feature matrix (must match value used during RF training; "
                             "default: 5)")
    parser.add_argument("--no-moore",    action="store_true", default=False,
                        help="Disable Moore-neighbourhood window constraint for rect-grid "
                             "tracking. The sensor action covers all n_cells at every "
                             "time-delay level. Use when physical node transitions violate "
                             "2×3 Moore adjacency (e.g. IoBT 6-node layout).")
    parser.add_argument("--iobt-prior", type=str, default="circular",
                        choices=["circular", "topo"],
                        help="Object-movement prior for synthetic IoBT pre-training. "
                             "'circular': deterministic loop along --circle path (default). "
                             "'topo': equal-probability random walk on IOBT_MAP topology "
                             "(stay or move to any neighbour with prob 1/(1+|neighbours|)).")
    parser.add_argument("--advanced-hparams", action="store_true", default=False,
                        help="Use slow-convergence PPO hyperparameters from trainer.py "
                             "(lr=1e-4, batch=4000, entropy=0.01, clip=0.2, grad_clip=30). "
                             "Individual --lr / --entropy / --train-batch overrides still apply.")
    parser.add_argument("--gps-eval",    action="store_true", default=False,
                        help="[--clf-pkl only] Evaluate accuracy against GPS ground truth "
                             "instead of RF classifier predictions. Training rewards still "
                             "use the RF sequence. Requires --gps-csv and --nodes-txt.")
    parser.add_argument("--node-clfs-dir", type=str, default=None,
                        help="Directory containing node_clf_N{nid}_*.pkl files from "
                             "train_node_classifiers.py. Enables binary-gated detection: "
                             "a node must be both activated by Track-MDP AND predict the "
                             "person present. Requires --clf-pkl or --realdata.")

    # ── IoBT 10-node soft-probability reward mode ─────────────────────────
    _D10 = os.path.join(project_root, "iobt_data_10")
    parser.add_argument("--soft-reward",      action="store_true",
                        help="Use soft probability reward for 10-node IoBT fine-tuning. "
                             "Requires --iobt-gt-session and --pooled-clf. "
                             "Forces --env iobt and no-moore-constraint.")
    parser.add_argument("--iobt-gt-session",  type=str, default=None,
                        metavar="YYYYMMDD_HHMMSS",
                        help="[--soft-reward] Session subdir under --iobt-data-dir "
                             "(e.g. 20250812_165739). Drives the IoBT object from GPS. "
                             "This session is always used for the final eval, even if "
                             "--iobt-extra-sessions adds more sessions to training.")
    parser.add_argument("--iobt-extra-sessions", type=str, default=None,
                        metavar="SESSION1,SESSION2,...",
                        help="[--soft-reward] Comma-separated extra session IDs whose "
                             "GT sequence + P_fused are concatenated onto --iobt-gt-session "
                             "for TRAINING only (multi-session training set). Eval always "
                             "stays on --iobt-gt-session alone so accuracy numbers remain "
                             "comparable across runs.")
    parser.add_argument("--iobt-data-dir",    type=str, default=_D10,
                        help=f"[--soft-reward] Root data directory for 10-node IoBT sessions "
                             f"(default: {_D10})")
    parser.add_argument("--pooled-clf",       type=str, default=None,
                        help="[--soft-reward] Path to pooled_clf_{{ts}}.pkl from "
                             "train_node_classifiers_pooled.py. "
                             "Paired pooled_scaler_{{ts}}.pkl must be alongside it.")
    parser.add_argument("--node-clfs-dir-10", type=str, default=None,
                        help="[--soft-reward] Directory of 10-node per-node binary "
                             "classifiers (node_clf_N{1..10}_multisession_v3_*.pkl + "
                             "paired node_scaler_N*.pkl, from train_node_classifiers_10.py). "
                             "When set, P_fused is gated: zeroed at (t, node) where the "
                             "binary classifier does not confirm a hit.")
    parser.add_argument("--calibrators",      type=str, default=None,
                        help="[--soft-reward] Path to pooled_calibrators_{{ts}}.pkl "
                             "(isotonic per-node calibration, optional).")
    parser.add_argument("--soft-scale",       type=float, default=1.0,
                        help="[--soft-reward] Multiplier on soft detection probability "
                             "(default: 1.0)")
    parser.add_argument("--soft-threshold",   type=float, default=0.5,
                        help="[--soft-reward] Probability threshold for binary "
                             "state-transition trigger (default: 0.5)")
    parser.add_argument("--fusion-mode",      type=str,   default="audio",
                        choices=["audio", "camera", "or_max", "weighted", "cam_fallback", "gt"],
                        help="[--soft-reward] Sensor fusion strategy for P_fused. "
                             "'audio': audio-only (default, backward compatible). "
                             "'camera': YOLO camera-only. "
                             "'or_max': elementwise max of audio and camera. "
                             "'weighted': audio_weight*P_audio + cam_weight*P_cam. "
                             "'cam_fallback': camera when P_cam>0, else audio "
                             "(camera-priority with audio gap-fill). "
                             "'gt': oracle upper bound — P_fused is a perfect one-hot "
                             "built directly from GPS ground truth, bypassing the audio/camera "
                             "classifiers entirely (RESUME.md next-step #7).")
    parser.add_argument("--audio-weight",     type=float, default=0.5,
                        help="[--fusion-mode weighted] Weight for audio channel (default: 0.5)")
    parser.add_argument("--cam-weight",       type=float, default=0.5,
                        help="[--fusion-mode weighted] Weight for camera channel (default: 0.5)")
    parser.add_argument("--cam-fallback-thresh", type=float, default=0.0,
                        help="[--fusion-mode cam_fallback] Camera confidence threshold above which "
                             "camera is used instead of audio. 0.0=any detection fires camera "
                             "(default, equiv to or_max for this dataset). 0.65=only "
                             "high-confidence YOLO detections suppress audio.")
    parser.add_argument("--cam-classes", type=str, default="car",
                        help="Comma-separated YOLO classes accepted as the tracked "
                             "object (default: car). Session 20250812_091600's object "
                             "is detected as 'truck' — the hardcoded 'car' filter "
                             "discarded 84%% of its GT-bin detections (diag_camera_audit).")
    parser.add_argument("--cam-min-conf", type=float, default=0.5,
                        help="Min YOLO confidence for camera detections (default 0.5).")
    parser.add_argument("--cam-smooth-bins", type=int, default=0,
                        help="Temporally max-pool P_cam over +/-N bins (N*0.5s) per node "
                             "before fusion, carrying camera detections forward/backward "
                             "to bridge short detection gaps (the object cannot teleport "
                             "at 0.5s resolution). 0 = no smoothing (default).")
    parser.add_argument("--sensor-rew", type=float, default=None,
                        help="[--soft-reward] Per-active-sensor energy penalty in the "
                             "reward (default: keep RealIoBTEnv's -0.25). Weaker "
                             "penalties (e.g. -0.05..-0.10) trade energy for accuracy — "
                             "the run-612 diagnosis: tl=3 slack went to frugality "
                             "because -0.25 dominated. Does not affect the obs/action "
                             "space, so checkpoints remain compatible across values.")
    parser.add_argument("--max-sensors", type=int, default=None,
                        help="Max sensors the policy may activate per step outside the "
                             "missing state (default: keep IOBT_DEFAULTS' 6 of 10). "
                             "Energy trade-off lever; does not affect the obs/action "
                             "space, so checkpoints remain compatible across values.")

    # ── Real-data replay mode ─────────────────────────────────────────────
    _D = os.path.join(project_root, "iobt_data")
    parser.add_argument("--realdata",       action="store_true",
                        help="Drive object from real GPS+RSSI data (forces rect 2×3). "
                             "Training env uses GPS ground truth; eval uses RSSI posterior only.")
    parser.add_argument("--gt",             action="store_true", default=False,
                        help="Ground-truth mode: drive object from GPS for BOTH training "
                             "and evaluation. No FLAC data or path-loss params needed. "
                             "Forces --env rect --nrows 2 --ncols 3. "
                             "Use to establish an oracle upper bound independent of "
                             "the per-node acoustic classifiers.")
    parser.add_argument("--nodes-txt",      type=str,
                        default=os.path.join(_D, "node_positions.txt"),
                        help="[--realdata] Tab-delimited node_positions.txt "
                             "(default: iobt_data/node_positions.txt)")
    parser.add_argument("--gps-csv",        type=str,
                        default=os.path.join(_D, "20260416_154037_gps2_gps.csv"),
                        help="[--realdata] GPS ground-truth CSV, training only "
                             "(default: iobt_data/20260416_154037_gps2_gps.csv)")
    parser.add_argument("--path-loss-json", type=str,
                        default=os.path.join(_D, "path_loss_params_20260416.json"),
                        help="[--realdata] Trained path-loss params JSON; auto-trained if absent "
                             "(default: iobt_data/path_loss_params_20260416.json)")
    parser.add_argument("--flac-dir",       type=str,
                        default=_D,
                        help="[--realdata] Folder with FLAC files "
                             "(default: iobt_data/)")
    parser.add_argument("--session",        type=str,
                        default="20260416_154037",
                        help="[--realdata] Session prefix to filter FLAC files "
                             "(default: 20260416_154037)")

    args = parser.parse_args()

    # ── Pick defaults for the selected env mode ───────────────────────────
    cfg = dict(IOBT_DEFAULTS if args.env == "iobt" else RECT_DEFAULTS)
    cfg["env"] = args.env

    if args.run        is not None: cfg["run_number"]          = args.run
    if args.nrows      is not None: cfg["nrows"]               = args.nrows
    if args.ncols      is not None: cfg["ncols"]               = args.ncols
    if args.iterations is not None: cfg["training_iterations"] = args.iterations
    if args.eval_episodes is not None: cfg["eval_episodes"]    = args.eval_episodes
    if args.max_ep_steps  is not None: cfg["max_ep_steps"]     = args.max_ep_steps
    if args.time_limit is not None:
        cfg["time_limit"]     = args.time_limit
        cfg["time_limit_max"] = args.time_limit
        print(f"[time-limit] time_limit = time_limit_max = {args.time_limit} "
              f"(obs/action space differs from time_limit_max=1 checkpoints)")
    if args.sensor_rew is not None:
        print(f"[sensor-rew] per-sensor energy penalty = {args.sensor_rew} "
              f"(default -0.25; checkpoint-compatible)")
    if args.max_sensors is not None:
        cfg["max_sensors"] = args.max_sensors
        print(f"[max-sensors] max_sensors = {args.max_sensors} "
              f"(default 6 of 10; checkpoint-compatible)")
    cfg["no_moore_constraint"] = args.no_moore

    if args.advanced_hparams:
        cfg.update(ADVANCED_HPARAMS)

    # Per-param overrides always win over --advanced-hparams
    if args.lr          is not None: cfg["lr"]               = args.lr
    if args.train_batch is not None: cfg["train_batch_size"] = args.train_batch
    if args.entropy     is not None: cfg["entropy_coeff"]    = args.entropy

    new_run = args.new_run if args.new_run is not None else cfg["run_number"]

    # ── Determine movement model: real data, transition matrix, or circle ───
    train_sequence = None
    eval_sequence  = None

    if args.clf_pkl:
        if not os.path.exists(args.clf_pkl):
            print(f"[ERROR] --clf-pkl file not found: {args.clf_pkl}")
            sys.exit(1)
        cfg["env"]   = "rect"
        cfg["nrows"] = 2
        cfg["ncols"] = 3
        print("[clf-pkl] Building RF classifier sequence …")
        rf_sequence = build_rf_sequence(
            args.clf_pkl, args.flac_dir, args.nodes_txt,
            args.session, rssi_window=args.rssi_window, step_s=0.5,
        )
        train_sequence    = rf_sequence
        eval_sequence     = rf_sequence
        if args.gps_eval:
            print("[gps-eval] Building GPS ground-truth eval sequence ...")
            eval_sequence = build_gps_sequence(args.nodes_txt, args.gps_csv)
        circle_path       = None
        transition_matrix = None
        args.realdata     = True
    elif args.realdata:
        cfg["env"]   = "rect"
        cfg["nrows"] = 2
        cfg["ncols"] = 3
        print("[realdata] Building GPS training sequence ...")
        train_sequence = build_gps_sequence(args.nodes_txt, args.gps_csv)
        print("[realdata] Building RSSI eval sequence ...")
        eval_sequence = build_rssi_sequence(
            args.nodes_txt, args.flac_dir, args.path_loss_json,
            gps_csv=args.gps_csv, session=args.session, step_s=0.5,
        )
        circle_path       = None
        transition_matrix = None
    elif args.gt:
        cfg["env"]   = "rect"
        cfg["nrows"] = 2
        cfg["ncols"] = 3
        print("[gt] Building GPS ground-truth sequence (train + eval) ...")
        gps_sequence   = build_gps_sequence(args.nodes_txt, args.gps_csv)
        train_sequence = gps_sequence
        eval_sequence  = gps_sequence
        circle_path       = None
        transition_matrix = None
        args.realdata     = True   # reuse existing use_realdata code path
    elif args.intensity:
        # Intensity .npy → 6×6 transition matrix; force rect 2×3
        intensity_path = os.path.abspath(args.intensity)
        if not os.path.exists(intensity_path):
            print(f"[ERROR] Intensity file not found: {intensity_path}")
            sys.exit(1)
        if args.env != "rect":
            print("[INFO] --intensity forces --env rect")
            cfg["env"] = "rect"
            args.env   = "rect"
        if cfg["nrows"] != 2 or cfg["ncols"] != 3:
            print("[INFO] --intensity forces --nrows 2 --ncols 3")
            cfg["nrows"] = 2
            cfg["ncols"] = 3
        transition_matrix = intensity_to_transition(
            intensity_path, alpha=args.intensity_alpha
        )
        print(f"[INFO] Derived 6×6 transition matrix from intensity: {intensity_path}")
        circle_path = None
    elif args.random_transition or args.transition:
        if args.transition:
            T_path = os.path.abspath(args.transition)
            if not os.path.exists(T_path):
                print(f"[ERROR] Transition matrix file not found: {T_path}")
                sys.exit(1)
            transition_matrix = np.load(T_path)
            print(f"[INFO] Loaded transition matrix from: {T_path}  shape={transition_matrix.shape}")
        else:
            n_cells = (IOBT_NUM_NODES if args.env == "iobt"
                       else cfg["nrows"] * cfg["ncols"])
            transition_matrix = make_random_transition_matrix(n_cells)
            print(f"[INFO] Generated random transition matrix  shape={transition_matrix.shape}")
        circle_path = None
    else:
        # Circle mode (original behaviour)
        if args.circle:
            try:
                circle_path = [int(x.strip()) for x in args.circle.split(",")]
            except ValueError:
                print(f"[ERROR] --circle must be comma-separated integers, got: {args.circle}")
                sys.exit(1)
        elif args.env == "iobt":
            circle_path = list(IOBT_CIRCLE_PATH)
        else:
            circle_path = list(RECT_CIRCLE_PATH)

        # Validate circle
        if args.env == "iobt":
            ok, msg = validate_circle_iobt(circle_path)
        else:
            ok, msg = validate_circle_rect(circle_path, cfg["nrows"], cfg["ncols"])
        if not ok:
            print(f"[ERROR] Circle path invalid: {msg}")
            print(f"  circle_path = {circle_path}")
            sys.exit(1)
        transition_matrix = None

    # ── Binary detection matrix (per-node classifiers) ───────────────────
    binary_detections = None
    if args.node_clfs_dir:
        if not (args.clf_pkl or getattr(args, 'realdata', False) or args.gt):
            print("[WARN] --node-clfs-dir is only used with --clf-pkl or --realdata; "
                  "ignoring.")
        else:
            print("[node-clfs] Building binary detection matrix …")
            binary_detections = build_binary_detection_matrix(
                args.node_clfs_dir, args.flac_dir, args.session,
                rssi_window=args.rssi_window,
            )

    # ── IoBT 10-node soft-reward mode ─────────────────────────────────────
    iobt_gt_sequence      = None
    iobt_p_audio          = None
    iobt_gt_sequence_eval = None
    iobt_p_audio_eval     = None

    if getattr(args, 'soft_reward', False):
        if not args.iobt_gt_session:
            print("[ERROR] --soft-reward requires --iobt-gt-session")
            sys.exit(1)
        if not args.pooled_clf:
            print("[ERROR] --soft-reward requires --pooled-clf")
            sys.exit(1)
        if not os.path.exists(args.pooled_clf):
            print(f"[ERROR] --pooled-clf file not found: {args.pooled_clf}")
            sys.exit(1)

        # Force IoBT mode with no-moore constraint
        args.env                 = "iobt"
        cfg["env"]               = "iobt"
        args.no_moore            = True
        cfg["no_moore_constraint"] = True

        fusion_mode = getattr(args, 'fusion_mode', 'audio')
        import joblib as _jl
        from pathlib import Path as _Path
        calibrators = _jl.load(args.calibrators) if args.calibrators else None

        def _build_session_fused(session_id):
            """Return (gt_seq, P_fused_gps) for one session, GPS-covered steps only."""
            session_dir = os.path.join(args.iobt_data_dir, session_id)
            if not os.path.isdir(session_dir):
                print(f"[ERROR] IoBT session directory not found: {session_dir}")
                sys.exit(1)

            print(f"[soft-reward] [{session_id}] Building IoBT GPS ground-truth sequence …")
            gt_seq, gps_mask = build_iobt_gt_sequence(session_dir, step_s=0.5)

            if fusion_mode == 'gt':
                # Oracle upper bound: P_fused is a perfect one-hot from GPS ground
                # truth, bypassing the audio/camera classifiers entirely.
                print(f"[soft-reward] [{session_id}] Building ORACLE P_fused "
                      f"(one-hot from GPS ground truth, no classifiers) …")
                T_full = len(gps_mask)
                P_fused_full = np.zeros((T_full, 10), dtype=np.float32)
                P_fused_full[gps_mask, gt_seq] = 1.0
                print(f"  [oracle] P_fused: mean={P_fused_full.mean():.3f}  "
                      f"max={P_fused_full.max():.3f}  gps_coverage={gps_mask.mean():.3f}")
                p_fused_gps = P_fused_full[gps_mask]  # (T_gps, 10)
                print(f"[soft-reward] [{session_id}] Aligned P_fused: {p_fused_gps.shape}  "
                      f"gt_seq: {len(gt_seq)}")
                return gt_seq, p_fused_gps

            print(f"[soft-reward] [{session_id}] Building pooled audio probability matrix …")
            P_audio_full = build_p_audio_iobt_10(
                args.pooled_clf, session_dir,
                list(range(1, 11)), calibrators=calibrators, step_s=0.5,
            )  # (T_full, 10)

            if fusion_mode != 'audio':
                from collection.train_node_classifiers_10 import (
                    build_p_cam_10 as _build_p_cam_10,
                    _parse_session_start_10,
                )
                session_start_unix = _parse_session_start_10(session_id).timestamp()
                T_full = P_audio_full.shape[0]
                print(f"[soft-reward] [{session_id}] Building camera probability matrix "
                      f"(fusion={fusion_mode}) …")
                P_cam_full = _build_p_cam_10(
                    _Path(session_dir), list(range(1, 11)),
                    session_start_unix, T_full, bin_size_s=0.5,
                    target_class=set(
                        getattr(args, 'cam_classes', 'car').split(',')),
                    min_conf=getattr(args, 'cam_min_conf', 0.5),
                )  # (T_full, 10)
                print(f"  [camera] P_cam: mean={P_cam_full.mean():.3f}  "
                      f"max={P_cam_full.max():.3f}  "
                      f"positive_rate={(P_cam_full > 0).mean():.3f}")

                smooth_w = getattr(args, 'cam_smooth_bins', 0) or 0
                if smooth_w > 0:
                    # Max-pool each node's P_cam over a +/-smooth_w bin window
                    smoothed = P_cam_full.copy()
                    for s in range(1, smooth_w + 1):
                        smoothed[s:]  = np.maximum(smoothed[s:],  P_cam_full[:-s])
                        smoothed[:-s] = np.maximum(smoothed[:-s], P_cam_full[s:])
                    P_cam_full = smoothed
                    print(f"  [cam-smooth] max-pooled P_cam over +/-{smooth_w} bins "
                          f"({smooth_w * 0.5:.1f}s): mean={P_cam_full.mean():.3f}  "
                          f"positive_rate={(P_cam_full > 0).mean():.3f}")

                if fusion_mode == 'camera':
                    P_fused_full = P_cam_full.astype(np.float32)
                elif fusion_mode == 'or_max':
                    P_fused_full = np.maximum(P_audio_full, P_cam_full).astype(np.float32)
                elif fusion_mode == 'weighted':
                    aw = getattr(args, 'audio_weight', 0.5)
                    cw = getattr(args, 'cam_weight',   0.5)
                    P_fused_full = np.clip(
                        aw * P_audio_full + cw * P_cam_full, 0.0, 1.0
                    ).astype(np.float32)
                elif fusion_mode == 'cam_fallback':
                    # Use camera above threshold, audio elsewhere
                    cam_thresh = getattr(args, 'cam_fallback_thresh', 0.0)
                    P_fused_full = np.where(
                        P_cam_full > cam_thresh, P_cam_full, P_audio_full
                    ).astype(np.float32)
                else:
                    P_fused_full = P_audio_full

                print(f"  [fused]  P_fused: mean={P_fused_full.mean():.3f}  "
                      f"max={P_fused_full.max():.3f}")
                if fusion_mode == 'cam_fallback':
                    cam_covered = (P_cam_full > 0).mean()
                    audio_fill  = ((P_cam_full == 0) & (P_audio_full > 0)).mean()
                    print(f"  [cam_fallback] cam coverage: {cam_covered:.3f}  "
                          f"audio fill-in: {audio_fill:.3f}")
            else:
                P_fused_full = P_audio_full

            if args.node_clfs_dir_10:
                print(f"[soft-reward] [{session_id}] Building 10-node binary "
                      f"detection gate …")
                B_full = build_binary_detection_matrix_10(
                    args.node_clfs_dir_10, session_dir, list(range(1, 11)),
                    step_s=0.5,
                )  # (T_full, 10) bool
                T_gate = min(P_fused_full.shape[0], B_full.shape[0])
                pre_mean = P_fused_full[:T_gate].mean()
                P_fused_full = P_fused_full[:T_gate] * B_full[:T_gate].astype(np.float32)
                gps_mask = gps_mask[:T_gate]
                gt_seq   = gt_seq[:gps_mask.sum()] if len(gt_seq) > gps_mask.sum() else gt_seq
                print(f"  [node-clfs-10 gate] P_fused mean {pre_mean:.4f} -> "
                      f"{P_fused_full.mean():.4f} "
                      f"(zeroed {1.0 - B_full[:T_gate].mean():.3f} of cells)")

            # Align to GPS-covered timesteps (same for all modalities)
            p_fused_gps = P_fused_full[gps_mask]  # (T_gps, 10)
            print(f"[soft-reward] [{session_id}] Aligned P_fused: {p_fused_gps.shape}  "
                  f"gt_seq: {len(gt_seq)}")
            return gt_seq, p_fused_gps

        primary_gt, primary_p = _build_session_fused(args.iobt_gt_session)
        iobt_gt_sequence_eval, iobt_p_audio_eval = primary_gt, primary_p

        extra_sessions = [s.strip() for s in (args.iobt_extra_sessions or "").split(",")
                           if s.strip()]
        if extra_sessions:
            gt_parts = [primary_gt]
            p_parts  = [primary_p]
            for sess in extra_sessions:
                sess_gt, sess_p = _build_session_fused(sess)
                gt_parts.append(sess_gt)
                p_parts.append(sess_p)
            iobt_gt_sequence = np.concatenate(gt_parts).astype(np.int32)
            iobt_p_audio     = np.concatenate(p_parts, axis=0).astype(np.float32)
            print(f"[soft-reward] Multi-session training set: {[args.iobt_gt_session] + extra_sessions} "
                  f"-> combined gt_seq={len(iobt_gt_sequence)}  P_fused={iobt_p_audio.shape}  "
                  f"(eval stays on {args.iobt_gt_session} alone: "
                  f"gt_seq={len(iobt_gt_sequence_eval)})")
        else:
            iobt_gt_sequence = primary_gt
            iobt_p_audio     = primary_p

    # ── Locate source checkpoint ──────────────────────────────────────────
    if args.scratch:
        source_ckpt = None
    elif args.checkpoint:
        source_ckpt = args.checkpoint
        if not os.path.exists(source_ckpt):
            print(f"[ERROR] Checkpoint not found: {source_ckpt}")
            sys.exit(1)
    else:
        source_ckpt, search_dir = find_latest_checkpoint(
            cfg["run_number"], save_dir=args.save_dir
        )
        if source_ckpt is None:
            print(f"[ERROR] No checkpoint found for run {cfg['run_number']} in {search_dir}")
            sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("  TRACK-MDP — DETERMINISTIC FINE-TUNE")
    print("=" * 65)
    print(f"  Mode              : {args.env.upper()}")
    if not args.scratch:
        print(f"  Source run        : {cfg['run_number']}")
        print(f"  Source checkpoint : {source_ckpt}")
    else:
        print(f"  Source checkpoint : (none — training from scratch)")
    print(f"  Output run        : {new_run}")
    if args.env == "iobt":
        print(f"  Graph             : IoBT 10-node (Camp Buckner)")
    else:
        print(f"  Grid              : {cfg['nrows']} × {cfg['ncols']}")
    if getattr(args, 'soft_reward', False) and iobt_gt_sequence is not None:
        print(f"  Movement model    : IoBT GPS + soft audio reward")
        print(f"  Session           : {args.iobt_gt_session}")
        print(f"  Pooled clf        : {args.pooled_clf}")
        print(f"  Calibrators       : {args.calibrators or 'none'}")
        print(f"  Soft scale        : {args.soft_scale}")
        print(f"  Soft threshold    : {args.soft_threshold}")
        print(f"  Sensor penalty    : "
              f"{args.sensor_rew if args.sensor_rew is not None else -0.25}")
        print(f"  Max sensors       : {cfg['max_sensors']}")
        print(f"  Fusion mode       : {getattr(args, 'fusion_mode', 'audio')}")
        print(f"  GT seq length     : {len(iobt_gt_sequence)}")
        print(f"  P_audio shape     : {iobt_p_audio.shape}")
    elif args.clf_pkl:
        eval_label = "GPS ground truth" if args.gps_eval else "RF classifier"
        print(f"  Movement model    : RF classifier (train) / {eval_label} (eval)")
        print(f"  Classifier pkl    : {args.clf_pkl}")
        print(f"  RSSI window       : {args.rssi_window}")
        print(f"  Train seq length  : {len(train_sequence)}")
        print(f"  Eval seq length   : {len(eval_sequence)}")
    elif args.gt:
        print(f"  Movement model    : GPS ground truth (train + eval)")
        print(f"  GPS CSV           : {args.gps_csv}")
        print(f"  Seq length        : {len(train_sequence)}")
    elif args.realdata:
        print(f"  Movement model    : Real data (GPS train / RSSI eval)")
        print(f"  Train seq length  : {len(train_sequence)}")
        print(f"  Eval seq length   : {len(eval_sequence)}")
    elif transition_matrix is not None:
        n = transition_matrix.shape[0]
        src = args.intensity or args.transition or "random"
        print(f"  Movement model    : Transition matrix ({n}×{n})  [source: {src}]")
    else:
        print(f"  Circle path       : {circle_path}")
    print(f"  Iterations        : {cfg['training_iterations']}")
    print(f"  Eval every        : {cfg['eval_interval']} iteration(s)")
    print(f"  Max ep steps      : {cfg['max_ep_steps']}")
    if transition_matrix is not None:
        print()
        print("  Transition matrix (rows = from, cols = to):")
        n = transition_matrix.shape[0]
        header = "       " + "".join(f"  [{j}]" for j in range(n))
        print(header)
        for i, row in enumerate(transition_matrix):
            print(f"    [{i}]  " + "".join(f"{v:6.3f}" for v in row))
    print()
    regime = "advanced (slow-convergence)" if args.advanced_hparams else "fast-convergence"
    print(f"  PPO hyperparameters ({regime}):")
    for k in ("lr", "train_batch_size", "num_sgd_iter", "sgd_minibatch_size",
              "clip_param", "entropy_coeff", "rollout_fragment_length", "num_workers"):
        print(f"    {k:30s}: {cfg[k]}")
    output_dir = args.output or os.path.join(
        project_root, "runs", f"agent_run{new_run}_ppo"
    )
    if args.eval_finetuned:
        mode_label = "eval fine-tuned (1 episode)"
    elif args.eval_only:
        mode_label = "eval source only"
    else:
        mode_label = "fine-tune"
    print(f"  Output dir        : {output_dir}")
    print(f"  Mode              : {mode_label}")

    try:
        finetune(cfg, source_ckpt, new_run, circle_path=circle_path,
                 transition_matrix=transition_matrix, eval_only=args.eval_only,
                 output_dir=output_dir, eval_finetuned=args.eval_finetuned,
                 realdata=getattr(args, 'realdata', False),
                 train_sequence=train_sequence, eval_sequence=eval_sequence,
                 no_moore=args.no_moore,
                 binary_detections=binary_detections,
                 gps_eval=args.gps_eval,
                 iobt_gt_sequence=iobt_gt_sequence,
                 iobt_p_audio=iobt_p_audio,
                 iobt_gt_sequence_eval=iobt_gt_sequence_eval,
                 iobt_p_audio_eval=iobt_p_audio_eval,
                 soft_threshold=getattr(args, 'soft_threshold', 0.5),
                 soft_scale=getattr(args, 'soft_scale', 1.0),
                 iobt_prior=getattr(args, 'iobt_prior', 'circular'),
                 sensor_rew=getattr(args, 'sensor_rew', None))
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
