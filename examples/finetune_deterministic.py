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


# Augment vecs (must match gym_wrapper)
def _build_augment_vecs(tlm):
    v1 = [(2 * i + 3) ** 2 for i in range(tlm + 1)]
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
                 circle_path):
        # Set _circle BEFORE super().__init__() — iobt_env.__init__ ends with
        # self.reset_object_state(), which our override requires.
        for node in circle_path:
            assert 0 <= node < IOBT_NUM_NODES, \
                f"Circle node {node} out of range (0–{IOBT_NUM_NODES-1})"
        self._circle     = circle_path
        self._circle_len = len(circle_path)
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
        """Same as parent but with terminal_flag forced to 0."""
        reward, next_state, _, new_delay = super().get_reward_next_state(
            current_state, current_action, time_delay
        )
        return reward, next_state, 0, new_delay


class CircularIoBTLearner:
    """Wraps CircularIoBTEnv to match the qobj interface expected by the wrappers."""

    def __init__(self, run_number, num_trans, max_sensors, max_sensors_null,
                 time_limit, time_limit_max, circle_path):
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
    av1, av        = _build_augment_vecs(time_limit_max)

    total_rewards, ep_lengths, sensors_ep = [], [], []
    total_steps = total_found = 0

    for _ in range(cfg["eval_episodes"]):
        env.reset_object_state()
        current_state = missing_state
        time_delay    = 0
        actions_list  = []
        ep_reward = ep_steps = ep_sensors = 0

        while ep_steps < cfg["max_ep_steps"]:
            num_sensors = (2 * time_delay + 3) ** 2
            was_missing = (current_state == missing_state)

            action_vec = [1] * ACTION_HISTORY_LEN
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
                td_ac       = av1[time_delay - 1]
                action_bool = [1] * MAX_ACTION_SZ if was_missing else list(action_full)
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
             output_dir=None, eval_finetuned=False):
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

    use_transition = transition_matrix is not None

    # ── Build env + wrapper depending on mode and movement model ─────────────
    if env_mode == "iobt":
        if use_transition:
            qobj = TransitionMatrixIoBTLearner(
                cfg["run_number"], cfg["num_trans"],
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                transition_matrix,
            )
        else:
            qobj = CircularIoBTLearner(
                cfg["run_number"], cfg["num_trans"],
                cfg["max_sensors"], cfg["max_sensors_null"],
                time_limit, time_limit_max,
                circle_path,
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
        "qobj":                qobj,
        "time_limit_schedule": [2000],
        "time_limit_max":      time_limit_max,
        "max_ep_steps":        cfg["max_ep_steps"],
    }

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
        )

    if eval_only:
        print(f"Loading checkpoint: {source_checkpoint}")
        algo    = PPO.from_checkpoint(source_checkpoint)
        metrics = evaluate_policy(algo, qobj.grid_env, cfg)
        _print_metrics("EVAL (source)", metrics)
        return

    if eval_finetuned:
        ft_ckpt, _ = find_latest_checkpoint(new_run, save_dir=save_dir)
        if ft_ckpt is None:
            print(f"[ERROR] No fine-tuned checkpoint found in {save_dir}")
            print("        Run fine-tuning first, then use --eval-finetuned.")
            return
        print(f"Loading fine-tuned checkpoint: {ft_ckpt}")
        algo = PPO.from_checkpoint(ft_ckpt)
        eval_cfg = dict(cfg, eval_episodes=1)
        metrics  = evaluate_policy(algo, qobj.grid_env, eval_cfg)
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

    # ── Restore from source checkpoint ────────────────────────────────────
    print(f"Building PPO and restoring from: {source_checkpoint}")
    algo = ppo_cfg.build()
    algo.restore(source_checkpoint)
    print("Checkpoint restored.\n")

    os.makedirs(save_dir, exist_ok=True)

    # ── Baseline eval before any new training ─────────────────────────────
    print("Baseline (prior policy on circle) ...")
    baseline = evaluate_policy(algo, qobj.grid_env, cfg)
    _print_metrics("BASELINE", baseline)
    best_accuracy = baseline["tracking_accuracy"]
    best_ckpt     = source_checkpoint

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"Fine-tuning for {cfg['training_iterations']} iterations "
          f"(eval every {cfg['eval_interval']}) ...\n")
    print(f"  {'iter':>5}  {'reward':>10}  {'accuracy':>10}  {'sensors':>8}  {'ep_len':>8}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    for i in range(1, cfg["training_iterations"] + 1):
        result      = algo.train()
        reward_mean = (result.get("episode_reward_mean") or
                       result.get("env_runners", {}).get("episode_reward_mean",
                                                          float("nan")))

        if i % cfg["eval_interval"] == 0 or i == cfg["training_iterations"]:
            metrics = evaluate_policy(algo, qobj.grid_env, cfg)
            print(
                f"  {i:5d}  {reward_mean:+10.3f}  "
                f"{metrics['tracking_accuracy']:10.4f}  "
                f"{metrics['mean_sensors_used']:8.2f}  "
                f"{metrics['mean_length']:8.1f}"
            )

            if metrics["tracking_accuracy"] > best_accuracy:
                best_accuracy = metrics["tracking_accuracy"]
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
    if use_transition:
        n = transition_matrix.shape[0]
        print(f"  Movement model    : Transition matrix ({n}×{n})")
    else:
        print(f"  Circle path       : {circle_path}")
    print(f"  Baseline accuracy : {baseline['tracking_accuracy']:.4f}")
    print(f"  Best accuracy     : {best_accuracy:.4f}  ({best_accuracy*100:.2f}%)")
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
    parser.add_argument("--save-dir",    type=str,   default=None)
    parser.add_argument("--circle",      type=str,   default=None,
                        help="Comma-separated circle path, e.g. '0,4,3,6,2,1'")
    parser.add_argument("--transition",  type=str,   default=None,
                        help="Path to .npy transition matrix file (n_cells×n_cells, row-stochastic)")
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
    args = parser.parse_args()

    # ── Pick defaults for the selected env mode ───────────────────────────
    cfg = dict(IOBT_DEFAULTS if args.env == "iobt" else RECT_DEFAULTS)
    cfg["env"] = args.env

    if args.run        is not None: cfg["run_number"]          = args.run
    if args.nrows      is not None: cfg["nrows"]               = args.nrows
    if args.ncols      is not None: cfg["ncols"]               = args.ncols
    if args.iterations is not None: cfg["training_iterations"] = args.iterations
    if args.lr         is not None: cfg["lr"]                  = args.lr
    if args.train_batch is not None: cfg["train_batch_size"]   = args.train_batch
    if args.entropy    is not None: cfg["entropy_coeff"]       = args.entropy
    if args.eval_episodes is not None: cfg["eval_episodes"]    = args.eval_episodes
    if args.max_ep_steps  is not None: cfg["max_ep_steps"]     = args.max_ep_steps

    new_run = args.new_run if args.new_run is not None else cfg["run_number"]

    # ── Determine movement model: transition matrix or circle path ────────
    if args.random_transition or args.transition:
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

    # ── Locate source checkpoint ──────────────────────────────────────────
    if args.checkpoint:
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
    print(f"  Source run        : {cfg['run_number']}")
    print(f"  Source checkpoint : {source_ckpt}")
    print(f"  Output run        : {new_run}")
    if args.env == "iobt":
        print(f"  Graph             : IoBT 10-node (Camp Buckner)")
    else:
        print(f"  Grid              : {cfg['nrows']} × {cfg['ncols']}")
    if transition_matrix is not None:
        n = transition_matrix.shape[0]
        print(f"  Movement model    : Transition matrix ({n}×{n})")
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
    print("  Fast-convergence PPO hyperparameters:")
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
                 output_dir=output_dir, eval_finetuned=args.eval_finetuned)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
