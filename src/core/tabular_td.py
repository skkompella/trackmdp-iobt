"""
tabular_td.py — tabular SARSA(lambda) for the 10-node IoBT tracking MDP.

Track-MDP's state is deliberately tiny — last known node plus a staleness
counter — so the whole problem fits in a table of a few thousand floats:

    time_limit=1, max_sensors=2  ->  21 states x   55 actions =  1,155 entries
    time_limit=1, max_sensors=6  ->  21 states x  847 actions = 17,787 entries

against PPO's MultiDiscrete([2]*16) = 65,536 joint actions.  That is why a
per-step TD update can adapt within a handful of visits per state, where PPO
needs a full 512-step batch before it can move at all.

Two deliberate design choices:

1.  The agent duck-types ``compute_single_action(obs, explore=False)``, which
    is the ONLY method evaluate_policy() (examples/finetune_deterministic.py)
    calls on a policy.  So the existing evaluation stack — evaluate_per_loop,
    the per-loop tables, the seeded dedicated eval env — works on this agent
    with no changes, and tabular numbers stay comparable with PPO's.

2.  The state is (node, delay) only.  The PPO observation also carries the
    action history since last detection; including it would multiply the table
    by 2**(10*(time_limit+1)) and is a non-starter.  At time_limit=1 the loss
    is close to nil — one miss sends the tracker to the missing state, so there
    is barely any history to carry — which is a reason to prefer tl=1 here.
    Any tabular-vs-PPO comparison must state this asymmetry.

Action enumeration is delegated to build_action_map() in gym_wrapper_dqn.py
rather than the env's generate_combination_lists_new(): that method is invoked
in exactly one place repo-wide (src/core/iobt_environment.py), so
combination_dict is an empty dict for MultiLoopIoBTEnv and would fail silently.
"""
from __future__ import annotations

import numpy as np

from .gym_wrapper_dqn import build_action_map
from .iobt_loops import IOBT_NUM_NODES

# Backing grid is 4x4 for the 10-node map; actions are emitted at this width
# because that is what the env and evaluate_policy index into.
IOBT_N = 4
N_GRID = IOBT_N * IOBT_N   # 16


def action_index_size(max_sensors: int, n_nodes: int = IOBT_NUM_NODES) -> int:
    """Number of sensor subsets with 1 <= |S| <= max_sensors."""
    from math import comb
    return sum(comb(n_nodes, k) for k in range(1, max_sensors + 1))


def state_index(node, delay: int, time_limit: int,
                n_nodes: int = IOBT_NUM_NODES) -> int:
    """
    Pack (node, delay) into a row index; ``node=None`` means the missing state.

    Follows the state = node*(time_limit+1) + delay convention, but over the 10
    REAL nodes rather than the 16-cell backing grid: the 6 phantom cells are
    never occupied, so allocating rows for them would just waste the table and
    dilute learning.
    """
    if node is None:
        return n_nodes * (time_limit + 1)
    return int(node) * (time_limit + 1) + int(delay)


class TabularTDAgent:
    """
    SARSA(lambda) with replacing traces over the compact (node, delay) state.

    Set ``lam=0.0`` for plain one-step SARSA.
    """

    def __init__(self, time_limit: int, max_sensors: int, missing_state: int,
                 alpha: float = 0.1, gamma: float = 0.95, lam: float = 0.8,
                 epsilon: float = 0.1, optimistic_init: float = 0.0,
                 seed: int | None = None, n_nodes: int = IOBT_NUM_NODES):
        self.time_limit    = int(time_limit)
        self.max_sensors   = int(max_sensors)
        self.missing_state = int(missing_state)
        self.n_nodes       = int(n_nodes)

        self.alpha           = float(alpha)
        self.gamma           = float(gamma)
        self.lam             = float(lam)
        self.epsilon         = float(epsilon)
        self.optimistic_init = float(optimistic_init)

        self._rng = np.random.default_rng(seed)

        # Action set: every subset of the 10 real nodes with 1 <= |S| <= k,
        # widened to the 16-cell grid the env indexes into.
        narrow = build_action_map(self.n_nodes, self.max_sensors)
        self._actions = np.zeros((len(narrow), N_GRID), dtype=np.int64)
        for i, vec in enumerate(narrow):
            self._actions[i, :self.n_nodes] = vec

        self.n_actions = len(self._actions)
        self.n_states  = self.n_nodes * (self.time_limit + 1) + 1

        self.q      = np.full((self.n_states, self.n_actions),
                              self.optimistic_init, dtype=np.float64)
        self.traces = np.zeros_like(self.q)

    # ── action / state plumbing ────────────────────────────────────────────

    def action_vector(self, idx: int) -> np.ndarray:
        return self._actions[int(idx)].copy()

    def obs_to_state(self, obs) -> int:
        """
        Map the wrapper's (state_pos, state_time, history) tuple to a row.

        evaluate_policy signals the missing state by passing state_pos ==
        n_cells (the backing-grid size), not the missing_state sentinel.
        """
        state_pos, state_time = int(obs[0]), int(obs[1])
        if state_pos >= self.n_nodes:
            return state_index(None, 0, self.time_limit, self.n_nodes)
        return state_index(state_pos, state_time, self.time_limit, self.n_nodes)

    # ── policy ─────────────────────────────────────────────────────────────

    def select_action(self, state: int, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        # Tie-break at random so an all-equal table (optimistic init) explores
        # rather than always picking action 0.
        return int(best[0] if best.size == 1 else self._rng.choice(best))

    def compute_single_action(self, obs, explore: bool = False,
                              **_ignored) -> np.ndarray:
        """Duck-types the RLlib Algorithm method evaluate_policy() calls."""
        return self.action_vector(
            self.select_action(self.obs_to_state(obs), explore=explore)
        )

    # ── learning ───────────────────────────────────────────────────────────

    def observe(self, state: int, action: int, reward: float,
                next_state: int, next_action: int, done: bool = False) -> float:
        """One SARSA(lambda) update. Returns the TD error."""
        target = reward if done else reward + self.gamma * self.q[next_state,
                                                                  next_action]
        delta  = target - self.q[state, action]

        self.traces[state, action] = 1.0          # replacing traces
        self.q += self.alpha * delta * self.traces
        self.traces *= self.gamma * self.lam
        return float(delta)

    def end_episode(self) -> None:
        self.traces.fill(0.0)

    # ── restart support (used by the change detector) ──────────────────────

    def reset(self) -> None:
        """Cold restart: forget everything, as DAL_GLB.reset() does."""
        self.q.fill(self.optimistic_init)
        self.traces.fill(0.0)

    def snapshot(self) -> np.ndarray:
        return self.q.copy()

    def restore(self, table: np.ndarray) -> None:
        """Warm restart: reload a previously saved table."""
        table = np.asarray(table, dtype=np.float64)
        if table.shape != self.q.shape:
            raise ValueError(
                f"table shape {table.shape} does not match {self.q.shape}"
            )
        self.q = table.copy()
        self.traces.fill(0.0)


__all__ = ["TabularTDAgent", "state_index", "action_index_size",
           "N_GRID", "IOBT_N"]
