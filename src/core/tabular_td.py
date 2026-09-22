"""
tabular_td.py — tabular SARSA(lambda) for Track-MDP tracking problems.

Track-MDP's state is deliberately tiny — last known cell plus a staleness
counter — so the value function fits in a small table and a per-step TD update
can adapt within a handful of visits per state, where PPO needs a full 512-step
batch before it can move at all.

Two action representations, because the right one depends on the sensor budget:

  "subset"    one value per sensor SUBSET, enumerated by build_action_map.
              Exact, and fine when the budget is small:
                  10 nodes, max_sensors=2  ->  21 x   55 =  1,155 entries
                  10 nodes, max_sensors=6  ->  21 x  847 = 17,787 entries

  "factored"  one weight per CELL; Q(s, A) = sum of w[s, c] for c in A, and the
              greedy action is the highest-weighted cells with positive value,
              up to the budget.  This is linear function approximation with
              binary action features, and it is what makes a large budget
              tractable at all:
                  25 cells, max_sensors=6  ->  subset:   51 x 245,505 = 12.5M
                                               factored: 51 x      25 =  1,275

Why factored is a good fit here rather than a concession: expected immediate
reward really is additive in the activated set, because the object occupies
exactly one cell (so detection probability is the sum of per-cell occupancy
probabilities) and sensor cost is linear in the count.  Only the continuation
term is approximated.  It also matches how PPO already treats this action
space — MultiDiscrete([2]*n) is n independent Bernoulli heads — so the two
learners score cells the same way.

The one behavioural difference: factored mode can choose to activate NOTHING
when every weight is negative, which subset mode cannot (build_action_map
starts at size 1).  That is deliberate — with a per-sensor cost, declining to
activate is a legitimate action, and removing it would break the energy
trade-off the experiments are about.

IMPORTANT, measured: use lam=0.0 in factored mode.  Eligibility traces smear a
single TD error across every (state, cell) pair with a live trace, which in the
subset representation touches one entry per state but in the factored one
touches many cells at once and corrupts per-cell credit assignment.  On the 5x5
switching grid, lam=0.8 gave 4.95% accuracy and lam=0.0 gave 84.35% with
everything else identical.  Traces remain the default for subset mode, where
they help.

The agent duck-types ``compute_single_action(obs, explore=False)``, the only
method evaluate_policy() (examples/finetune_deterministic.py) calls on a policy,
so the existing evaluation stack works on it unchanged in both modes.
"""
from __future__ import annotations

import numpy as np

from .gym_wrapper_dqn import build_action_map
from .iobt_loops import IOBT_NUM_NODES

# Default backing-grid width, for the 10-node IoBT map (4x4).  Overridable per
# agent via ``grid_width`` — a 5x5 grid needs 25, and the old module constant
# silently truncated anything wider.
IOBT_N = 4
N_GRID = IOBT_N * IOBT_N   # 16

ACTION_MODES = ("subset", "factored")


def action_index_size(max_sensors: int, n_nodes: int = IOBT_NUM_NODES) -> int:
    """Number of sensor subsets with 1 <= |S| <= max_sensors."""
    from math import comb
    return sum(comb(n_nodes, k) for k in range(1, max_sensors + 1))


def state_index(node, delay: int, time_limit: int,
                n_nodes: int = IOBT_NUM_NODES) -> int:
    """
    Pack (node, delay) into a row index; ``node=None`` means the missing state.

    Follows the state = node*(time_limit+1) + delay convention, but over the
    REAL cells rather than any larger backing grid, so no rows are wasted on
    cells the object can never occupy.
    """
    if node is None:
        return n_nodes * (time_limit + 1)
    return int(node) * (time_limit + 1) + int(delay)


class TabularTDAgent:
    """SARSA(lambda) with replacing traces over the compact (cell, delay) state."""

    def __init__(self, time_limit: int, max_sensors: int, missing_state: int,
                 alpha: float = 0.1, gamma: float = 0.95, lam: float = 0.8,
                 epsilon: float = 0.1, optimistic_init: float = 0.0,
                 seed: int | None = None, n_nodes: int = IOBT_NUM_NODES,
                 grid_width: int | None = None,
                 action_mode: str = "subset"):
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")

        self.time_limit    = int(time_limit)
        self.max_sensors   = int(max_sensors)
        self.missing_state = int(missing_state)
        self.n_nodes       = int(n_nodes)
        self.grid_width    = int(grid_width if grid_width is not None else N_GRID)
        self.action_mode   = action_mode

        if self.grid_width < self.n_nodes:
            raise ValueError(
                f"grid_width {self.grid_width} is smaller than n_nodes "
                f"{self.n_nodes}; action vectors would be truncated")

        self.alpha           = float(alpha)
        self.gamma           = float(gamma)
        self.lam             = float(lam)
        self.epsilon         = float(epsilon)
        self.optimistic_init = float(optimistic_init)

        self._rng     = np.random.default_rng(seed)
        self.n_states = self.n_nodes * (self.time_limit + 1) + 1

        if action_mode == "subset":
            narrow = build_action_map(self.n_nodes, self.max_sensors)
            self._actions = np.zeros((len(narrow), self.grid_width),
                                     dtype=np.int64)
            for i, vec in enumerate(narrow):
                self._actions[i, :self.n_nodes] = vec
            self.n_actions = len(self._actions)
            cols = self.n_actions
        else:
            self._actions = None
            self.n_actions = None          # not enumerable
            # One weight per cell PLUS a per-state bias in the last column.
            # Without the bias, Q(s, empty) is identically 0 -- the empty action
            # has no features -- so "activate nothing" has a hardcoded value
            # that can never be learned, while every real action's value can go
            # negative.  Measured consequence: weights ran away to -14, the
            # agent activated nothing, stopped bumping traces, and sat at 0.00%
            # accuracy permanently.  The bias makes doing nothing learnable.
            cols = self.n_nodes + 1

        self.q      = np.full((self.n_states, cols), self.optimistic_init,
                              dtype=np.float64)
        self.traces = np.zeros_like(self.q)

    # ── action plumbing ────────────────────────────────────────────────────

    def action_vector(self, action) -> np.ndarray:
        """
        Binary grid-width vector for an action.

        ``action`` is an int index in subset mode, or an iterable of cell
        indices in factored mode.
        """
        if self.action_mode == "subset":
            return self._actions[int(action)].copy()
        vec = np.zeros(self.grid_width, dtype=np.int64)
        cells = np.asarray(list(action), dtype=int)
        if cells.size:
            vec[cells] = 1
        return vec

    def obs_to_state(self, obs) -> int:
        """
        Map the wrapper's (state_pos, state_time, history) tuple to a row.

        evaluate_policy signals the missing state by passing state_pos equal to
        the backing-grid size, not the missing_state sentinel.
        """
        state_pos, state_time = int(obs[0]), int(obs[1])
        if state_pos >= self.n_nodes:
            return state_index(None, 0, self.time_limit, self.n_nodes)
        return state_index(state_pos, state_time, self.time_limit, self.n_nodes)

    def action_value(self, state: int, action) -> float:
        """Q(s, a) — a table lookup, or the sum of the activated cells' weights."""
        if self.action_mode == "subset":
            return float(self.q[state, int(action)])
        cells = np.asarray(list(action), dtype=int)
        value = float(self.q[state, -1])          # always-on bias
        if cells.size:
            value += float(self.q[state, cells].sum())
        return value

    # ── policy ─────────────────────────────────────────────────────────────

    def _greedy(self, state: int):
        if self.action_mode == "subset":
            row  = self.q[state]
            best = np.flatnonzero(row == row.max())
            # Random tie-break so an all-equal table (optimistic init) explores
            # rather than always picking action 0.
            return int(best[0] if best.size == 1 else self._rng.choice(best))

        row = self.q[state, :self.n_nodes]   # the bias is common to all actions
        # Random tie-break. Without it, an all-equal row (which is exactly what
        # optimistic initialisation produces) makes argsort return cells
        # 0,1,2,... every time, so the agent systematically watches the
        # lowest-indexed cells and explores the rest only through epsilon.
        mixer = self._rng.random(row.size)
        order = np.lexsort((mixer, -row))[:self.max_sensors]
        # Only activate cells whose marginal value is positive; declining to
        # activate is a legitimate action once sensors cost something.
        return tuple(int(c) for c in order if row[c] > 0.0)

    def _random_action(self, state: int):
        if self.action_mode == "subset":
            return int(self._rng.integers(self.n_actions))
        k = int(self._rng.integers(1, self.max_sensors + 1))
        return tuple(int(c) for c in
                     self._rng.choice(self.n_nodes, size=k, replace=False))

    def select_action(self, state: int, explore: bool = True):
        if explore and self._rng.random() < self.epsilon:
            return self._random_action(state)
        return self._greedy(state)

    def compute_single_action(self, obs, explore: bool = False,
                              **_ignored) -> np.ndarray:
        """Duck-types the RLlib Algorithm method evaluate_policy() calls."""
        return self.action_vector(
            self.select_action(self.obs_to_state(obs), explore=explore))

    # ── learning ───────────────────────────────────────────────────────────

    def _bump_traces(self, state: int, action) -> int:
        """Mark the active features and return how many there are."""
        if self.action_mode == "subset":
            self.traces[state, int(action)] = 1.0
            return 1
        self.traces[state, -1] = 1.0              # bias is always active
        cells = np.asarray(list(action), dtype=int)
        if cells.size:
            self.traces[state, cells] = 1.0
        return int(cells.size) + 1

    def observe(self, state: int, action, reward: float,
                next_state: int, next_action, done: bool = False) -> float:
        """One SARSA(lambda) update. Returns the TD error."""
        target = (reward if done else
                  reward + self.gamma * self.action_value(next_state, next_action))
        delta  = target - self.action_value(state, action)

        n_active = self._bump_traces(state, action)
        # Normalise the step by the number of active features.  Q(s,A) is the
        # SUM over |A| weights, so adding alpha*delta to each would move Q by
        # alpha*delta*|A| — an |A|-fold overshoot that diverges outright at
        # max_sensors=6 (measured: the agent collapsed to 0% accuracy).  This is
        # the standard normalised-LMS step for linear TD; in subset mode
        # n_active is 1 and the behaviour is unchanged.
        step = self.alpha / max(1, n_active)
        self.q += step * delta * self.traces
        self.traces *= self.gamma * self.lam
        return float(delta)

    def end_episode(self) -> None:
        self.traces.fill(0.0)

    # ── restart support (used by the change detector) ──────────────────────

    def reset(self) -> None:
        """Cold restart: forget everything."""
        self.q.fill(self.optimistic_init)
        self.traces.fill(0.0)

    def snapshot(self) -> np.ndarray:
        return self.q.copy()

    def restore(self, table: np.ndarray) -> None:
        """Warm restart: reload a previously saved table."""
        table = np.asarray(table, dtype=np.float64)
        if table.shape != self.q.shape:
            raise ValueError(
                f"table shape {table.shape} does not match {self.q.shape}")
        self.q = table.copy()
        self.traces.fill(0.0)


__all__ = ["TabularTDAgent", "state_index", "action_index_size",
           "ACTION_MODES", "N_GRID", "IOBT_N"]
