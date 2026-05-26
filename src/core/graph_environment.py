"""
IoBT Graph Environment for Object Tracking
============================================
Drop-in replacement for grid_env that operates on a fixed 10-node graph
instead of an N×N grid.

Only four things change from the original grid_env:

    1.  IOBT_ADJACENCY / NUM_IOBT_NODES  — the graph topology (module-level)
    2.  object_transition_matrix_iobt()   — replaces object_transition_matrix_random_3x3
    3.  build_iobt_cum_probs()            — replaces the ad-hoc formula in demo_environment
    4.  check_valid_sensor_iobt()  \
        get_valid_q_indices_iobt()  }     — replace the grid-radius sensor logic
        realign_obj_iobt()         /

Everything else (object_move, get_reward_next_state, reset_object_state,
get_valid_q_indices_dict, combination dicts, rewards) is unchanged and
inherited directly from grid_env.

Node numbering is 0-indexed internally; the spec below uses 1-indexed labels.

Spec (1-indexed):
    1  -> 2, 8, 5
    2  -> 1, 8, 7, 3
    3  -> 2, 7, 4
    4  -> 3, 6, 7
    5  -> 6, 3, 4, 1, 2
    6  -> 5, 4
    7  -> 2, 3, 9
    8  -> 10, 9, 1, 2
    9  -> 10, 7
    10 -> 9, 8
"""

import numpy as np
import random as rnd
from collections import deque

# Re-use everything from the original environment that we are not replacing
try:
    from .environment import grid_env          # when imported as src.core.graph_environment
except ImportError:
    from environment import grid_env           # when run directly as a script


# ===========================================================================
# Module-level graph definition  (0-indexed)
# ===========================================================================

# Each key is a node index 0-9; values are lists of reachable neighbour indices.
IOBT_ADJACENCY = {
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

NUM_IOBT_NODES = len(IOBT_ADJACENCY)   # 10


# ===========================================================================
# 1.  Cumulative probability builder
# ===========================================================================

def build_iobt_cum_probs(num_trans, terminal_prob=0.005, stay_prob=0.15):
    """
    Build the cumulative probability list for the IoBT graph environment.

    The original grid formula divides movement probability equally across
    (num_trans - 1) slots.  That still works here, but is worth making
    explicit so it is easy to tune.

    Probability budget
    ------------------
    terminal_prob : P(object disappears / enters terminal state this step)
    stay_prob     : P(object stays at current node)  — fills the *last* slot
    movement_prob : remainder, split equally across the first (num_trans-1) slots

    The returned list has exactly `num_trans` entries and is strictly
    increasing.  object_move() uses it as an inverse-CDF lookup:

        u ~ Uniform(0, 1)
        slot = number of thresholds that u exceeds
        if slot < num_trans  ->  go to obj_trans_matrix[node][slot]
        else                 ->  go to terminal state  (N*N)

    Parameters
    ----------
    num_trans    : int   — number of transition slots per node (same as
                          grid_env's num_trans)
    terminal_prob: float — probability of entering the terminal state
    stay_prob    : float — probability of staying put

    Returns
    -------
    list[float] of length num_trans
    """
    assert num_trans >= 2, "Need at least 2 transition slots"
    assert terminal_prob + stay_prob < 1.0, \
        "terminal_prob + stay_prob must be < 1"

    move_prob = 1.0 - terminal_prob - stay_prob
    # Spread movement probability uniformly across the first (num_trans-1) slots
    step = move_prob / (num_trans - 1)
    cum = [step * i for i in range(1, num_trans)]   # length = num_trans - 1
    cum.append(cum[-1] + stay_prob)                  # last slot = stay
    # cum[-1] < 1.0  =>  residual = terminal_prob falls beyond the list
    return cum


# ===========================================================================
# 2.  Transition matrix
# ===========================================================================

def object_transition_matrix_iobt(num_trans):
    """
    Build the IoBT object transition matrix.

    Replaces object_transition_matrix_random_3x3 from grid_env.

    For each of the NUM_IOBT_NODES active nodes, `num_trans` destinations
    are sampled uniformly WITH REPLACEMENT from:

        { node itself }  ∪  { all graph neighbours of node }

    Including the node itself means the "stay" slot in prob_list_cum always
    has a valid destination.  The last row (index NUM_IOBT_NODES) is the
    terminal row — it maps entirely to NUM_IOBT_NODES, matching the original
    convention of row N*N being the absorbing terminal state.

    Parameters
    ----------
    num_trans : int  — number of slots per row (same num_trans passed to grid_env)

    Returns
    -------
    list of (NUM_IOBT_NODES + 1) rows, each a list of num_trans int destinations
    """
    matrix = []
    for node in range(NUM_IOBT_NODES):
        # Candidate destinations: neighbours PLUS the node itself (for "stay")
        candidates = IOBT_ADJACENCY[node] + [node]
        destinations = list(
            np.random.choice(candidates, size=num_trans, replace=True)
        )
        matrix.append(destinations)

    # Terminal / absorbing row — always stays at terminal state
    matrix.append([NUM_IOBT_NODES] * num_trans)
    return matrix


# ===========================================================================
# 3.  Graph-distance sensor logic
#     Replaces check_valid_sensor, get_valid_q_indices, and realign_obj
# ===========================================================================

def _bfs_reachable(start_node, max_hops, adjacency):
    """
    BFS from start_node up to max_hops steps.
    Returns an ordered list of reachable node indices (including start_node).
    """
    visited = {start_node}
    frontier = {start_node}
    for _ in range(max_hops):
        next_frontier = set()
        for n in frontier:
            for nb in adjacency[n]:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


def check_valid_sensor_iobt(base_node, time_value, adjacency=None):
    """
    Graph replacement for check_valid_sensor.

    Instead of a spatial radius, the sensor window at time_delay t covers
    all nodes reachable within (t + 1) hops from base_node.  The window
    size is always NUM_IOBT_NODES (the full graph) so the action vector
    length is constant, matching generate_combination_lists_new's behaviour.

    Returns a binary list of length NUM_IOBT_NODES:
        valid_ind[i] = 1  if node i is within (time_value + 1) hops of base_node
        valid_ind[i] = 0  otherwise
    """
    if adjacency is None:
        adjacency = IOBT_ADJACENCY

    max_hops = time_value + 1          # delay 0 -> 1 hop, delay 1 -> 2 hops, ...
    reachable = set(_bfs_reachable(base_node, max_hops, adjacency))

    return [1 if node in reachable else 0 for node in range(NUM_IOBT_NODES)]


def get_valid_q_indices_iobt(state, time_value, time_limit, adjacency=None):
    """
    Graph replacement for get_valid_q_indices.

    Decodes the base node from the encoded state, then returns the binary
    reachability mask from check_valid_sensor_iobt.

    State encoding (unchanged from grid_env):
        state = base_node * (time_limit + 1) + time_delay_counter
    """
    base_node = state // (time_limit + 1)
    base_node = min(base_node, NUM_IOBT_NODES - 1)   # safety clamp
    return np.array(
        check_valid_sensor_iobt(base_node, time_value, adjacency),
        dtype=int,
    )


def realign_obj_iobt(object_node, current_state, time_delay, time_limit,
                     adjacency=None):
    """
    Graph replacement for realign_obj.

    In the original code, realign_obj:
      - checks whether the object is within the spatial sensing radius
      - returns its *index* within the sensor window vector

    Here, the sensor window for belief node b at delay t is the sorted
    list of nodes reachable within (t+1) hops.  The object's slot index
    is its position in that sorted list (or 0 / out-of-window).

    Returns
    -------
    obj_rel_pos : int  — index of object_node in the window list
    obj_in_window : int — 1 if object is reachable, 0 otherwise
    """
    if adjacency is None:
        adjacency = IOBT_ADJACENCY

    base_node = current_state // (time_limit + 1)
    base_node = min(base_node, NUM_IOBT_NODES - 1)

    max_hops  = time_delay + 1
    reachable = _bfs_reachable(base_node, max_hops, adjacency)   # sorted list

    if object_node in reachable:
        obj_rel_pos = reachable.index(object_node)
        return obj_rel_pos, 1
    else:
        return 0, 0


# ===========================================================================
# 4.  iobt_env — subclass of grid_env that swaps in the new methods
# ===========================================================================

class iobt_env(grid_env):
    """
    IoBT tracking environment.

    Inherits grid_env and overrides only the four things that depend on
    grid geometry.  All reward logic, state encoding, combination dicts,
    and the training loop interface are unchanged.

    The 'N' parameter is kept for API compatibility with grid_env and
    learning_grid_sarsa_0, but internally the environment operates on
    the 10-node graph defined by IOBT_ADJACENCY.  Pass N=10 to keep
    downstream code consistent (N*N = 100 >> 10, so the terminal state
    sentinel N*N is safely above all valid node indices).
    """

    # --- 2. Transition matrix ---

    def object_transition_matrix_random_3x3(self):
        """
        Override: build the IoBT graph transition table instead of the
        3x3 grid neighbourhood table.

        The method name is kept identical so grid_env.__init__ calls this
        version transparently via self.object_transition_matrix_random_3x3().
        """
        return object_transition_matrix_iobt(self.num_trans)

    # --- 3a. Sensor validity mask ---

    def check_valid_sensor(self, time_value, b_l, b_r, b_d, b_u):
        """
        Override: the grid boundary parameters (b_l, b_r, b_d, b_u) are
        ignored.  The base_node is recovered later in get_valid_q_indices.
        Returns a fixed-length mask over NUM_IOBT_NODES nodes.

        Because get_valid_q_indices_iobt needs the base_node (not just the
        boundary box), the override is done at the get_valid_q_indices level
        instead (see below).  This method is left as a no-op fallback.
        """
        # Fallback: all sensors valid (should not be reached in normal operation)
        return [1] * NUM_IOBT_NODES

    def get_valid_q_indices(self, state, time_value):
        """
        Override: use graph BFS reachability instead of grid radius.
        """
        return get_valid_q_indices_iobt(
            state, time_value, self.time_limit, IOBT_ADJACENCY
        )

    def get_valid_q_indices_dict(self):
        """
        Override: iterate only over the NUM_IOBT_NODES real node states
        instead of the N^2*(time_limit+1) phantom grid-cell states that
        grid_env.__init__ builds.  The parent builds entries for states
        0..N^2*(time_limit+1) — with N=10 that is 200 phantom states, most
        of which map to nonsense node indices clamped to node 9.
        Here we build exactly NUM_IOBT_NODES*(time_limit+1) entries, one
        per real (node, delay) pair.
        """
        valid_dict = {}
        for t in range(self.time_limit + 1):
            valid_dict[t] = {}
            for s in range(NUM_IOBT_NODES * (self.time_limit + 1)):
                valid_dict[t][s] = self.get_valid_q_indices(s, t)
        return valid_dict

    # --- 3b. Object relocalisation within sensor window ---

    def realign_obj(self, object_pos, current_state, time_delay):
        """
        Override: locate the object within the graph-BFS sensor window.
        """
        return realign_obj_iobt(
            object_pos, current_state, time_delay,
            self.time_limit, IOBT_ADJACENCY
        )

    # --- Housekeeping: keep only valid nodes for reset / terminal check ---

    def reset_object_state(self):
        """Place the object uniformly on one of the 10 active nodes."""
        self.object_pos = rnd.randint(0, NUM_IOBT_NODES - 1)

    def object_move(self):
        """
        Unchanged logic from grid_env, but terminal state is NUM_IOBT_NODES
        (not N*N).  Override to use the correct sentinel.
        """
        states_mov = self.obj_trans_matrix[self.object_pos]
        u = np.random.uniform(0, 1)
        slot = int(np.sum(self.prob_list_cum <= u))
        if slot < self.num_trans:
            self.object_pos = states_mov[slot]
        else:
            self.object_pos = NUM_IOBT_NODES   # terminal
        return 0

    def get_reward_next_state(self, current_state, current_action, time_delay):
        """
        Thin override: replaces the (2t+3)^2 sensor-window size with
        NUM_IOBT_NODES, and uses NUM_IOBT_NODES as the terminal sentinel.

        Everything else (reward formula, state encoding, time_delay logic)
        is identical to the original.
        """
        obj_position = self.object_pos
        obj_found = 0

        no_of_time_sensors = NUM_IOBT_NODES   # fixed — whole graph is the window

        if current_state != self.missing_state:
            # Clip action to the graph window size
            current_action_clip = current_action[-no_of_time_sensors:]
            # Zero out sensors that are outside BFS reach at this delay
            current_action_sensors = np.multiply(
                current_action_clip,
                self.valid_q_indices_dict[time_delay][current_state]
            )
            obj_rel_pos, obj_in_window = self.realign_obj(
                self.object_pos, current_state, time_delay
            )
        else:
            # Missing state: the agent's actual action is used so the agent
            # can learn to deploy sensors selectively even when lost.
            # obj is guaranteed findable (obj_in_window=1, slot 0) so the
            # agent just needs at least one sensor active to re-acquire.
            current_action_clip = np.array(current_action[-no_of_time_sensors:], dtype=int)
            current_action_sensors = current_action_clip   # all slots valid when lost
            obj_rel_pos, obj_in_window = 0, 1              # object always in window

        if obj_in_window and current_action_sensors[int(obj_rel_pos)]:
            obj_found = 1
            next_state = obj_position * (self.time_limit + 1)
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
        no_sensor_on = np.sum(current_action_sensors)

        # Unified reward formula — sensor cost reflects actual activations
        # in both normal and missing states.  The original charged NUM_IOBT_NODES
        # sensors flat in the missing state, which gave the agent no incentive
        # to be selective and caused runaway negative rewards.
        reward = (obj_found * (self.tracking_rew if current_state != self.missing_state
                               else self.tracking_rew_missing)
                  + (1 - obj_found) * (self.tracking_miss_rew if current_state != self.missing_state
                                       else self.tracking_miss_rew_missing)
                  + no_sensor_on * self.sensor_rew)

        terminal_st_obj = 1 if self.object_pos == NUM_IOBT_NODES else 0
        return reward, next_state, terminal_st_obj, time_delay_sense, obj_found


# ===========================================================================
# 5.  learning_iobt_sarsa — mirrors learning_grid_sarsa_0
# ===========================================================================

class learning_iobt_sarsa:
    """
    Wrapper that mirrors learning_grid_sarsa_0 but uses iobt_env.

    Pass N=10 to keep the missing_state sentinel (N*N+1 = 101) safely above
    all valid node indices (0-9) and the terminal sentinel (10).
    """

    def __init__(self, run_number, num_trans, state_trans_cum_prob,
                 max_sensors, max_sensors_null, time_limit, time_limit_max,
                 N=10):
        self.run_number     = run_number
        self.N              = N
        self.num_trans      = num_trans
        self.prob_list_cum  = state_trans_cum_prob
        self.time_limit     = time_limit
        self.time_limit_max = time_limit_max

        # Use NUM_IOBT_NODES as the basis for the missing_state sentinel,
        # not N*N.  With N=10, N*N=100 creates 100*(time_limit_max+1)=200
        # phantom states in get_valid_q_indices_dict — most map to nonexistent
        # nodes, corrupting the sensor masks.  The sentinel just needs to be
        # strictly greater than the highest real encoded state, which is:
        #   (NUM_IOBT_NODES-1)*(time_limit_max+1) + time_limit_max
        self.missing_state = NUM_IOBT_NODES * (time_limit_max + 1) + 1

        self.graph_env = iobt_env(
            N, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null,
            self.missing_state, time_limit,
        )

        self.exploration_epsilon = 0.1
        self.total_actions      = self.graph_env.action_space_size
        self.total_actions_null = self.graph_env.action_space_size_null

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

        self.episode_start      = 0
        self.file_save_directory = "./iobt_qsave"
        self.save_directory     = None

    def update_time_limit(self, new_time_limit):
        self.time_limit = new_time_limit
        self.graph_env.time_limit = new_time_limit
        self.graph_env.valid_q_indices_dict = \
            self.graph_env.get_valid_q_indices_dict()


# ===========================================================================
# 6.  Demo
# ===========================================================================

def demo_iobt():
    print("=" * 60)
    print("  IoBT GRAPH ENVIRONMENT DEMO")
    print("=" * 60)

    print("\nGraph topology (1-indexed):")
    for node, neighbours in IOBT_ADJACENCY.items():
        print(f"  Node {node+1:2d}  ->  {[n+1 for n in neighbours]}")

    num_trans     = 4
    terminal_prob = 0.005
    stay_prob     = 0.15
    cum_probs     = build_iobt_cum_probs(num_trans, terminal_prob, stay_prob)
    print(f"\nCumulative probs: {[round(p, 4) for p in cum_probs]}")
    print(f"  Slot 0..{num_trans-2}: movement  "
          f"(P ≈ {cum_probs[0]:.3f} each)")
    print(f"  Slot {num_trans-1}    : stay      "
          f"(P ≈ {stay_prob:.3f})")
    print(f"  Above {cum_probs[-1]:.3f} : terminal  "
          f"(P ≈ {terminal_prob:.3f})")

    qobj = learning_iobt_sarsa(
        run_number=9999,
        num_trans=num_trans,
        state_trans_cum_prob=cum_probs,
        max_sensors=4,
        max_sensors_null=4,
        time_limit=1,
        time_limit_max=1,
    )

    print(f"\nInitial object position: node {qobj.graph_env.object_pos + 1}")
    print("-" * 40)

    for step in range(10):
        pos = qobj.graph_env.object_pos
        if pos == NUM_IOBT_NODES:
            print(f"Step {step+1:2d}: TERMINAL — resetting")
            qobj.graph_env.reset_object_state()
            continue
        neighbours = [n + 1 for n in IOBT_ADJACENCY[pos]]
        print(f"Step {step+1:2d}: node {pos+1:2d}  "
              f"(neighbours: {neighbours})")
        qobj.graph_env.object_move()

    print("\nDemo complete.")


if __name__ == "__main__":
    demo_iobt()