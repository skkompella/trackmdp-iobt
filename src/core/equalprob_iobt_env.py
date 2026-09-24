"""
equalprob_iobt_env.py — the IoBT analogue of the grid's matrix C.

The switching-grid experiment found that warm restart wins precisely when the
base model is better than what the learner reaches on its own, and that such a
base model exists when the object's motion is EQUAL-PROBABILITY: on the grid,
matrix C (stay or any neighbour, all equally likely) produced an agent scoring
99% under every regime, because with enough sensors "watch every possible
destination" is simultaneously optimal everywhere.

The IoBT base model used until now was trained on the pooled LOOPS, which is the
analogue of training the grid's prior on matrices A and B rather than on C. It
scored 70.35% -- worse than what the learner reached unaided -- so restoring it
was a downgrade and warm restart lost.

This env supplies the missing piece: the object performs an equal-probability
random walk over IOBT_MAP, staying put or moving to any adjacent node with equal
probability. It is the same movement model run 241 was trained on, which scored
99.00% on all ten loops at max_sensors=6.

Node degrees in IOBT_MAP are 2 to 5, so covering every destination from the
worst node (node 4, degree 5) needs 6 sensors; that is why the sensor budget
decides whether a universal base model can exist here at all.
"""
from __future__ import annotations

from .iobt_loops import IOBT_MAP, IOBT_NUM_NODES
from .multiloop_iobt_env import MultiLoopIoBTEnv

# A valid cycle, passed only to satisfy the parent's validation. Movement is
# overridden below, so the loop is never walked.
_PLACEHOLDER_LOOP = [0, 4, 3, 6, 2, 1]


class EqualProbIoBTEnv(MultiLoopIoBTEnv):
    """MultiLoopIoBTEnv whose object random-walks IOBT_MAP instead of a loop."""

    def __init__(self, max_sensors, max_sensors_null, missing_state, time_limit,
                 seed=None, no_moore_constraint=True, reward_preset="grid",
                 sensor_rew=None):
        super().__init__(max_sensors, max_sensors_null, missing_state,
                         time_limit, loops=[_PLACEHOLDER_LOOP], seed=seed,
                         no_moore_constraint=no_moore_constraint,
                         reward_preset=reward_preset, sensor_rew=sensor_rew)

    # ── movement: equal probability over stay + neighbours ─────────────────

    def reset_object_state(self):
        self._loop_idx = 0
        self._pos_idx = 0
        self.object_pos = int(self._rng.integers(IOBT_NUM_NODES))

    def object_move(self):
        options = list(IOBT_MAP[self.object_pos]) + [self.object_pos]
        self.object_pos = int(self._rng.choice(options))
        return 0

    # force_matrix/force_loop are meaningless here; keep them harmless so the
    # env can be dropped into the existing schedule machinery.
    def force_loop(self, idx):
        self._forced = None


def successor_counts() -> dict[int, int]:
    """How many destinations each node has, including staying put."""
    return {n: len(IOBT_MAP[n]) + 1 for n in range(IOBT_NUM_NODES)}


__all__ = ["EqualProbIoBTEnv", "successor_counts"]
