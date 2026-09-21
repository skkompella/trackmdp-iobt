"""
iobt_loops.py — loop ("lap") generation over the 10-node Camp Buckner IoBT graph.

A *loop* is a simple directed cycle of node indices: the object walks it one
node per timestep and wraps around forever.  This is the synthetic stand-in for
a real GPS trajectory, used to train and evaluate a tracker without any audio,
camera or GPS data in the loop.

The module is deliberately free of ray/torch/sklearn imports so the loop set can
be enumerated, sampled and tested cheaply.

The topology is duplicated from examples/finetune_deterministic.py rather than
imported, because importing that module pulls in ray.  tests/test_iobt_loops.py
asserts the two copies stay identical.

Note the graph is NOT symmetric: 3->6, 4->2, 4->3, 4->1 and 7->8 are one-way,
so loop direction matters and validation checks it.
"""
from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Topology (mirrors IOBT_MAP in examples/finetune_deterministic.py)
# ---------------------------------------------------------------------------

IOBT_MAP: dict[int, list[int]] = {
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

MIN_LOOP_LEN = 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_loop(loop, adjacency: dict[int, list[int]] | None = None):
    """
    Check that ``loop`` is a simple directed cycle in ``adjacency``.

    Returns (ok, message).  Mirrors validate_circle_iobt() in
    examples/finetune_deterministic.py, plus a repeated-node check.
    """
    adj = adjacency if adjacency is not None else IOBT_MAP
    n_nodes = len(adj)

    if len(loop) < MIN_LOOP_LEN:
        return False, f"loop too short ({len(loop)} < {MIN_LOOP_LEN})"

    for node in loop:
        if not (0 <= node < n_nodes):
            return False, f"node {node} out of range (0-{n_nodes - 1})"

    if len(set(loop)) != len(loop):
        return False, f"loop repeats a node: {loop}"

    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        if b not in adj.get(a, []):
            return False, (f"nodes {a} and {b} are not adjacent "
                           f"(adjacency[{a}]={adj.get(a, [])})")

    return True, "OK"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_cycles(adjacency: dict[int, list[int]] | None = None,
                     min_len: int = MIN_LOOP_LEN,
                     max_len: int | None = None) -> list[list[int]]:
    """
    Enumerate every simple directed cycle, each exactly once.

    Each cycle is returned in canonical form: rotated to start at its smallest
    node.  Uniqueness falls out of only extending a path with nodes greater
    than the start node, so a cycle is discovered just once — from its minimum.

    Direction is preserved, so a cycle and its reverse are distinct entries
    when both directions are traversable.
    """
    adj = adjacency if adjacency is not None else IOBT_MAP
    cycles: list[list[int]] = []

    def walk(start: int, node: int, path: list[int], on_path: set[int]) -> None:
        if max_len is not None and len(path) > max_len:
            return
        for nxt in adj.get(node, []):
            if nxt == start:
                if min_len <= len(path) and (max_len is None or len(path) <= max_len):
                    cycles.append(list(path))
            elif nxt > start and nxt not in on_path:
                on_path.add(nxt)
                path.append(nxt)
                walk(start, nxt, path, on_path)
                path.pop()
                on_path.discard(nxt)

    for start in sorted(adj):
        walk(start, start, [start], {start})

    return cycles


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_loops(n: int = 10,
                 seed: int = 0,
                 adjacency: dict[int, list[int]] | None = None,
                 min_len: int = MIN_LOOP_LEN,
                 max_len: int | None = None) -> list[list[int]]:
    """
    Draw ``n`` distinct loops uniformly at random from every valid cycle.

    Deterministic given ``seed``: the same seed always yields the same loops,
    in the same order, so a training run is reproducible from the seed alone.
    """
    pool = enumerate_cycles(adjacency, min_len=min_len, max_len=max_len)
    if n > len(pool):
        raise ValueError(
            f"requested {n} loops but only {len(pool)} cycles exist "
            f"(min_len={min_len}, max_len={max_len})"
        )
    return [list(loop) for loop in random.Random(seed).sample(pool, n)]


def describe_loops(loops) -> str:
    """Human-readable summary, for run logs and the loops.json sidecar."""
    lines = [f"{len(loops)} loops over {IOBT_NUM_NODES} nodes:"]
    for i, loop in enumerate(loops):
        lines.append(f"  loop {i:2d}  len={len(loop):2d}  {loop}")
    covered = sorted({n for loop in loops for n in loop})
    lines.append(f"  nodes covered: {covered}")
    missing = sorted(set(range(IOBT_NUM_NODES)) - set(covered))
    lines.append(f"  nodes never visited: {missing if missing else 'none'}")
    return "\n".join(lines)


__all__ = [
    "IOBT_MAP",
    "IOBT_NUM_NODES",
    "validate_loop",
    "enumerate_cycles",
    "sample_loops",
    "describe_loops",
]
