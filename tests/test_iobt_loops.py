"""
Tests for src/core/iobt_loops.py — cycle enumeration and seeded loop sampling
over the 10-node Camp Buckner IoBT topology.

These must run without ray/torch: iobt_loops is deliberately dependency-free so
the loop set can be inspected and tested cheaply.
"""
import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.iobt_loops import (  # noqa: E402
    IOBT_MAP,
    IOBT_NUM_NODES,
    enumerate_cycles,
    sample_loops,
    validate_loop,
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# Topology consistency
# ---------------------------------------------------------------------------

def _iobt_map_from_finetune_script():
    """
    Pull IOBT_MAP out of examples/finetune_deterministic.py by parsing the
    source, so we don't import that module (it pulls in ray).
    """
    path = os.path.join(PROJECT_ROOT, "examples", "finetune_deterministic.py")
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IOBT_MAP":
                    return ast.literal_eval(node.value)
    raise AssertionError("IOBT_MAP not found in finetune_deterministic.py")


def test_topology_matches_finetune_script():
    """
    iobt_loops keeps its own copy of the topology so it stays ray-free.  This
    test is what makes that duplication safe: if either copy is edited without
    the other, this fails.
    """
    assert IOBT_MAP == _iobt_map_from_finetune_script()


def test_num_nodes_matches_map():
    assert IOBT_NUM_NODES == len(IOBT_MAP) == 10


def test_topology_has_directed_only_edges():
    """
    The Camp Buckner graph is not symmetric — a handful of edges are one-way.
    Loop validation has to respect direction, so pin the asymmetry down here.
    """
    directed_only = {
        (a, b)
        for a, nbrs in IOBT_MAP.items()
        for b in nbrs
        if a not in IOBT_MAP[b]
    }
    assert directed_only == {(3, 6), (4, 2), (4, 3), (4, 1), (7, 8)}


# ---------------------------------------------------------------------------
# validate_loop
# ---------------------------------------------------------------------------

def test_validate_accepts_known_good_loop():
    # The loop hardcoded as IOBT_CIRCLE_PATH in finetune_deterministic.py.
    ok, msg = validate_loop([0, 4, 3, 6, 2, 1])
    assert ok, msg


def test_validate_rejects_out_of_range_node():
    ok, msg = validate_loop([0, 4, 99])
    assert not ok
    assert "out of range" in msg


def test_validate_rejects_nonadjacent_hop():
    # 0 -> 5 is not an edge: IOBT_MAP[0] == [1, 7, 4]
    ok, msg = validate_loop([0, 5, 4])
    assert not ok
    assert "not adjacent" in msg


def test_validate_rejects_unclosed_loop():
    # 2 -> 0 is not an edge, so this path does not close back to the start.
    ok, msg = validate_loop([0, 1, 2])
    assert not ok


def test_validate_respects_edge_direction():
    """7 -> 8 is a real edge; 8 -> 7 is not.  Direction must matter."""
    assert 8 in IOBT_MAP[7]
    assert 7 not in IOBT_MAP[8]
    ok_forward, _ = validate_loop([7, 8, 9])   # 7->8, 8->9, 9->7 all valid
    assert ok_forward
    ok_reverse, _ = validate_loop([7, 9, 8])   # 8->7 invalid
    assert not ok_reverse


def test_validate_rejects_repeated_node():
    ok, msg = validate_loop([0, 1, 0, 4])
    assert not ok
    assert "repeat" in msg.lower()


def test_validate_rejects_too_short():
    ok, msg = validate_loop([0, 1])
    assert not ok


# ---------------------------------------------------------------------------
# enumerate_cycles
# ---------------------------------------------------------------------------

def test_enumerate_finds_expected_count():
    """
    Guards the enumerator against silent regressions.  81 simple directed
    cycles of length >= 3 exist in this topology.
    """
    assert len(enumerate_cycles()) == 81


def test_every_enumerated_cycle_is_valid():
    for cycle in enumerate_cycles():
        ok, msg = validate_loop(cycle)
        assert ok, f"{cycle}: {msg}"


def test_enumerated_cycles_are_unique_up_to_rotation():
    seen = set()
    for cycle in enumerate_cycles():
        # canonical rotation: start at the minimum node
        i = cycle.index(min(cycle))
        key = tuple(cycle[i:] + cycle[:i])
        assert key not in seen, f"duplicate cycle {cycle}"
        seen.add(key)


def test_enumerated_cycles_start_at_their_minimum():
    for cycle in enumerate_cycles():
        assert cycle[0] == min(cycle)


def test_length_filters_are_respected():
    cycles = enumerate_cycles(min_len=4, max_len=6)
    assert cycles
    assert all(4 <= len(c) <= 6 for c in cycles)


def test_every_node_appears_in_some_cycle():
    covered = {n for c in enumerate_cycles() for n in c}
    assert covered == set(range(IOBT_NUM_NODES))


# ---------------------------------------------------------------------------
# sample_loops
# ---------------------------------------------------------------------------

def test_sample_is_deterministic_under_seed():
    assert sample_loops(10, seed=1234) == sample_loops(10, seed=1234)


def test_sample_differs_across_seeds():
    assert sample_loops(10, seed=1) != sample_loops(10, seed=2)


def test_sample_returns_requested_count():
    assert len(sample_loops(10, seed=7)) == 10


def test_sampled_loops_are_distinct():
    loops = sample_loops(10, seed=7)
    keys = {tuple(l) for l in loops}
    assert len(keys) == 10


def test_sampled_loops_are_all_valid():
    for loop in sample_loops(10, seed=7):
        ok, msg = validate_loop(loop)
        assert ok, f"{loop}: {msg}"


def test_sample_rejects_more_than_available():
    with pytest.raises(ValueError):
        sample_loops(1000, seed=0)


def test_sample_respects_length_filters():
    loops = sample_loops(10, seed=3, min_len=4, max_len=8)
    assert all(4 <= len(l) <= 8 for l in loops)
