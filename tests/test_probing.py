"""Tests for src/core/probing.py — coverage probes for transition streams."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.probing import CoverageProber  # noqa: E402
from src.core.transition_detector import row_supports  # noqa: E402

S = row_supports("grid", 3, 3)


def test_rho_zero_never_due():
    p = CoverageProber(S, 5, 0.0)
    for _ in range(100):
        p.record(4, [4])
    assert not p.due(4)


def test_due_when_a_cell_falls_below_floor():
    p = CoverageProber(S, 5, 0.5)
    for _ in range(4):
        p.record(4, [4, 5])
    assert p.due(4)
    assert p.probe_cells(4) == [1, 3, 4, 5, 7]
    assert p.n_probes == 1


def test_floor_rounds_down():
    p = CoverageProber(S, 5, 0.1)
    for _ in range(9):
        p.record(4, [4])
    assert not p.due(4)          # floor(0.9) == 0


def test_budget_smaller_than_support_picks_least_sampled():
    p = CoverageProber(S, 3, 0.5)
    for _ in range(4):
        p.record(4, [4, 5])
    assert p.probe_cells(4) == [1, 3, 7]          # ties broken by cell index


def test_not_due_when_all_covered_and_reset():
    p = CoverageProber(S, 5, 0.5)
    for _ in range(10):
        p.record(4, list(S[4]))
    assert not p.due(4)
    p.probe_cells(4)
    p.reset()
    assert p.n_probes == 0
    for _ in range(4):
        p.record(4, [4])
    assert p.due(4)              # counts were cleared too


def test_iobt_node4_probe_respects_budget():
    p = CoverageProber(row_supports("iobt"), 5, 0.5)
    for _ in range(4):
        p.record(4, [4])
    cells = p.probe_cells(4)
    assert len(cells) == 5 and 4 not in cells


def test_off_support_watched_cells_ignored():
    p = CoverageProber(S, 5, 0.5)
    for _ in range(4):
        p.record(4, [4, 8])      # 8 is not in S(4)
    assert p.due(4)
