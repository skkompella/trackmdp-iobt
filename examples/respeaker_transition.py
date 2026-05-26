#!/usr/bin/env python3
"""
respeaker_transition.py — Extract an empirical transition matrix and dominant
cycle from a single respeaker power recording file.

Input
-----
One respeaker_power_*.npy file  (shape: (T, 6)  float32)
    ground-truth cell at timestep t = argmax(row[t])

Default input file: respeaker_power_20260416_152832.npy  (in <project>/data/)

Outputs
-------
transition_matrix.npy       float64 (6, 6) — empirical P(j|i) excluding self-loops
transition_matrix_raw.npy   float64 (6, 6) — raw counts
dominant_cycle.json         {"cycle": [0,1,2,5,4,3], "coverage": 0.81, ...}
transition_report.txt       human-readable summary with usage instructions

Usage
-----
    python examples/respeaker_transition.py
    python examples/respeaker_transition.py --file data/respeaker_power_20260416_153007.npy
    python examples/respeaker_transition.py --min-dwell 2 --min-prob 0.1 --no-plot

Then plug the cycle directly into finetune_deterministic.py:
    python examples/finetune_deterministic.py --run 201 --circle 0,1,2,5,4,3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

N_CELLS  = 6     # 2×3 grid
NROWS, NCOLS = 2, 3

NODE_IDS            = [11, 12, 13, 14, 15, 16]   # cell i = NODE_IDS[i]
_IOBT_DATA          = os.path.join(project_root, "iobt_data")
DEFAULT_SESSION     = "20260416_154037"
DEFAULT_NODES_TXT   = os.path.join(_IOBT_DATA, "node_positions.txt")
DEFAULT_PATH_LOSS   = os.path.join(_IOBT_DATA, "path_loss_params_20260416.json")
DEFAULT_FLAC_DIR    = _IOBT_DATA
DEFAULT_STEP_S      = 0.5
DEFAULT_RSSI_WINDOW = 1   # set >1 to apply a rolling-dB average before inference


# ===========================================================================
# Data loading — single file
# ===========================================================================

DEFAULT_FILE = os.path.join(project_root, "data", "respeaker_power_20260416_152832.npy")


def load_positions(filepath: str) -> np.ndarray:
    """
    Load one respeaker_power_*.npy file and convert to a cell-index sequence.
    Returns int32 (T,) array of cell indices 0–5.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    arr      = np.load(filepath)                      # (T, 6)  float32
    positions = arr.argmax(axis=1).astype(np.int32)  # (T,)  cell 0–5

    counts = np.bincount(positions, minlength=N_CELLS)
    print(f"  File   : {os.path.basename(filepath)}")
    print(f"  Shape  : {arr.shape}  (timesteps × nodes)")
    print(f"  Total  : {len(positions)} timesteps")
    print(f"  Cell occupancy [0-5]: {counts.tolist()}")
    print(f"  Dominant cell: {int(counts.argmax())}  "
          f"({counts.max()/len(positions)*100:.1f}% of time)")
    return positions


# ===========================================================================
# Data loading — Bayesian RSSI inference from FLAC
# ===========================================================================

def load_positions_from_flac(flac_dir, session, nodes_txt, path_loss_json,
                              step_s=0.5, rssi_window=1) -> np.ndarray:
    """
    Run Bayesian path-loss inference on FLAC files for one session.

    For each 0.5-s window, converts linear power → dB for each node, evaluates
    the Gaussian posterior over a 20×20 hypothesis grid, and returns the cell
    index (0-5) of the nearest node to the MAP hypothesis.

    Returns int32 (T,) — same shape as load_positions(), compatible with all
    downstream processing (min-dwell, transition matrix, cycle extraction).
    """
    from collections import deque
    from pathlib import Path as _Path
    sys.path.insert(0, os.path.join(project_root, "collection"))
    from flac_to_trackmdp import (
        discover_flac_files, load_node_positions_ordered, load_path_loss,
        build_hypothesis_grid, compute_power_per_step, compute_posterior_top1,
    )

    flac_files = discover_flac_files(_Path(flac_dir), session=session)
    missing = [nid for nid in NODE_IDS if nid not in flac_files]
    if missing:
        raise FileNotFoundError(f"Missing FLAC for nodes: {missing}")
    print(f"  Found {len(flac_files)} FLAC files for session '{session}'")

    node_xy = load_node_positions_ordered(_Path(nodes_txt), NODE_IDS)
    P0, ETA, SIG2, MASK = load_path_loss(_Path(path_loss_json), NODE_IDS)
    print(f"  Loaded path-loss params: {os.path.basename(path_loss_json)}")

    hypos, dist, nearest_idx, _H, _W = build_hypothesis_grid(node_xy)

    power_arrays = {}
    for nid in NODE_IDS:
        pwr = compute_power_per_step(flac_files[nid], step_s)
        power_arrays[nid] = pwr
        print(f"    node {nid}: {len(pwr)} steps")
    T = min(len(power_arrays[nid]) for nid in NODE_IDS)
    print(f"  Using {T} steps (shortest file).")

    buffers  = {nid: deque(maxlen=rssi_window) for nid in NODE_IDS}
    positions = np.zeros(T, dtype=np.int32)
    prev_cell = 0
    for t in range(T):
        rss_map = {}
        for nid in NODE_IDS:
            db = 10.0 * np.log10(max(float(power_arrays[nid][t]), 1e-12))
            buffers[nid].append(db)
            rss_map[nid] = float(np.mean(buffers[nid]))

        top1 = compute_posterior_top1(rss_map, NODE_IDS, dist, nearest_idx,
                                      P0, ETA, SIG2, MASK, hypos)
        if top1 is not None:
            prev_cell = NODE_IDS.index(top1)
        positions[t] = prev_cell

    counts = np.bincount(positions, minlength=N_CELLS)
    print(f"  Cell occupancy [0-5]: {counts.tolist()}")
    print(f"  Dominant cell: {int(counts.argmax())}  "
          f"({counts.max()/T*100:.1f}% of time)")
    return positions


# ===========================================================================
# Data loading — supervised nearest-centroid classifier
# ===========================================================================

def load_positions_supervised(flac_dir, session, nodes_txt, gps_csv,
                               step_s=0.5) -> np.ndarray:
    """
    Train a nearest-centroid classifier on GPS-labelled RSSI vectors, then
    apply it to all FLAC timesteps.  Returns int32 (T,) cell indices 0-5.
    """
    from pathlib import Path as _Path
    sys.path.insert(0, os.path.join(project_root, "collection"))
    from flac_to_trackmdp import (
        discover_flac_files, load_node_positions_ordered, load_gps_track,
        _parse_session_start, build_training_pairs,
        fit_supervised_classifier, predict_supervised_sequence,
        compute_power_per_step,
    )

    flac_files = discover_flac_files(_Path(flac_dir), session=session)
    missing = [nid for nid in NODE_IDS if nid not in flac_files]
    if missing:
        raise FileNotFoundError(f"Missing FLAC for nodes: {missing}")
    print(f"  Found {len(flac_files)} FLAC files for session '{session}'")

    node_xy    = load_node_positions_ordered(_Path(nodes_txt), NODE_IDS)
    gps_df     = load_gps_track(_Path(gps_csv))
    flac_start = _parse_session_start(session)

    print("  Computing audio power …")
    power_arrays = {}
    for nid in NODE_IDS:
        pwr = compute_power_per_step(flac_files[nid], step_s)
        power_arrays[nid] = pwr
        print(f"    node {nid}: {len(pwr)} steps")
    T = min(len(power_arrays[nid]) for nid in NODE_IDS)

    distances, rssi_db, _ = build_training_pairs(
        gps_df, node_xy, power_arrays, NODE_IDS, flac_start, step_s
    )
    print(f"  Training on {len(rssi_db)} GPS-labelled samples …")
    classifier = fit_supervised_classifier(rssi_db, distances, NODE_IDS)

    print(f"  Predicting {T} steps …")
    positions = predict_supervised_sequence(power_arrays, NODE_IDS, classifier, step_s)

    counts = np.bincount(positions, minlength=N_CELLS)
    print(f"  Cell occupancy [0-5]: {counts.tolist()}")
    print(f"  Dominant cell: {int(counts.argmax())}  "
          f"({counts.max()/T*100:.1f}% of time)")
    return positions


# ===========================================================================
# Optional noise filtering — minimum dwell filter
# ===========================================================================

def apply_min_dwell(positions: np.ndarray, min_dwell: int) -> np.ndarray:
    """
    Remove single-step blips: if a cell appears for fewer than min_dwell
    consecutive steps, replace it with the previous stable cell.

    e.g. [0, 0, 1, 0, 0] with min_dwell=2 → [0, 0, 0, 0, 0]
         [0, 0, 1, 1, 0] with min_dwell=2 → [0, 0, 1, 1, 0]  (kept)
    """
    if min_dwell <= 1:
        return positions

    filtered = positions.copy()
    i = 0
    while i < len(filtered):
        cell  = filtered[i]
        run   = 1
        while i + run < len(filtered) and filtered[i + run] == cell:
            run += 1
        if run < min_dwell and i > 0:
            filtered[i:i+run] = filtered[i-1]   # replace blip with prior cell
        i += run

    n_changed = int((filtered != positions).sum())
    pct       = n_changed / len(positions) * 100
    print(f"\n  Min-dwell filter (dwell={min_dwell}): "
          f"replaced {n_changed} timesteps ({pct:.1f}%)")
    return filtered


# ===========================================================================
# Transition matrix
# ===========================================================================

def build_transition_matrix(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build raw count matrix and normalised probability matrix.

    T_prob[i, j] = P(move to j | currently at i), self-loops excluded.
    Rows with zero off-diagonal counts (cell never left) are left as all-zero.

    Returns (T_prob, T_count) both shape (N_CELLS, N_CELLS).
    """
    T_count = np.zeros((N_CELLS, N_CELLS), dtype=np.int64)
    for i, j in zip(positions[:-1], positions[1:]):
        T_count[i, j] += 1

    T_prob = np.zeros((N_CELLS, N_CELLS), dtype=np.float64)
    for i in range(N_CELLS):
        row = T_count[i].astype(float)
        row[i] = 0.0                     # exclude self-loop
        total  = row.sum()
        if total > 0:
            T_prob[i] = row / total

    return T_prob, T_count


# ===========================================================================
# Dominant cycle extraction
# ===========================================================================

def extract_greedy_cycle(T_prob: np.ndarray, min_prob: float) -> list[int]:
    """
    Greedy extraction of the dominant cycle.

    Starting from the cell with the highest total outgoing probability mass,
    repeatedly follow the highest-probability unvisited next cell until:
      - we return to start (cycle complete), OR
      - no unvisited next cell has probability >= min_prob

    Returns list of cell indices forming the cycle (not including the repeated
    start at the end).
    """
    # Start at the cell with most outgoing activity
    start   = int(T_prob.sum(axis=1).argmax())
    cycle   = [start]
    visited = {start}
    current = start

    for _ in range(N_CELLS - 1):
        row = T_prob[current].copy()
        row[list(visited)] = 0.0         # mask already-visited cells

        best_prob = row.max()
        if best_prob < min_prob:
            break                        # no strong next step

        nxt = int(row.argmax())
        cycle.append(nxt)
        visited.add(nxt)
        current = nxt

    return cycle


def cycle_coverage(cycle: list[int], T_count: np.ndarray) -> float:
    """
    Fraction of all observed transitions that are explained by the cycle.
    A transition (i, j) is explained if both i and j are in the cycle
    and j is the next cell after i in the cycle (wrapping around).
    """
    cycle_set = set(zip(cycle, cycle[1:] + [cycle[0]]))
    total_transitions = int(T_count.sum())
    if total_transitions == 0:
        return 0.0
    explained = sum(
        int(T_count[i, j]) for (i, j) in cycle_set
    )
    return explained / total_transitions


def validate_cycle_rect(cycle: list[int]) -> tuple[bool, str]:
    """Check cycle cells are in-bounds and Moore-adjacent on the 2×3 grid."""
    for c in cycle:
        if not (0 <= c < N_CELLS):
            return False, f"cell {c} out of range"
    for k in range(len(cycle)):
        a  = cycle[k]
        b  = cycle[(k+1) % len(cycle)]
        ac = a % NCOLS;  ar = a // NCOLS
        bc = b % NCOLS;  br = b // NCOLS
        if abs(ac - bc) > 1 or abs(ar - br) > 1:
            return False, f"cells {a}↔{b} not Moore-adjacent (diagonal or too far)"
    return True, "OK"


# ===========================================================================
# Saving outputs
# ===========================================================================

def save_outputs(T_prob, T_count, cycle, coverage, out_dir: str,
                 positions: np.ndarray, min_dwell: int, min_prob: float,
                 suffix: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)

    tag = f"_{suffix}" if suffix else ""
    np.save(os.path.join(out_dir, f"transition_matrix{tag}.npy"),     T_prob)
    np.save(os.path.join(out_dir, f"transition_matrix_raw{tag}.npy"), T_count)

    ok, msg = validate_cycle_rect(cycle)

    meta = {
        "cycle":          cycle,
        "cycle_str":      ",".join(str(c) for c in cycle),
        "coverage":       round(float(coverage), 4),
        "valid_for_rect": ok,
        "validation_msg": msg,
        "min_dwell":      min_dwell,
        "min_prob":       min_prob,
        "n_timesteps":    int(len(positions)),
        "cell_counts":    np.bincount(positions, minlength=N_CELLS).tolist(),
    }
    with open(os.path.join(out_dir, f"dominant_cycle{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Human-readable report
    lines = []
    lines.append("=" * 60)
    lines.append("  RESPEAKER TRANSITION ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"  Timesteps analysed : {len(positions)}")
    lines.append(f"  Min-dwell filter   : {min_dwell} steps")
    lines.append(f"  Min-prob threshold : {min_prob}")
    lines.append("")
    lines.append("  Cell occupancy (0-5):")
    counts = np.bincount(positions, minlength=N_CELLS)
    for i in range(N_CELLS):
        bar = "█" * int(counts[i] / max(counts) * 20)
        lines.append(f"    cell {i}: {counts[i]:6d} steps  {bar}")
    lines.append("")
    lines.append("  Transition probability matrix P[i→j]  (self-loops excluded):")
    header = "       " + "  ".join(f"  j={j}" for j in range(N_CELLS))
    lines.append(header)
    for i in range(N_CELLS):
        row_str = "  ".join(f"{T_prob[i,j]:6.3f}" for j in range(N_CELLS))
        lines.append(f"    i={i}  {row_str}")
    lines.append("")
    lines.append(f"  Dominant cycle     : {cycle}")
    lines.append(f"  Cycle coverage     : {coverage*100:.1f}% of observed transitions")
    lines.append(f"  Valid (rect grid)  : {ok}  — {msg}")
    lines.append("")
    lines.append("  " + "─" * 56)
    lines.append("  HOW TO USE THIS OUTPUT")
    lines.append("  " + "─" * 56)
    if ok:
        circle_arg = ",".join(str(c) for c in cycle)
        lines.append("")
        lines.append("  Pass the dominant cycle to finetune_deterministic.py:")
        lines.append("")
        lines.append(f"    python examples/finetune_deterministic.py \\")
        lines.append(f"        --run 201 \\")
        lines.append(f"        --circle {circle_arg} \\")
        lines.append(f"        --iterations 200")
        lines.append("")
        lines.append("  Or load programmatically:")
        lines.append("")
        lines.append("    import json")
        lines.append("    meta  = json.load(open('data/dominant_cycle.json'))")
        lines.append("    cycle = meta['cycle']  # e.g. [0, 1, 2, 5, 4, 3]")
    else:
        lines.append("")
        lines.append("  WARNING: extracted cycle is not valid for the 2×3 rect grid.")
        lines.append(f"  Reason: {msg}")
        lines.append("  The vehicle may not have followed a grid-connected path.")
        lines.append("  Options:")
        lines.append("    1. Increase --min-dwell to filter more noise")
        lines.append("    2. Increase --min-prob to require stronger transitions")
        lines.append("    3. Manually specify --circle in finetune_deterministic.py")
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    with open(os.path.join(out_dir, f"transition_report{tag}.txt"), "w") as f:
        f.write(report)

    print(report)


# ===========================================================================
# Optional matplotlib visualisation
# ===========================================================================

def plot_results(T_prob: np.ndarray, T_count: np.ndarray,
                 positions: np.ndarray, cycle: list[int],
                 out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Respeaker Transition Analysis", fontsize=13, fontweight="bold")

    # ── Panel 1: heatmap of transition probabilities ──────────────────────
    ax = axes[0]
    im = ax.imshow(T_prob, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Transition Probability P[i→j]\n(self-loops excluded)")
    ax.set_xlabel("To cell j"); ax.set_ylabel("From cell i")
    ax.set_xticks(range(N_CELLS)); ax.set_yticks(range(N_CELLS))
    for i in range(N_CELLS):
        for j in range(N_CELLS):
            ax.text(j, i, f"{T_prob[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if T_prob[i,j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── Panel 2: raw counts ───────────────────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(T_count, cmap="Oranges")
    ax.set_title("Raw Transition Counts")
    ax.set_xlabel("To cell j"); ax.set_ylabel("From cell i")
    ax.set_xticks(range(N_CELLS)); ax.set_yticks(range(N_CELLS))
    for i in range(N_CELLS):
        for j in range(N_CELLS):
            ax.text(j, i, str(int(T_count[i,j])), ha="center", va="center",
                    fontsize=8)
    plt.colorbar(im2, ax=ax, fraction=0.046)

    # ── Panel 3: 2×3 grid with cycle overlay ─────────────────────────────
    ax = axes[2]
    ax.set_title(f"Dominant Cycle: {cycle}")
    ax.set_xlim(-0.5, NCOLS - 0.5); ax.set_ylim(-0.5, NROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.invert_yaxis()

    # Grid cells
    counts = np.bincount(positions, minlength=N_CELLS)
    max_count = max(counts.max(), 1)
    for cell in range(N_CELLS):
        col, row = cell % NCOLS, cell // NCOLS
        intensity = counts[cell] / max_count
        color = plt.cm.Greens(0.2 + 0.6 * intensity)
        rect = mpatches.FancyBboxPatch(
            (col - 0.45, row - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05", facecolor=color,
            edgecolor="black", linewidth=1.5
        )
        ax.add_patch(rect)
        in_cycle = cell in cycle
        ax.text(col, row, f"{cell}", ha="center", va="center",
                fontsize=13, fontweight="bold" if in_cycle else "normal",
                color="white" if intensity > 0.5 else "black")

    # Cycle arrows
    for k in range(len(cycle)):
        a = cycle[k]
        b = cycle[(k+1) % len(cycle)]
        ax_col, ar = a % NCOLS, a // NCOLS
        bx_col, br = b % NCOLS, b // NCOLS
        dx = (bx_col - ax_col) * 0.55
        dy = (br - ar) * 0.55
        ax.annotate(
            "", xy=(bx_col - dx*0.1, br - dy*0.1),
            xytext=(ax_col + dx*0.1, ar + dy*0.1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2)
        )

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "transition_analysis.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build an empirical transition matrix from a single respeaker "
                    "power recording and extract the dominant cycle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files
------------
  transition_matrix.npy       float64 (6,6) — empirical P[i→j]
  transition_matrix_raw.npy   int64   (6,6) — raw transition counts
  dominant_cycle.json         {"cycle":[0,1,2,5,4,3], "coverage":0.81, ...}
  transition_report.txt       summary + exact finetune_deterministic.py command
  transition_analysis.png     heatmaps + grid visualisation (if matplotlib available)

Typical workflow
----------------
  1. Collect data with mqtt_collector.py / respeaker nodes
  2. Run this script to extract the dominant cycle from one recording:
       python examples/respeaker_transition.py --file data/respeaker_power_20260416_152832.npy
  3. Fine-tune on the extracted cycle:
       python examples/finetune_deterministic.py --run 201 --circle 0,1,2,5,4,3

Tuning parameters
-----------------
  --min-dwell   Increase if data is very noisy (try 2-5). Removes blips where
                a cell appears for fewer than N consecutive steps.
  --min-prob    Increase if extracted cycle is too long or invalid. Only follows
                transitions with probability above this threshold.
        """
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to a single respeaker_power_*.npy file (raw argmax mode). "
             "If omitted, FLAC-Bayesian mode is used instead."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: iobt_data/ for FLAC mode, "
             "same dir as --file for .npy mode)"
    )
    parser.add_argument(
        "--min-dwell", type=int, default=2,
        help="Minimum consecutive steps to count as a real cell visit "
             "(default: 2, set to 1 to disable)"
    )
    parser.add_argument(
        "--min-prob", type=float, default=0.05,
        help="Minimum transition probability to follow in cycle extraction "
             "(default: 0.05)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib visualisation"
    )
    # ── FLAC mode flags (supervised + Bayesian) ───────────────────────────
    parser.add_argument(
        "--flac-dir", type=str, default=None,
        help="Folder with FLAC files (default: iobt_data/)"
    )
    parser.add_argument(
        "--session", type=str, default=DEFAULT_SESSION,
        help=f"Session prefix to filter FLAC files (default: {DEFAULT_SESSION})"
    )
    parser.add_argument(
        "--nodes-txt", type=str, default=DEFAULT_NODES_TXT,
        help="Tab-delimited node_positions.txt, row order = nodes 11–16 "
             "(default: iobt_data/node_positions.txt)"
    )
    parser.add_argument(
        "--gps-csv", type=str,
        default=os.path.join(_IOBT_DATA, "20260416_154037_gps2_gps.csv"),
        help="GPS CSV for supervised training "
             "(default: iobt_data/20260416_154037_gps2_gps.csv)"
    )
    parser.add_argument(
        "--bayesian", action="store_true",
        help="Use Bayesian path-loss inference instead of supervised classifier"
    )
    parser.add_argument(
        "--path-loss-json", type=str, default=DEFAULT_PATH_LOSS,
        help="[--bayesian] Trained path-loss params JSON "
             "(default: iobt_data/path_loss_params_20260416.json)"
    )
    parser.add_argument(
        "--step", type=float, default=DEFAULT_STEP_S,
        help="FLAC window size in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--rssi-window", type=int, default=DEFAULT_RSSI_WINDOW,
        help="[--bayesian] Rolling-average dB window (default: 1 = no rolling)"
    )
    args = parser.parse_args()

    # ── Choose mode ───────────────────────────────────────────────────────
    if args.file is not None:
        mode = "npy"
    elif args.bayesian:
        mode = "bayesian"
    elif os.path.isfile(args.gps_csv):
        mode = "supervised"
    else:
        mode = "bayesian"   # no GPS CSV → fall back to Bayesian

    flac_dir = args.flac_dir or DEFAULT_FLAC_DIR

    if mode == "npy":
        filepath = os.path.abspath(args.file)
        if not os.path.isfile(filepath):
            print(f"[ERROR] File not found: {filepath}")
            sys.exit(1)
        mode_label = f"raw argmax  ({os.path.basename(filepath)})"
        out_dir    = args.out_dir or os.path.dirname(filepath)
        stem       = Path(filepath).stem
        prefix     = "respeaker_power_"
        suffix     = stem[len(prefix):] if stem.startswith(prefix) else stem
    else:
        filepath   = None
        suffix     = args.session
        out_dir    = args.out_dir or DEFAULT_FLAC_DIR
        if mode == "supervised":
            mode_label = f"supervised  (session={args.session})"
        else:
            mode_label = f"Bayesian  (session={args.session})"

    print("=" * 60)
    print("  RESPEAKER → TRANSITION MATRIX")
    print("=" * 60)
    print(f"  Mode       : {mode_label}")
    if mode == "npy":
        print(f"  Input file : {filepath}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Min dwell  : {args.min_dwell}")
    print(f"  Min prob   : {args.min_prob}")
    print()

    # ── Load positions ────────────────────────────────────────────────────
    if mode == "supervised":
        print("[1/4] Training supervised classifier from GPS + FLAC …")
        positions = load_positions_supervised(
            flac_dir, args.session, args.nodes_txt, args.gps_csv,
            step_s=args.step,
        )
    elif mode == "bayesian":
        print("[1/4] Running Bayesian RSSI inference from FLAC …")
        positions = load_positions_from_flac(
            flac_dir, args.session, args.nodes_txt, args.path_loss_json,
            step_s=args.step, rssi_window=args.rssi_window,
        )
    else:
        print("[1/4] Loading positions …")
        positions = load_positions(filepath)

    print("\n[2/4] Filtering noise …")
    positions = apply_min_dwell(positions, args.min_dwell)

    # ── Transition matrix ─────────────────────────────────────────────────
    print("\n[3/4] Building transition matrix …")
    T_prob, T_count = build_transition_matrix(positions)

    # ── Cycle extraction ──────────────────────────────────────────────────
    print("[4/4] Extracting dominant cycle …")
    cycle    = extract_greedy_cycle(T_prob, args.min_prob)
    coverage = cycle_coverage(cycle, T_count)
    ok, msg  = validate_cycle_rect(cycle)
    print(f"       Cycle     : {cycle}")
    print(f"       Coverage  : {coverage*100:.1f}% of transitions")
    print(f"       Valid     : {ok}  ({msg})")

    # ── Save outputs ──────────────────────────────────────────────────────
    save_outputs(T_prob, T_count, cycle, coverage, out_dir,
                 positions, args.min_dwell, args.min_prob, suffix=suffix)

    if not args.no_plot:
        print("\nGenerating plot …")
        plot_results(T_prob, T_count, positions, cycle, out_dir)

    # ── Final command ─────────────────────────────────────────────────────
    if ok:
        circle_arg = ",".join(str(c) for c in cycle)
        print(f"\nNext step — run finetune_deterministic.py with:")
        print(f"  python examples/finetune_deterministic.py \\")
        print(f"      --run 201 --circle {circle_arg}")
    else:
        print(f"\n[WARN] Extracted cycle is not grid-valid: {msg}")
        print("  Try --min-dwell 3 or --min-prob 0.15 to clean up the data.")


if __name__ == "__main__":
    main()