"""
Diagnostic: plot raw dB power per node per session, shaded by GPS-derived
"person present / absent" label.

Reveals whether the acoustic signal is visually discriminable before any
classifier is involved.  Run after confirming the RL method works (--gt mode).

Usage:
    python collection/diagnostic_plot_power_vs_label.py \
        --flac-dir iobt_data \
        --nodes-txt iobt_data/node_positions.txt \
        --sessions 20260416_154037 20260417_100634 20260417_103802 \
        --gps-dir iobt_data \
        --out-dir results/diagnostics
"""
import argparse
import os
import sys
import glob
from datetime import timedelta
from pathlib import Path

# Ensure project root is on sys.path so `collection` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NODE_IDS = [11, 12, 13, 14, 15, 16]
STEP_S_DEFAULT = 0.5


def gps_labels_per_step(gps_df, node_xy, session_start, T, step_s):
    """
    Returns (N, T) bool array.  label[i, t] is True when the person is
    nearest to node i at audio step t.
    """
    ts_arr = np.array([
        (session_start + timedelta(seconds=(t + 0.5) * step_s)).timestamp()
        for t in range(T)
    ])
    gps_ts = np.array([ts.timestamp() for ts in gps_df["timestamp"]])
    idx = np.argmin(np.abs(ts_arr[:, None] - gps_ts[None, :]), axis=1)  # (T,)
    xs = gps_df["x"].values[idx]
    ys = gps_df["y"].values[idx]
    dx = xs[:, None] - node_xy[None, :, 0]   # (T, N)
    dy = ys[:, None] - node_xy[None, :, 1]
    nearest = np.argmin(dx ** 2 + dy ** 2, axis=1)   # (T,) cell index
    return nearest[None, :] == np.arange(len(node_xy))[:, None]  # (N, T)


def plot_session(session, flac_dir, nodes_txt, gps_dir, step_s, out_dir):
    from collection.flac_to_trackmdp import (
        compute_power_per_step,
        discover_flac_files,
        load_gps_track,
        load_node_positions_ordered,
        _parse_session_start,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    file_map = discover_flac_files(Path(flac_dir), session=session)
    missing = [n for n in NODE_IDS if n not in file_map]
    if missing:
        print(f"[{session}] Skipping — missing nodes: {missing}")
        return

    gps_pattern = os.path.join(gps_dir, f"{session}_gps2_gps.csv")
    matches = glob.glob(gps_pattern)
    if not matches:
        print(f"[{session}] Skipping — no GPS CSV matching {gps_pattern}")
        return
    gps_csv = matches[0]

    node_xy      = load_node_positions_ordered(Path(nodes_txt), NODE_IDS)
    gps_df       = load_gps_track(Path(gps_csv))
    session_start = _parse_session_start(session)

    # Compute dB power per node
    power_db = {}
    for nid in NODE_IDS:
        linear = compute_power_per_step(file_map[nid], step_s)
        power_db[nid] = 10.0 * np.log10(np.maximum(linear, 1e-12))

    T = min(len(v) for v in power_db.values())
    labels = gps_labels_per_step(gps_df, node_xy, session_start, T, step_s)  # (6, T)
    time_ax = np.arange(T) * step_s  # seconds

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    fig.suptitle(f"Session {session} — raw dB power vs GPS label", fontsize=13)

    for col_idx, nid in enumerate(NODE_IDS):
        ax = axes[col_idx // 3][col_idx % 3]
        db = power_db[nid][:T]
        present = labels[col_idx]   # (T,) bool

        # Shade present regions
        in_span = False
        span_start = 0
        for t in range(T):
            if present[t] and not in_span:
                span_start = t
                in_span = True
            elif not present[t] and in_span:
                ax.axvspan(time_ax[span_start], time_ax[t],
                           alpha=0.15, color="green", linewidth=0)
                in_span = False
        if in_span:
            ax.axvspan(time_ax[span_start], time_ax[-1],
                       alpha=0.15, color="green", linewidth=0)

        ax.plot(time_ax, db, color="steelblue", linewidth=0.6, alpha=0.85)

        # Mean lines
        if present.any():
            mu_pres = db[present].mean()
            ax.axhline(mu_pres, color="green", linestyle="--",
                       linewidth=1.2, alpha=0.8)
        else:
            mu_pres = np.nan
        if (~present).any():
            mu_abs = db[~present].mean()
            ax.axhline(mu_abs, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.8)
        else:
            mu_abs = np.nan

        pct = present.mean() * 100
        delta = mu_pres - mu_abs if not (np.isnan(mu_pres) or np.isnan(mu_abs)) else float("nan")
        ax.set_title(f"Node {nid}  |  present {pct:.0f}%  |  Δ={delta:+.1f} dB",
                     fontsize=9)
        ax.set_ylabel("dB")
        ax.set_xlabel("time (s)")

    legend_handles = [
        mpatches.Patch(color="green", alpha=0.3, label="present (GPS)"),
        plt.Line2D([], [], color="green", linestyle="--", label="mean present"),
        plt.Line2D([], [], color="red",   linestyle="--", label="mean absent"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"diagnostic_{session}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{session}] Saved → {out_path}")

    # Print Δ summary
    print(f"  {'Node':>6}  {'present%':>9}  {'Δ dB':>8}")
    for col_idx, nid in enumerate(NODE_IDS):
        db = power_db[nid][:T]
        present = labels[col_idx]
        mu_p = db[present].mean() if present.any() else float("nan")
        mu_a = db[~present].mean() if (~present).any() else float("nan")
        delta = mu_p - mu_a
        print(f"  {nid:>6}  {present.mean()*100:>8.1f}%  {delta:>+8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw dB power vs GPS-derived present/absent label per node")
    parser.add_argument("--flac-dir",   required=True)
    parser.add_argument("--nodes-txt",  required=True)
    parser.add_argument("--gps-dir",    required=True,
                        help="Directory scanned for {session}_gps2_gps.csv")
    parser.add_argument("--sessions",   nargs="+",
                        default=["20260416_154037", "20260417_100634", "20260417_103802"])
    parser.add_argument("--step-s",     type=float, default=STEP_S_DEFAULT)
    parser.add_argument("--out-dir",    default="results/diagnostics")
    args = parser.parse_args()

    for session in args.sessions:
        print(f"\n=== {session} ===")
        plot_session(session, args.flac_dir, args.nodes_txt,
                     args.gps_dir, args.step_s, args.out_dir)


if __name__ == "__main__":
    main()
