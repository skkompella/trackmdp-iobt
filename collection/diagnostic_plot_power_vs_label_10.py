"""
Diagnostic: plot raw dB power per node per session for the 10-node IOBT environment,
shaded by GPS/GT-derived "person present / absent" label.

Node positions are read from meta_data.json (local metre coordinates).
Ground-truth tracks are read from gt_tracks.csv (x, y already in same metre system).
FLAC files: node1_respeaker.flac … node10_respeaker.flac (no session prefix).

Usage:
    python collection/diagnostic_plot_power_vs_label_10.py \
        --flac-dir iobt_data_10 \
        --meta-json iobt_data_10/meta_data.json \
        --gt-csv iobt_data_10/gt_tracks.csv \
        --out-dir results/diagnostics_10
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NODE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
STEP_S_DEFAULT = 0.5
NCOLS = 5


def load_node_positions_from_meta(meta_json):
    """Returns (10, 2) float64 array of (X, Y) in local metres, ordered by NODE_IDS."""
    with open(meta_json) as f:
        d = json.load(f)
    sensors = d["sensors"]
    coords = []
    for nid in NODE_IDS:
        key = f"node{nid}_respeaker"
        coords.append([sensors[key]["X"], sensors[key]["Y"]])
    return np.array(coords, dtype=np.float64)


def load_gt_tracks(gt_csv):
    """
    Returns DataFrame with columns: timestamp (UTC-aware datetime), x (m), y (m).
    gt_tracks.csv format: DateTime,ElapsedTime,ObjectID,x,y,z,...
    """
    df = pd.read_csv(gt_csv)
    df["timestamp"] = pd.to_datetime(df["DateTime"]).dt.tz_localize("UTC")
    return df[["timestamp", "x", "y"]].reset_index(drop=True)


def session_start_from_meta(meta_json):
    """Parses the 'name' field of meta_data.json to a UTC-aware datetime."""
    with open(meta_json) as f:
        d = json.load(f)
    name = d["name"]  # e.g. "20250812_165739"
    return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def gps_labels_per_step(gt_df, node_xy, session_start, T, step_s):
    """
    Returns (N, T) bool array.  label[i, t] is True when the person is
    nearest to node i at audio step t.
    """
    ts_arr = np.array([
        (session_start + timedelta(seconds=(t + 0.5) * step_s)).timestamp()
        for t in range(T)
    ])
    gt_ts = np.array([ts.timestamp() for ts in gt_df["timestamp"]])
    idx = np.argmin(np.abs(ts_arr[:, None] - gt_ts[None, :]), axis=1)  # (T,)
    xs = gt_df["x"].values[idx]
    ys = gt_df["y"].values[idx]
    dx = xs[:, None] - node_xy[None, :, 0]   # (T, N)
    dy = ys[:, None] - node_xy[None, :, 1]
    nearest = np.argmin(dx ** 2 + dy ** 2, axis=1)   # (T,) node index
    return nearest[None, :] == np.arange(len(node_xy))[:, None]  # (N, T)


def run(flac_dir, meta_json, gt_csv, step_s, out_dir):
    from collection.flac_to_trackmdp import compute_power_per_step, discover_flac_files

    # ── Load metadata ────────────────────────────────────────────────────────
    node_xy       = load_node_positions_from_meta(meta_json)
    gt_df         = load_gt_tracks(gt_csv)
    session_start = session_start_from_meta(meta_json)
    session_name  = Path(meta_json).parent.name or datetime.strftime(session_start, "%Y%m%d_%H%M%S")

    # ── Discover FLAC files (no session prefix — pass session=None) ───────────
    file_map = discover_flac_files(Path(flac_dir), session=None)
    missing = [n for n in NODE_IDS if n not in file_map]
    if missing:
        print(f"Missing nodes: {missing}")
        return

    # ── Compute dB power ────────────────────────────────────────────────────
    power_db = {}
    for nid in NODE_IDS:
        linear = compute_power_per_step(file_map[nid], step_s)
        power_db[nid] = 10.0 * np.log10(np.maximum(linear, 1e-12))

    T = min(len(v) for v in power_db.values())
    labels  = gps_labels_per_step(gt_df, node_xy, session_start, T, step_s)
    time_ax = np.arange(T) * step_s

    print(f"T={T} steps ({T*step_s:.0f} s),  GT rows={len(gt_df)}")

    # ── Plot (2 rows × 5 cols) ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, NCOLS, figsize=(28, 8), sharex=True)
    fig.suptitle(f"Session {session_name} — raw dB power vs GT label (10-node IOBT)",
                 fontsize=13)

    for col_idx, nid in enumerate(NODE_IDS):
        ax = axes[col_idx // NCOLS][col_idx % NCOLS]
        db = power_db[nid][:T]
        present = labels[col_idx]

        # Shade present regions
        in_span, span_start = False, 0
        for t in range(T):
            if present[t] and not in_span:
                span_start, in_span = t, True
            elif not present[t] and in_span:
                ax.axvspan(time_ax[span_start], time_ax[t],
                           alpha=0.15, color="green", linewidth=0)
                in_span = False
        if in_span:
            ax.axvspan(time_ax[span_start], time_ax[-1],
                       alpha=0.15, color="green", linewidth=0)

        ax.plot(time_ax, db, color="steelblue", linewidth=0.6, alpha=0.85)

        mu_pres = db[present].mean()  if present.any()  else np.nan
        mu_abs  = db[~present].mean() if (~present).any() else np.nan
        if not np.isnan(mu_pres):
            ax.axhline(mu_pres, color="green", linestyle="--", linewidth=1.2, alpha=0.8)
        if not np.isnan(mu_abs):
            ax.axhline(mu_abs,  color="red",   linestyle="--", linewidth=1.2, alpha=0.8)

        pct   = present.mean() * 100
        delta = mu_pres - mu_abs if not (np.isnan(mu_pres) or np.isnan(mu_abs)) else float("nan")
        ax.set_title(f"Node {nid}  |  present {pct:.0f}%  |  Δ={delta:+.1f} dB", fontsize=9)
        ax.set_ylabel("dB")
        ax.set_xlabel("time (s)")

    fig.legend(handles=[
        mpatches.Patch(color="green", alpha=0.3, label="present (GT)"),
        plt.Line2D([], [], color="green", linestyle="--", label="mean present"),
        plt.Line2D([], [], color="red",   linestyle="--", label="mean absent"),
    ], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"diagnostic_{session_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Δ summary
    print(f"\n  {'Node':>6}  {'present%':>9}  {'Δ dB':>8}")
    for col_idx, nid in enumerate(NODE_IDS):
        db = power_db[nid][:T]
        present = labels[col_idx]
        mu_p = db[present].mean()  if present.any()  else float("nan")
        mu_a = db[~present].mean() if (~present).any() else float("nan")
        print(f"  {nid:>6}  {present.mean()*100:>8.1f}%  {mu_p - mu_a:>+8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw dB power vs GT label per node (10-node IOBT)")
    parser.add_argument("--flac-dir",  default="iobt_data_10")
    parser.add_argument("--meta-json", default="iobt_data_10/meta_data.json")
    parser.add_argument("--gt-csv",    default="iobt_data_10/gt_tracks.csv")
    parser.add_argument("--step-s",    type=float, default=STEP_S_DEFAULT)
    parser.add_argument("--out-dir",   default="results/diagnostics_10")
    args = parser.parse_args()

    run(args.flac_dir, args.meta_json, args.gt_csv, args.step_s, args.out_dir)


if __name__ == "__main__":
    main()
