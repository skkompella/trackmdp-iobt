#!/usr/bin/env python3
"""
eval_classifier_accuracy_10.py — Check classifier predictions against GPS ground truth.

Accuracy definition
-------------------
At each timestep t (where GPS coverage exists), score +1 if the nearest node to
the car's actual GPS position fired positive in the detection matrix B.

  correct[t] = B[t, node_to_col[gt_nid[t]]]

This rewards "did we alert the right node" regardless of false positives elsewhere.

Compares three modes:
  audio   — P_audio >= audio_threshold
  cam     — P_cam   >= cam_threshold
  fused   — audio OR cam  (must be >= audio-only and >= cam-only)

Usage
-----
  python collection/eval_classifier_accuracy_10.py \\
      --data-dir  iobt_data_10 \\
      --clfs-dir  results/v3_10node

  python collection/eval_classifier_accuracy_10.py \\
      --data-dir  iobt_data_10 \\
      --clfs-dir  results/v3_10node \\
      --session   20250812_165739 \\
      --plot
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import everything reusable from train_node_classifiers_10
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_node_classifiers_10 import (
    EXPECTED_NODE_IDS,
    DEFAULT_STEP_S,
    WINDOW_STEPS,
    FEATURE_NAMES,
    SESSION_RE,
    load_node_positions_from_meta,
    load_gps_track_10,
    build_ground_truth_sequence_10,
    _parse_session_start_10,
    _discover_session_flac_10,
    build_node_features,
    aggregate_rolling,
    build_p_cam_10,
    get_audio_probabilities_10,
    fuse_or,
)

import soundfile as sf


# ---------------------------------------------------------------------------
# Classifier loader
# ---------------------------------------------------------------------------

def load_classifiers_from_dir(
    clfs_dir: Path,
    node_ids: list[int],
) -> dict[int, tuple]:
    """
    Load trained classifiers and scalers from clfs_dir.

    Globs for node_clf_N{nid}_*v3*.pkl and node_scaler_N{nid}_*v3*.pkl.
    Takes the most recent match (sorted alphabetically, last = newest timestamp).

    Returns {nid: (clf, scaler)}.  Missing nodes are warned and skipped.
    """
    import joblib

    clfs: dict[int, tuple] = {}
    for nid in node_ids:
        clf_matches    = sorted(glob.glob(str(clfs_dir / f"node_clf_N{nid}_*v3*.pkl")))
        scaler_matches = sorted(glob.glob(str(clfs_dir / f"node_scaler_N{nid}_*v3*.pkl")))

        if not clf_matches or not scaler_matches:
            print(f"  [WARN] No classifier found for node {nid} in {clfs_dir}")
            continue

        clf    = joblib.load(clf_matches[-1])
        scaler = joblib.load(scaler_matches[-1])
        clfs[nid] = (clf, scaler)
        print(f"  [OK  ] node{nid}: {Path(clf_matches[-1]).name}")

    return clfs


# ---------------------------------------------------------------------------
# Core accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(
    B:           np.ndarray,
    gt_sequence: np.ndarray,
    node_ids:    list[int],
) -> dict:
    """
    Compute accuracy metrics comparing binary detection matrix B against GPS GT.

    Parameters
    ----------
    B           : (T, N) bool  — detection matrix
    gt_sequence : (T,) int32   — nearest-node ID per step; 0 = no GPS coverage
    node_ids    : ordered list of N node IDs (column order of B)

    Returns
    -------
    dict with keys:
      overall_accuracy   float  — correct / total_gps_steps
      correct_steps      int
      total_gps_steps    int
      no_fire_steps      int    — steps where B is all-zero (certain miss)
      multi_fire_steps   int    — steps where >1 node fires
      per_node_recall    {nid: float}  — TP / GT-positive per node
      per_node_fpr       {nid: float}  — FP / GT-negative per node
      per_node_gt_count  {nid: int}    — number of GPS-labelled steps at nid
      correct_mask       (T,) bool     — True where correct (only GPS-covered steps)
    """
    T = len(gt_sequence)
    assert B.shape[0] == T, f"B has {B.shape[0]} rows but gt_sequence has {T} elements"

    node_to_col = {nid: i for i, nid in enumerate(node_ids)}

    correct_mask = np.zeros(T, dtype=bool)
    gps_mask     = gt_sequence > 0

    tp  = {nid: 0 for nid in node_ids}   # B fires AND gt == nid
    fp  = {nid: 0 for nid in node_ids}   # B fires AND gt != nid
    gt_count = {nid: 0 for nid in node_ids}

    for t in range(T):
        gt_nid = int(gt_sequence[t])
        if gt_nid == 0:
            continue

        gt_col = node_to_col.get(gt_nid)
        if gt_col is None:
            continue

        gt_count[gt_nid] += 1

        if B[t, gt_col]:
            correct_mask[t] = True
            tp[gt_nid] += 1

        # False positives: nodes that fired but are NOT the true nearest
        for col, nid in enumerate(node_ids):
            if nid == gt_nid:
                continue
            if B[t, col]:
                fp[nid] += 1

    total_gps   = int(gps_mask.sum())
    correct     = int(correct_mask[gps_mask].sum())
    no_fire     = int((~B[gps_mask].any(axis=1)).sum())
    multi_fire  = int((B[gps_mask].sum(axis=1) > 1).sum())

    per_node_recall = {}
    per_node_fpr    = {}
    for nid in node_ids:
        gt_pos = gt_count[nid]
        gt_neg = total_gps - gt_pos
        per_node_recall[nid] = tp[nid] / gt_pos if gt_pos > 0 else float("nan")
        per_node_fpr[nid]    = fp[nid] / gt_neg if gt_neg > 0 else float("nan")

    return dict(
        overall_accuracy  = correct / total_gps if total_gps > 0 else 0.0,
        correct_steps     = correct,
        total_gps_steps   = total_gps,
        no_fire_steps     = no_fire,
        multi_fire_steps  = multi_fire,
        per_node_recall   = per_node_recall,
        per_node_fpr      = per_node_fpr,
        per_node_gt_count = gt_count,
        correct_mask      = correct_mask,
    )


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------

def evaluate_top_k_accuracy(
    P:           np.ndarray,
    gt_sequence: np.ndarray,
    node_ids:    list[int],
    k:           int = 3,
) -> dict:
    """
    Compute top-k accuracy: +1 if the GT-nearest node is among the k nodes
    with the highest predicted probability at each GPS-covered timestep.

    Parameters
    ----------
    P           : (T, N) float — raw probabilities or scores (NOT thresholded)
    gt_sequence : (T,) int32  — nearest-node ID per step; 0 = no GPS coverage
    node_ids    : ordered list of N node IDs (column order of P)
    k           : number of top candidates to consider

    Returns
    -------
    dict with keys: top_k_accuracy, correct_steps, total_gps_steps,
                    correct_mask, k
    """
    node_to_col  = {nid: i for i, nid in enumerate(node_ids)}
    gps_mask     = gt_sequence > 0
    correct_mask = np.zeros(len(gt_sequence), dtype=bool)

    for t in np.where(gps_mask)[0]:
        gt_col = node_to_col.get(int(gt_sequence[t]))
        if gt_col is None:
            continue
        top_k_cols = np.argsort(P[t])[::-1][:k]   # descending, first k
        if gt_col in top_k_cols:
            correct_mask[t] = True

    total_gps = int(gps_mask.sum())
    correct   = int(correct_mask[gps_mask].sum())
    return dict(
        top_k_accuracy  = correct / total_gps if total_gps > 0 else 0.0,
        correct_steps   = correct,
        total_gps_steps = total_gps,
        correct_mask    = correct_mask,
        k               = k,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    session:       str,
    node_ids:      list[int],
    metrics_audio: dict,
    metrics_cam:   dict,
    metrics_fused: dict,
    audio_thr:     float,
    cam_thr:       float,
    topk_audio:    dict | None = None,
    topk_cam:      dict | None = None,
    topk_fused:    dict | None = None,
) -> None:
    W = 68
    print("\n" + "=" * W)
    print(f"  CLASSIFIER ACCURACY REPORT — session {session}")
    print("=" * W)
    print(f"  Thresholds: audio>={audio_thr}  cam>={cam_thr}")
    print()

    # Overall summary
    for label, m in [("Audio only", metrics_audio),
                     ("Camera only", metrics_cam),
                     ("Fused (OR)", metrics_fused)]:
        acc = m["overall_accuracy"]
        cor = m["correct_steps"]
        tot = m["total_gps_steps"]
        nf  = m["no_fire_steps"]
        mf  = m["multi_fire_steps"]
        print(f"  {label:<14} accuracy = {acc:.3f}  ({cor}/{tot} steps)  "
              f"no-fire={nf}  multi-fire={mf}")

    print()
    # Sanity checks
    if metrics_fused["overall_accuracy"] >= metrics_audio["overall_accuracy"] - 1e-9:
        print("  [OK] fused >= audio  ✓")
    else:
        print("  [WARN] fused < audio — unexpected!")
    if metrics_fused["overall_accuracy"] >= metrics_cam["overall_accuracy"] - 1e-9:
        print("  [OK] fused >= cam    ✓")
    else:
        print("  [WARN] fused < cam — unexpected!")

    # Top-k accuracy block (shown when top-k dicts are provided)
    if topk_audio is not None:
        k = topk_audio["k"]
        print()
        print(f"  Top-{k} accuracy  (GT in top-{k} by probability):")
        for label, m in [("Audio only ", topk_audio),
                          ("Camera only", topk_cam),
                          ("Fused (max)", topk_fused)]:
            acc = m["top_k_accuracy"]
            cor = m["correct_steps"]
            tot = m["total_gps_steps"]
            print(f"    {label}  {acc:.3f}  ({cor}/{tot} steps)")

    print()
    # Per-node table
    col_w = 8
    hdr = f"  {'Node':>5}  {'GT steps':>8}  "
    hdr += f"{'Recall(A)':>{col_w}}  {'Recall(C)':>{col_w}}  {'Recall(F)':>{col_w}}  "
    hdr += f"{'FPR(A)':>{col_w}}  {'FPR(C)':>{col_w}}  {'FPR(F)':>{col_w}}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for nid in node_ids:
        gt_n = metrics_audio["per_node_gt_count"][nid]
        ra   = metrics_audio ["per_node_recall"][nid]
        rc   = metrics_cam   ["per_node_recall"][nid]
        rf   = metrics_fused ["per_node_recall"][nid]
        fa   = metrics_audio ["per_node_fpr"   ][nid]
        fc   = metrics_cam   ["per_node_fpr"   ][nid]
        ff   = metrics_fused ["per_node_fpr"   ][nid]

        def _fmt(v):
            return f"{v:.3f}" if not (v != v) else "  n/a"  # NaN check

        print(f"  {nid:>5}  {gt_n:>8}  "
              f"{_fmt(ra):>{col_w}}  {_fmt(rc):>{col_w}}  {_fmt(rf):>{col_w}}  "
              f"{_fmt(fa):>{col_w}}  {_fmt(fc):>{col_w}}  {_fmt(ff):>{col_w}}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Optional timeline plot
# ---------------------------------------------------------------------------

def plot_timeline(
    gt_sequence:   np.ndarray,
    B_audio:       np.ndarray,
    B_cam:         np.ndarray,
    B_fused:       np.ndarray,
    node_ids:      list[int],
    session:       str,
    step_s:        float,
    out_path:      Path | None = None,
) -> None:
    """
    Plot per-node timeline showing GT position, audio fires, cam fires, fused fires.
    """
    import matplotlib
    matplotlib.use("Agg" if out_path else "TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    T   = len(gt_sequence)
    N   = len(node_ids)
    t_ax = np.arange(T) * step_s

    node_to_col = {nid: i for i, nid in enumerate(node_ids)}

    fig, axes = plt.subplots(N, 1, figsize=(16, 1.8 * N), sharex=True)
    if N == 1:
        axes = [axes]

    fig.suptitle(f"Classifier vs GPS ground truth — session {session}", fontsize=12)

    for row, nid in enumerate(node_ids):
        ax  = axes[row]
        col = node_to_col[nid]

        # GT presence (shaded background)
        gt_here = (gt_sequence == nid).astype(float)
        ax.fill_between(t_ax, 0, gt_here, step="post",
                        color="green", alpha=0.25, label="GT present")

        # Audio fires
        ax.step(t_ax, B_audio[:, col].astype(float) * 0.85 + 0.0,
                where="post", color="blue", linewidth=0.8, alpha=0.7, label="audio")

        # Cam fires
        ax.step(t_ax, B_cam[:, col].astype(float) * 0.70 + 0.0,
                where="post", color="orange", linewidth=0.8, alpha=0.7, label="cam")

        # Fused fires
        ax.step(t_ax, B_fused[:, col].astype(float) * 0.55 + 0.0,
                where="post", color="red", linewidth=1.1, alpha=0.9, label="fused")

        ax.set_ylabel(f"node{nid}", fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(-0.05, 1.15)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("Elapsed time (s)")

    # Single legend at top
    handles = [
        mpatches.Patch(color="green",  alpha=0.4, label="GT nearest node"),
        plt.Line2D([0], [0], color="blue",   linewidth=1.2, label="audio fires"),
        plt.Line2D([0], [0], color="orange", linewidth=1.2, label="cam fires"),
        plt.Line2D([0], [0], color="red",    linewidth=1.5, label="fused fires"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n  Plot saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-session evaluation
# ---------------------------------------------------------------------------

def evaluate_session(
    session_dir:     Path,
    clfs_dict:       dict[int, tuple],
    node_ids:        list[int],
    step_s:          float,
    audio_threshold: float,
    cam_threshold:   float,
    do_plot:         bool,
    plot_dir:        Path | None,
    top_k:           int = 3,
) -> dict:
    """Run all three modes and return combined metrics for one session."""

    session = session_dir.name
    print(f"\n{'─'*60}")
    print(f"  Evaluating session: {session}")
    print(f"{'─'*60}")

    # Determine T = min FLAC duration in bins
    file_map = _discover_session_flac_10(session_dir)
    T_list = []
    for nid in node_ids:
        fpath = file_map.get(nid)
        if fpath and fpath.exists():
            info = sf.info(str(fpath))
            T_list.append(int(info.duration / step_s))
    if not T_list:
        print(f"  [ERROR] No FLAC files found in {session_dir}")
        return {}
    n_timesteps = min(T_list)

    # GPS ground truth sequence
    node_positions = load_node_positions_from_meta(session_dir / "meta_data.json")
    gt_df          = load_gps_track_10(session_dir / "gt_tracks.csv")
    gt_sequence    = build_ground_truth_sequence_10(
        gt_df, node_positions, step_s, n_timesteps
    )
    n_gps = int((gt_sequence > 0).sum())
    print(f"  T={n_timesteps} bins  GPS-covered={n_gps} ({n_gps/n_timesteps:.1%})")

    # Audio probabilities
    print("  Building P_audio …")
    P_audio = get_audio_probabilities_10(clfs_dict, session_dir, n_timesteps, step_s)

    # Camera probabilities
    session_start_unix = _parse_session_start_10(session).timestamp()
    print("  Building P_cam …")
    P_cam = build_p_cam_10(
        session_dir, node_ids, session_start_unix, n_timesteps, bin_size_s=step_s,
    )

    # Three binary matrices
    B_audio = (P_audio >= audio_threshold)
    B_cam   = (P_cam   >= cam_threshold)
    B_fused = fuse_or(P_audio, P_cam, audio_threshold, cam_threshold)

    # Evaluate all three (threshold-based)
    m_audio = evaluate_accuracy(B_audio, gt_sequence, node_ids)
    m_cam   = evaluate_accuracy(B_cam,   gt_sequence, node_ids)
    m_fused = evaluate_accuracy(B_fused, gt_sequence, node_ids)

    # Top-k accuracy (probability-ranked)
    P_fused_score = np.maximum(P_audio, P_cam)
    topk_audio = evaluate_top_k_accuracy(P_audio,       gt_sequence, node_ids, top_k)
    topk_cam   = evaluate_top_k_accuracy(P_cam,         gt_sequence, node_ids, top_k)
    topk_fused = evaluate_top_k_accuracy(P_fused_score, gt_sequence, node_ids, top_k)

    print_report(session, node_ids, m_audio, m_cam, m_fused, audio_threshold, cam_threshold,
                 topk_audio, topk_cam, topk_fused)

    if do_plot:
        out_path = (plot_dir / f"timeline_{session}.png") if plot_dir else None
        plot_timeline(
            gt_sequence, B_audio, B_cam, B_fused,
            node_ids, session, step_s, out_path,
        )

    return dict(audio=m_audio, cam=m_cam, fused=m_fused, session=session,
                topk_audio=topk_audio, topk_cam=topk_cam, topk_fused=topk_fused)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classifier predictions against GPS ground truth (10-node IoBT).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python collection/eval_classifier_accuracy_10.py \\
      --data-dir iobt_data_10 --clfs-dir results/v3_10node

  python collection/eval_classifier_accuracy_10.py \\
      --data-dir iobt_data_10 --clfs-dir results/v3_10node \\
      --session 20250812_165739 --plot --plot-dir results/plots
        """,
    )
    _D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "iobt_data_10")
    _C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "results", "v3_10node")

    parser.add_argument("--data-dir",  type=Path, default=Path(_D))
    parser.add_argument("--clfs-dir",  type=Path, default=Path(_C))
    parser.add_argument("--session",   type=str,  default=None,
                        help="Evaluate only this session (default: all)")
    parser.add_argument("--step",      type=float, default=DEFAULT_STEP_S)
    parser.add_argument("--audio-threshold", type=float, default=0.6)
    parser.add_argument("--cam-threshold",   type=float, default=0.3)
    parser.add_argument("--top-k",     type=int, default=3,
                        help="k for top-k accuracy metric (default: 3)")
    parser.add_argument("--plot",      action="store_true",
                        help="Save timeline plots")
    parser.add_argument("--plot-dir",  type=Path, default=None,
                        help="Directory for plot output (default: results/plots)")
    args = parser.parse_args()

    plot_dir = args.plot_dir or (
        Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) /
        "results" / "plots"
    )

    print("=" * 60)
    print("  CLASSIFIER ACCURACY EVALUATION — 10-NODE IoBT")
    print("=" * 60)
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Clfs dir  : {args.clfs_dir}")
    print(f"  Step      : {args.step}s")
    print(f"  Thresholds: audio>={args.audio_threshold}  cam>={args.cam_threshold}")

    # Discover sessions
    if args.session:
        sessions = [args.session]
        sess_dirs = [args.data_dir / args.session]
        missing = [s for s in sess_dirs if not s.exists()]
        if missing:
            print(f"[ERROR] Session directory not found: {missing}")
            sys.exit(1)
    else:
        sess_dirs = sorted([
            sub for sub in args.data_dir.iterdir()
            if sub.is_dir() and SESSION_RE.match(sub.name)
        ])
        sessions = [s.name for s in sess_dirs]

    if not sessions:
        print(f"[ERROR] No sessions found in {args.data_dir}")
        sys.exit(1)
    print(f"  Sessions  : {sessions}")

    # Load classifiers
    print(f"\n  Loading classifiers from {args.clfs_dir} …")
    clfs_dict = load_classifiers_from_dir(args.clfs_dir, EXPECTED_NODE_IDS)
    if not clfs_dict:
        print("[ERROR] No classifiers loaded.")
        sys.exit(1)
    loaded_nodes = sorted(clfs_dict.keys())
    missing_nodes = [n for n in EXPECTED_NODE_IDS if n not in clfs_dict]
    if missing_nodes:
        print(f"  [WARN] Missing classifiers for nodes: {missing_nodes} — these columns will be zero")

    # Evaluate each session
    all_results = []
    for sess_dir in sess_dirs:
        result = evaluate_session(
            sess_dir, clfs_dict, EXPECTED_NODE_IDS,
            step_s=args.step,
            audio_threshold=args.audio_threshold,
            cam_threshold=args.cam_threshold,
            do_plot=args.plot,
            plot_dir=plot_dir,
            top_k=args.top_k,
        )
        if result:
            all_results.append(result)

    # Multi-session summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("  MULTI-SESSION SUMMARY")
        print("=" * 60)
        for mode_key, label in [("audio", "Audio only"),
                                 ("cam",   "Camera only"),
                                 ("fused", "Fused (OR)")]:
            accs = [r[mode_key]["overall_accuracy"] for r in all_results]
            print(f"  {label:<14} mean accuracy = {np.mean(accs):.3f}  "
                  f"range [{min(accs):.3f}, {max(accs):.3f}]")
        print()
        k = args.top_k
        print(f"  Top-{k} accuracy (GT in top-{k} by probability):")
        for topk_key, label in [("topk_audio", "Audio only"),
                                  ("topk_cam",   "Camera only"),
                                  ("topk_fused", "Fused (max)")]:
            accs = [r[topk_key]["top_k_accuracy"] for r in all_results if topk_key in r]
            if accs:
                print(f"    {label:<14} mean = {np.mean(accs):.3f}  "
                      f"range [{min(accs):.3f}, {max(accs):.3f}]")
        print("=" * 60)


if __name__ == "__main__":
    main()
