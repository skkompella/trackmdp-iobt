#!/usr/bin/env python3
"""
evaluate_classifier.py — Measure per-timestep accuracy of the supervised
classifier against GPS ground truth.

Compares ground_truth_*.npy (GPS-derived node labels) with
rankings_supervised_*.npy (classifier predictions) produced by
supervised_transition.py.

Usage
-----
Explicit files:
    python collection/evaluate_classifier.py \\
        --gt   results/20260417_100634/ground_truth_20260504_204435.npy \\
        --pred results/20260417_100634/rankings_supervised_20260504_204435.npy \\
        --out-dir results/20260417_100634

Auto-discover latest pair in a results directory:
    python collection/evaluate_classifier.py \\
        --results-dir results/20260417_100634

Outputs
-------
  eval_report_SESSION.txt       overall + per-node accuracy table
  confusion_matrix_SESSION.png  row-normalised heatmap
  accuracy_timeline_SESSION.png rolling accuracy over time
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _latest_pair(results_dir: Path) -> tuple[Path, Path]:
    """Return the (gt_path, pred_path) pair with the most recent timestamp."""
    ts_re = re.compile(r'(\d{8}_\d{6})')

    gt_files   = {m.group(1): p for p in results_dir.glob("ground_truth_*.npy")
                  if (m := ts_re.search(p.name)) and not p.name.endswith("_meta.json")}
    pred_files = {m.group(1): p for p in results_dir.glob("rankings_supervised_*.npy")
                  if (m := ts_re.search(p.name))}

    common = sorted(set(gt_files) & set(pred_files), reverse=True)
    if not common:
        raise FileNotFoundError(
            f"No matching ground_truth / rankings_supervised pair found in {results_dir}"
        )
    ts = common[0]
    return gt_files[ts], pred_files[ts]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    """
    Returns:
        node_ids    sorted list of node IDs present in gt
        overall_acc float
        per_node    dict node_id -> {accuracy, precision, recall, f1, support}
        C_raw       (N,N) int64 confusion counts  (row=true, col=pred)
        C_norm      (N,N) float64 row-normalised
        correct     (T,) bool mask over valid steps (for timeline plot)
        valid_mask  (T,) bool
    """
    valid    = gt > 0
    node_ids = sorted(int(n) for n in np.unique(gt[valid]))
    N        = len(node_ids)
    id2i     = {nid: i for i, nid in enumerate(node_ids)}

    n_valid  = int(valid.sum())
    n_correct = int(((pred == gt) & valid).sum())
    overall_acc = n_correct / n_valid if n_valid > 0 else 0.0

    per_node = {}
    for nid in node_ids:
        tp = int(((pred == nid) & (gt == nid) & valid).sum())
        fp = int(((pred == nid) & (gt != nid) & valid).sum())
        fn = int(((pred != nid) & (gt == nid) & valid).sum())
        support   = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        node_acc  = tp / support if support > 0 else 0.0
        per_node[nid] = dict(accuracy=node_acc, precision=precision,
                             recall=recall, f1=f1, support=support)

    C_raw = np.zeros((N, N), dtype=np.int64)
    for t in range(len(gt)):
        if not valid[t]:
            continue
        i = id2i.get(int(gt[t]))
        j = id2i.get(int(pred[t]))
        if i is not None and j is not None:
            C_raw[i, j] += 1

    row_sums = C_raw.sum(axis=1, keepdims=True)
    C_norm   = np.where(row_sums > 0, C_raw / row_sums.astype(float), 0.0)

    correct = (pred == gt) & valid

    return node_ids, overall_acc, per_node, C_raw, C_norm, correct, valid


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(node_ids, overall_acc, per_node, C_raw, C_norm,
                 n_valid, gt_path, pred_path, session):
    lines = [
        "=" * 64,
        "  CLASSIFIER EVALUATION REPORT",
        "=" * 64,
        f"  Ground truth : {gt_path.name}",
        f"  Predictions  : {pred_path.name}",
        f"  Session tag  : {session}",
        f"  Created      : {datetime.now(timezone.utc).isoformat()}",
        "",
        f"  Overall accuracy : {overall_acc * 100:.1f}%  "
        f"({int(overall_acc * n_valid)}/{n_valid} valid steps)",
        "",
        "  Per-node metrics:",
        f"  {'node':>6}  {'support':>8}  {'accuracy':>9}  "
        f"{'precision':>10}  {'recall':>7}  {'F1':>7}",
        "  " + "-" * 56,
    ]
    for nid in node_ids:
        m = per_node[nid]
        lines.append(
            f"  {nid:>6}  {m['support']:>8d}  {m['accuracy']*100:>8.1f}%  "
            f"{m['precision']:>10.3f}  {m['recall']:>7.3f}  {m['f1']:>7.3f}"
        )
    lines += [
        "",
        f"  Confusion matrix (row=true node, col=predicted, row-normalised):",
        "         " + "  ".join(f"n={nid:2d}" for nid in node_ids),
    ]
    for i, nid_i in enumerate(node_ids):
        row = "  ".join(f"{C_norm[i, j]:5.2f}" for j in range(len(node_ids)))
        lines.append(f"  n={nid_i:2d}  {row}")
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion(C_norm, node_ids, out_dir, session):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping confusion matrix plot.")
        return

    N   = len(node_ids)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion Matrix\n(row = true node, col = predicted, row-normalised)")
    ax.set_xlabel("Predicted node"); ax.set_ylabel("True node")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    labels = [str(n) for n in node_ids]
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(N):
        for j in range(N):
            v = C_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if v > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = out_dir / f"confusion_matrix_{session}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  confusion_matrix .png → {path}")


def plot_timeline(correct, valid, step_s, out_dir, session, window=50):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping timeline plot.")
        return

    T      = len(correct)
    times  = np.arange(T) * step_s

    # Causal rolling mean over `window` steps
    c_float = correct.astype(float)
    kernel  = np.ones(window) / window
    rolling = np.convolve(c_float, kernel, mode='full')[:T]
    # Zero out the first window-1 steps where the window isn't full yet
    rolling[:window - 1] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, rolling, linewidth=1.2, color="steelblue",
            label=f"Rolling accuracy (window={window} steps / {window * step_s:.0f}s)")
    ax.axhline(correct[valid].mean(), color="red", linestyle="--",
               linewidth=1, label=f"Overall accuracy")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Accuracy Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = out_dir / f"accuracy_timeline_{session}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  accuracy_timeline .png → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate supervised classifier accuracy against GPS ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Explicit files:
  python collection/evaluate_classifier.py \\
      --gt   results/20260417_100634/ground_truth_20260504_204435.npy \\
      --pred results/20260417_100634/rankings_supervised_20260504_204435.npy \\
      --out-dir results/20260417_100634

  # Auto-discover latest pair:
  python collection/evaluate_classifier.py \\
      --results-dir results/20260417_100634
        """
    )
    parser.add_argument("--gt",          type=Path,
                        help="Path to ground_truth_*.npy")
    parser.add_argument("--pred",        type=Path,
                        help="Path to rankings_supervised_*.npy")
    parser.add_argument("--results-dir", type=Path,
                        help="Results directory — auto-discovers latest matching pair")
    parser.add_argument("--out-dir",     type=Path,
                        help="Output directory for report + plots "
                             "(defaults to --results-dir or parent of --gt)")
    parser.add_argument("--step",        type=float, default=0.5,
                        help="FLAC step size in seconds, used for timeline x-axis "
                             "(default: 0.5)")
    parser.add_argument("--no-plot",     action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    # ── Resolve input files ───────────────────────────────────────────────
    if args.results_dir:
        if not args.results_dir.exists():
            print(f"[ERROR] --results-dir not found: {args.results_dir}")
            sys.exit(1)
        gt_path, pred_path = _latest_pair(args.results_dir)
        out_dir = args.out_dir or args.results_dir
    elif args.gt and args.pred:
        gt_path, pred_path = args.gt, args.pred
        out_dir = args.out_dir or args.gt.parent
    else:
        print("[ERROR] Provide either --results-dir or both --gt and --pred.")
        sys.exit(1)

    for p, name in [(gt_path, "ground truth"), (pred_path, "predictions")]:
        if not p.exists():
            print(f"[ERROR] {name} file not found: {p}")
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Session tag from timestamp embedded in gt filename
    m = re.search(r'(\d{8}_\d{6})', gt_path.name)
    session = m.group(1) if m else datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\n  Ground truth : {gt_path.name}")
    print(f"  Predictions  : {pred_path.name}")
    gt   = np.load(gt_path)
    pred = np.load(pred_path)

    if len(gt) != len(pred):
        print(f"[ERROR] Length mismatch: gt={len(gt)}, pred={len(pred)}")
        sys.exit(1)

    # ── Compute ───────────────────────────────────────────────────────────
    node_ids, overall_acc, per_node, C_raw, C_norm, correct, valid = \
        compute_metrics(gt, pred)
    n_valid = int(valid.sum())

    # ── Report ────────────────────────────────────────────────────────────
    report = build_report(node_ids, overall_acc, per_node, C_raw, C_norm,
                          n_valid, gt_path, pred_path, session)
    print("\n" + report)

    rpt_path = out_dir / f"eval_report_{session}.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"\n  eval_report .txt → {rpt_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_confusion(C_norm, node_ids, out_dir, session)
        plot_timeline(correct, valid, args.step, out_dir, session)


if __name__ == "__main__":
    main()
