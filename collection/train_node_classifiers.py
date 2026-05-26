#!/usr/bin/env python3
"""
train_node_classifiers.py — Train one binary classifier per acoustic node.

Each classifier answers "is the person at my node right now?" using only that
node's own audio features (rolling mean + std of received power in dB).  No
inter-node information is used, so the classifier can run independently on
a node that has been activated without requiring data from deactivated nodes.

This enables energy-honest detection in Track-MDP: a node must be both
activated by the policy AND predict the person present for a detection to
occur.

Outputs (per node, saved to --out-dir):
    node_clf_N{nid}_{SESSION}.pkl   — fitted binary sklearn classifier
    node_clf_report_{SESSION}.txt   — per-node precision / recall / F1 table

Usage
-----
    python collection/train_node_classifiers.py \\
        --flac-dir  iobt_data \\
        --gps-csv   iobt_data/gps.csv \\
        --nodes-txt iobt_data/node_positions.txt \\
        --session   20260417_100634 \\
        --window    5 \\
        --out-dir   results/20260417_100634
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from collection.supervised_transition import (
    build_ground_truth_sequence,
    apply_rolling_mean,
    apply_rolling_std,
    _parse_session_start,
)
from collection.flac_to_trackmdp import (
    discover_flac_files,
    compute_power_per_step,
    load_node_positions_ordered,
    load_gps_track,
)


# ---------------------------------------------------------------------------
# Per-node features
# ---------------------------------------------------------------------------

def build_node_features(power_array: np.ndarray, window: int) -> np.ndarray:
    """
    Build a (T, 2) feature matrix for a single node using only that node's
    own power signal.

    Features:
        col 0 — causal rolling mean of dB power (window steps)
        col 1 — causal rolling std  of dB power (window steps)

    Parameters
    ----------
    power_array : (T,) float array of linear acoustic power values
    window      : rolling-window length in timesteps

    Returns
    -------
    (T, 2) float64 array
    """
    db = 10.0 * np.log10(np.maximum(power_array, 1e-12))  # (T,)
    X  = db.reshape(-1, 1)                                  # (T, 1)
    mean = apply_rolling_mean(X, window)                    # (T, 1)
    std  = apply_rolling_std(X, window)                     # (T, 1)
    return np.concatenate([mean, std], axis=1)              # (T, 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_node_classifiers(
    power_arrays: dict,
    node_ids:     list[int],
    gt:           np.ndarray,
    window:       int,
) -> dict:
    """
    Train one binary RandomForest classifier per node.

    Returns
    -------
    dict mapping node_id -> {'clf': fitted_clf, 'metrics': {precision,recall,f1,support}}
    """
    try:
        import joblib as _joblib  # noqa: F401 — confirm available
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)

    T         = min(len(a) for a in power_arrays.values())
    valid_mask = gt > 0
    valid_idxs = np.where(valid_mask)[0]
    n_valid    = len(valid_idxs)

    split      = int(n_valid * 0.70)
    train_idxs = valid_idxs[:split]
    test_idxs  = valid_idxs[split:]

    print(f"  Hold-out split (70/30): {len(train_idxs)} train / "
          f"{len(test_idxs)} test GPS-labelled steps")

    results = {}

    for nid in node_ids:
        X = build_node_features(power_arrays[nid][:T], window)  # (T, 2)

        # Binary label: 1 if person is at this node, 0 otherwise
        y = (gt == nid).astype(np.int32)

        X_tr, y_tr = X[train_idxs], y[train_idxs]
        X_te, y_te = X[test_idxs],  y[test_idxs]

        pos_rate = y_tr.mean()
        print(f"\n  Node {nid}: positive rate = {pos_rate:.3f} "
              f"({y_tr.sum()} positive / {len(y_tr)} train steps)")

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",  # handles class imbalance
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)

        # Held-out evaluation
        y_pred = clf.predict(X_te)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="binary", zero_division=0
        )
        sup = int(y_te.sum())   # number of positive instances in test set
        acc = (y_pred == y_te).mean()
        print(f"    Hold-out: acc={acc:.3f}  prec={prec:.3f}  "
              f"rec={rec:.3f}  F1={f1:.3f}  support={sup}")

        # Retrain on all GPS-labelled data for final model
        X_all = X[valid_idxs]
        y_all = y[valid_idxs]
        clf.fit(X_all, y_all)

        results[nid] = {
            "clf":     clf,
            "metrics": dict(accuracy=float(acc), precision=float(prec),
                            recall=float(rec), f1=float(f1), support=int(sup)),
        }

    return results


# ---------------------------------------------------------------------------
# Full-sequence GPS correspondence evaluation
# ---------------------------------------------------------------------------

def evaluate_full_sequence(
    results: dict,
    power_arrays: dict,
    node_ids: list[int],
    gt: np.ndarray,
    window: int,
) -> None:
    """
    Evaluate each per-node classifier against GPS ground truth on ALL
    GPS-labelled timesteps (not just the held-out 30%).

    Also reports the coverage rate: the fraction of GPS-labelled timesteps
    where the correct node's classifier outputs a positive prediction.
    This is the upper bound on detection recall under binary gating regardless
    of which nodes the RL policy activates.
    """
    from sklearn.metrics import precision_recall_fscore_support

    T      = min(len(a) for a in power_arrays.values())
    valid  = gt > 0
    n_valid = int(valid.sum())

    print("\n" + "=" * 64)
    print("  FULL-SEQUENCE GPS CORRESPONDENCE")
    print(f"  ({n_valid} GPS-labelled steps out of {T} total)")
    print("=" * 64)
    print(f"  {'node':>6}  {'recall':>7}  {'precision':>10}  "
          f"{'F1':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>6}")
    print("  " + "-" * 60)

    node_id_to_col = {nid: i for i, nid in enumerate(node_ids)}
    covered = np.zeros(n_valid, dtype=bool)
    valid_idxs = np.where(valid)[0]

    for nid, res in sorted(results.items()):
        clf   = res["clf"]
        X     = build_node_features(power_arrays[nid][:T], window)
        y     = (gt == nid).astype(np.int32)

        X_val = X[valid]
        y_val = y[valid]
        preds = clf.predict(X_val)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0
        )
        TP = int(((preds == 1) & (y_val == 1)).sum())
        FP = int(((preds == 1) & (y_val == 0)).sum())
        FN = int(((preds == 0) & (y_val == 1)).sum())
        TN = int(((preds == 0) & (y_val == 0)).sum())

        print(f"  {nid:>6}  {rec:>7.3f}  {prec:>10.3f}  "
              f"{f1:>7.3f}  {TP:>6}  {FP:>6}  {FN:>6}  {TN:>6}")

        col = node_id_to_col[nid]
        for i, t in enumerate(valid_idxs):
            if gt[t] == nid:
                covered[i] = (preds[i] == 1)

    coverage = covered.mean()
    print("  " + "-" * 60)
    print(f"\n  Correct-node coverage : {coverage:.3f}  "
          f"({coverage*100:.1f}% of GPS steps where clf agreed with GPS)")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_classifiers(results: dict, out_dir: Path, session: str) -> None:
    import joblib

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        "=" * 56,
        "  PER-NODE BINARY CLASSIFIER REPORT",
        "=" * 56,
        f"  Session : {session}",
        f"  Created : {datetime.now().isoformat()}",
        "",
        f"  {'node':>6}  {'accuracy':>9}  {'precision':>10}  "
        f"{'recall':>7}  {'F1':>7}  {'support':>8}",
        "  " + "-" * 52,
    ]

    for nid, res in sorted(results.items()):
        m = res["metrics"]
        pkl_path = out_dir / f"node_clf_N{nid}_{session}.pkl"
        joblib.dump(res["clf"], pkl_path)
        print(f"  Saved: {pkl_path.name}")

        lines.append(
            f"  {nid:>6}  {m['accuracy']*100:>8.1f}%  "
            f"{m['precision']:>10.3f}  {m['recall']:>7.3f}  "
            f"{m['f1']:>7.3f}  {m['support']:>8d}"
        )

    lines.append("=" * 56)
    report = "\n".join(lines)
    print("\n" + report)

    rpt_path = out_dir / f"node_clf_report_{session}.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"\n  Report → {rpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one binary classifier per acoustic node.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _D = os.path.join(project_root, "iobt_data")
    parser.add_argument("--flac-dir",  type=Path,
                        default=Path(_D))
    parser.add_argument("--gps-csv",   type=Path,
                        default=Path(os.path.join(_D, "20260417_100634_gps2_gps.csv")))
    parser.add_argument("--nodes-txt", type=Path,
                        default=Path(os.path.join(_D, "node_positions.txt")))
    parser.add_argument("--session",   type=str, default="20260417_100634")
    parser.add_argument("--window",    type=int, default=5,
                        help="Rolling-mean window in timesteps (default: 5)")
    parser.add_argument("--step",      type=float, default=0.5,
                        help="FLAC step size in seconds (default: 0.5)")
    parser.add_argument("--out-dir",   type=Path,
                        default=Path("results"))
    args = parser.parse_args()

    print("=" * 56)
    print("  PER-NODE BINARY CLASSIFIER TRAINING")
    print("=" * 56)
    print(f"  Session   : {args.session}")
    print(f"  FLAC dir  : {args.flac_dir}")
    print(f"  GPS csv   : {args.gps_csv}")
    print(f"  Nodes txt : {args.nodes_txt}")
    print(f"  Window    : {args.window} steps ({args.window * args.step:.1f}s)")
    print(f"  Out dir   : {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── FLAC discovery & power extraction ────────────────────────────────
    print(f"\n[1/4] Discovering FLAC files …")
    file_map = discover_flac_files(args.flac_dir, session=args.session)
    if not file_map:
        print(f"[ERROR] No FLAC files found for session '{args.session}' "
              f"in {args.flac_dir}")
        sys.exit(1)
    node_ids = sorted(file_map.keys())
    print(f"  Found {len(node_ids)} nodes: {node_ids}")

    print(f"\n[2/4] Computing per-node power (step={args.step}s) …")
    power_arrays = {}
    for nid, fpath in file_map.items():
        pwr = compute_power_per_step(fpath, args.step)
        power_arrays[nid] = pwr
        print(f"  node {nid}: {len(pwr)} steps")
    T = min(len(a) for a in power_arrays.values())

    # ── GPS ground truth ──────────────────────────────────────────────────
    print(f"\n[3/4] Building GPS ground truth …")
    node_xy    = load_node_positions_ordered(args.nodes_txt, node_ids)
    gps_df     = load_gps_track(args.gps_csv)
    flac_start = _parse_session_start(args.session)
    gt = build_ground_truth_sequence(
        gps_df, node_xy, node_ids, flac_start, args.step, T
    )
    n_valid = int((gt > 0).sum())
    print(f"  {n_valid}/{T} steps have GPS coverage "
          f"({n_valid/T*100:.1f}%)")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] Training binary classifiers (window={args.window}) …")
    results = train_node_classifiers(power_arrays, node_ids, gt, args.window)

    # ── Save ──────────────────────────────────────────────────────────────
    save_classifiers(results, args.out_dir, args.session)

    # ── Full-sequence GPS correspondence ───────────────────────────────────
    evaluate_full_sequence(results, power_arrays, node_ids, gt, args.window)


if __name__ == "__main__":
    main()
