#!/usr/bin/env python3
"""
train_node_classifiers_pooled.py — Single shared LightGBM trained on ALL
node-timestep observations pooled together, with node_id as a categorical
feature (10-node IoBT).

Motivation
----------
Per-node v3 LOSO results show nodes 8 and 10 score ~0 recall because each
held-out fold has only 3–9 positive training examples for those nodes.
Pooling all nodes' data into one model lets sparse-positive nodes benefit
from the shared acoustic "car present" signal learned from data-rich nodes
(1, 7, 9).  Node identity is encoded as a LightGBM categorical feature so
the model also learns node-specific offsets.

Architecture
------------
  Features  : 20 acoustic (same v3 pipeline: z-score → rolling mean+std)
              + 1 node_id column (0-based int, column 20)
  Model     : single LGBMClassifier, scale_pos_weight = n_neg/n_pos
  Scaler    : one shared StandardScaler fitted on all pooled training data
              (acoustic columns 0–19 only; node_id column left as-is)
  Thresholds: per-node, tuned by F1 on training fold  (--tune-thresholds)

Data layout (same as train_node_classifiers_10.py)
---------------------------------------------------
  iobt_data_10/<YYYYMMDD_HHMMSS>/
      nodeN_respeaker.flac
      meta_data.json
      gt_tracks.csv

Outputs (to --out-dir)
----------------------
  pooled_clf_{ts}.pkl
  pooled_scaler_{ts}.pkl
  pooled_thresholds_{ts}.json    (only with --tune-thresholds)
  pooled_report_{ts}.txt

Usage
-----
  python collection/train_node_classifiers_pooled.py \\
      --data-dir iobt_data_10 --out-dir results/pooled --loso-only --compare-baseline

  python collection/train_node_classifiers_pooled.py \\
      --data-dir iobt_data_10 --out-dir results/pooled --tune-thresholds

  python collection/train_node_classifiers_pooled.py \\
      --data-dir iobt_data_10 --out-dir results/pooled --exclude-nodes 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Re-use all shared infrastructure from train_node_classifiers_10.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collection.train_node_classifiers_10 import (
    EXPECTED_NODE_IDS,
    DEFAULT_STEP_S,
    FEATURE_NAMES,
    WINDOW_STEPS,
    N_FEATURES,
    N_RAW_FEATURES,
    aggregate_rolling,
    build_node_features,
    discover_valid_sessions_10,
    load_session_raw_per_node_10,
    loso_per_node,
    train_final_classifiers,
    validate_session_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_ID_COL       = N_FEATURES      # column index of node_id feature = 20
N_FEATURES_POOLED = N_FEATURES + 1  # 21 total

POOLED_FEATURE_NAMES = FEATURE_NAMES + ["node_id"]

DEFAULT_THRESHOLD = 0.5  # pooled model calibrated differently from v3's 0.6


# ===========================================================================
# Dataset construction
# ===========================================================================

def build_pooled_features(
    session_data: dict,
    node_ids: list[int],
    window: int = WINDOW_STEPS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert per-session per-node raw features into a pooled feature matrix.

    session_data: {session_id: {nid: (raw_features (M,10), y (M,))}}

    For each (session, node) pair:
      1. Apply causal rolling mean+std → (M, 20) features
      2. Append node_idx (0-based) as column 20

    StandardScaler is NOT applied here — caller fits/applies the shared scaler
    on the acoustic columns 0–19.

    Returns
    -------
    X_roll21 : (N_total, 21) float64  — rolling features (unscaled) + node_id
    y        : (N_total,)   int32
    meta     : DataFrame[session, node_id, node_idx, row_in_session]
    """
    all_X, all_y, all_meta = [], [], []

    for session, node_dict in sorted(session_data.items()):
        for node_idx, nid in enumerate(node_ids):
            if nid not in node_dict:
                continue
            raw, y_bin = node_dict[nid]   # (M, 10), (M,)
            if len(raw) == 0:
                continue

            # Causal rolling (no scaler yet — scaler applied by caller)
            rolled = aggregate_rolling(raw, window)          # (M, 20)
            node_col = np.full((len(rolled), 1), node_idx, dtype=np.float64)
            X21 = np.hstack([rolled, node_col])              # (M, 21)

            all_X.append(X21)
            all_y.append(y_bin)
            all_meta.append(pd.DataFrame({
                "session":        session,
                "node_id":        nid,
                "node_idx":       node_idx,
                "row_in_session": np.arange(len(y_bin)),
            }))

    if not all_X:
        raise ValueError("build_pooled_features: no data found in session_data")

    X_roll21 = np.vstack(all_X)
    y_all    = np.concatenate(all_y).astype(np.int32)
    meta     = pd.concat(all_meta, ignore_index=True)

    n_pos = int(y_all.sum())
    print(f"    Pooled: {len(y_all):,} rows  |  "
          f"{n_pos:,} positive ({100*n_pos/len(y_all):.2f}%)  |  "
          f"sessions={len(session_data)}  nodes={len(node_ids)}")

    return X_roll21, y_all, meta


def _apply_shared_scaler(
    X_roll21: np.ndarray,
    scaler,
    fit: bool = False,
):
    """
    Scale acoustic columns 0–19 in-place using scaler.
    node_id column (20) is left as int, not scaled.

    If fit=True, calls scaler.fit_transform; else calls scaler.transform.
    Returns scaled (N, 21) array.
    """
    X = X_roll21.copy()
    if fit:
        X[:, :NODE_ID_COL] = scaler.fit_transform(X[:, :NODE_ID_COL])
    else:
        X[:, :NODE_ID_COL] = scaler.transform(X[:, :NODE_ID_COL])
    X[:, NODE_ID_COL] = X[:, NODE_ID_COL].astype(int)
    return X


# ===========================================================================
# Classifier training
# ===========================================================================

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """n_neg / n_pos, clamped to at least 1.0."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return max(1.0, n_neg / max(n_pos, 1))


def train_pooled_clf(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train a single shared LightGBM on pooled (N, 21) features.
    Column 20 must be int — treated as categorical node_id.

    categorical_feature is passed to fit() (not the constructor) as required
    by LightGBM's sklearn API.
    """
    import lightgbm as lgb

    spw = compute_scale_pos_weight(y_train)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    X = X_train.copy()
    X[:, NODE_ID_COL] = X[:, NODE_ID_COL].astype(int)
    clf.fit(
        pd.DataFrame(X, columns=POOLED_FEATURE_NAMES),
        y_train,
        categorical_feature=[NODE_ID_COL],
    )
    return clf


# ===========================================================================
# Per-node threshold tuning
# ===========================================================================

def tune_node_thresholds(
    clf,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    meta_val: pd.DataFrame,
    node_ids: list[int],
    verbose:  bool = True,
) -> dict[int, float]:
    """
    For each node, find the probability threshold that maximises F1 on
    the provided (validation / training-fold) predictions.

    Returns dict {node_id: best_threshold}.
    """
    from sklearn.metrics import f1_score

    thresholds = np.arange(0.10, 0.91, 0.05)
    node_thresholds: dict[int, float] = {}

    X = X_val.copy()
    X[:, NODE_ID_COL] = X[:, NODE_ID_COL].astype(int)
    proba_all = clf.predict_proba(
        pd.DataFrame(X, columns=POOLED_FEATURE_NAMES)
    )[:, 1]

    for node_idx, nid in enumerate(node_ids):
        mask = meta_val["node_id"].values == nid
        if not mask.any():
            node_thresholds[nid] = DEFAULT_THRESHOLD
            continue

        p = proba_all[mask]
        y = y_val[mask]

        if y.sum() == 0:
            # No positive examples — keep default
            node_thresholds[nid] = DEFAULT_THRESHOLD
            continue

        best_t, best_f1 = DEFAULT_THRESHOLD, 0.0
        for t in thresholds:
            pred = (p >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        node_thresholds[nid] = best_t
        if verbose:
            print(f"    node{nid:>2}: thresh={best_t:.2f}  train_f1={best_f1:.3f}")

    return node_thresholds


# ===========================================================================
# Top-k helper for the pooled representation
# ===========================================================================

def _top_k_from_pooled(
    proba_flat: np.ndarray,
    y_flat:     np.ndarray,
    meta:       pd.DataFrame,
    k:          int = 3,
) -> tuple[int, int]:
    """
    Compute top-k accuracy from the pooled (timestep × node) representation.

    proba_flat : (N_rows,) probabilities aligned to meta rows
    y_flat     : (N_rows,) binary labels
    meta       : DataFrame with column "row_in_session" (same index as proba_flat)
    k          : number of top candidates

    At each unique row_in_session, ranks nodes by descending probability and
    checks whether the GT node (y==1) is among the top-k.

    Returns (correct_steps, total_steps).
    """
    correct = 0
    total   = 0
    for _, group in meta.groupby("row_in_session"):
        idx = group.index.values
        p   = proba_flat[idx]
        y   = y_flat[idx]
        if y.sum() == 0:
            continue
        gt_local = int(np.argmax(y))           # local index of the GT node
        top_k    = np.argsort(p)[::-1][:k]
        if gt_local in top_k:
            correct += 1
        total += 1
    return correct, total


# ===========================================================================
# LOSO cross-validation (pooled)
# ===========================================================================

def fit_node_calibrators(
    fold_calib_data: dict,
    node_ids:        list[int],
    verbose:         bool = True,
) -> dict:
    """
    Fit one IsotonicRegression per node using the LOSO held-out proba/y data.

    fold_calib_data: {held_out_session: {"proba": (N,), "y": (N,), "meta": DataFrame}}

    Returns dict {node_id: fitted IsotonicRegression}
    """
    from sklearn.isotonic import IsotonicRegression

    if verbose:
        print("\n  Fitting isotonic calibrators from LOSO held-out probabilities …")
        print(f"  {'Node':>6}  {'N':>6}  {'pos%':>6}  {'raw pos mean':>13}  {'cal pos mean':>13}  {'raw neg mean':>13}  {'cal neg mean':>13}")

    calibrators: dict = {}
    for nid in node_ids:
        all_p, all_y = [], []
        for fold_data in fold_calib_data.values():
            mask = fold_data["meta"]["node_id"].values == nid
            if mask.any():
                all_p.append(fold_data["proba"][mask])
                all_y.append(fold_data["y"][mask])

        if not all_p:
            calibrators[nid] = None
            continue

        p = np.concatenate(all_p)
        y = np.concatenate(all_y).astype(int)

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        calibrators[nid] = ir

        if verbose:
            p_cal = ir.predict(p)
            pos = y == 1
            neg = y == 0
            raw_pos = p[pos].mean()   if pos.any()  else float("nan")
            raw_neg = p[neg].mean()   if neg.any()  else float("nan")
            cal_pos = p_cal[pos].mean() if pos.any() else float("nan")
            cal_neg = p_cal[neg].mean() if neg.any() else float("nan")
            pct_pos = 100 * pos.mean()
            print(f"  node{nid:>2}:  {len(p):>6}  {pct_pos:>5.1f}%  "
                  f"{raw_pos:>13.4f}  {cal_pos:>13.4f}  "
                  f"{raw_neg:>13.4f}  {cal_neg:>13.4f}")

    return calibrators


def loso_pooled(
    session_data:    dict,
    node_ids:        list[int],
    sessions:        list[str],
    window:          int   = WINDOW_STEPS,
    do_tune:         bool  = False,
    fixed_threshold: float = DEFAULT_THRESHOLD,
    top_k:           int   = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Leave-One-Session-Out cross-validation for the pooled classifier.

    For each held-out session:
      1. Pool remaining sessions → build rolling features + node_id
      2. Fit shared StandardScaler on acoustic columns (training fold only)
      3. Train single shared LightGBM
      4. Optionally tune per-node thresholds on training fold
      5. Evaluate per-node on held-out session

    Returns
    -------
    df              : DataFrame with columns:
                      held_out, node_id, recall, precision, f1, gt_steps,
                      threshold, top_k_accuracy, top_k_k
                      (top_k_accuracy is a session-level metric — same value
                      for all node rows within a held-out fold.)
    fold_calib_data : dict {held_out_session: {"proba": array, "y": array, "meta": DataFrame}}
                      — held-out probabilities for fitting isotonic calibrators.
    """
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler

    all_rows        = []
    fold_calib_data = {}

    print("\n" + "=" * 68)
    print("  POOLED LOSO CROSS-VALIDATION")
    print("=" * 68)

    for held_out in sessions:
        train_sessions = [s for s in sessions if s != held_out]
        print(f"\n  Held-out: {held_out}  |  Training on {len(train_sessions)} sessions")

        # ── Training fold ────────────────────────────────────────────────────
        train_data = {s: session_data[s] for s in train_sessions}
        print("  Building training features …")
        X_raw_tr, y_tr, meta_tr = build_pooled_features(train_data, node_ids, window)

        scaler = StandardScaler()
        X_tr   = _apply_shared_scaler(X_raw_tr, scaler, fit=True)

        print("  Training pooled classifier …")
        clf = train_pooled_clf(X_tr, y_tr)

        # ── Optional threshold tuning on training fold ────────────────────────
        if do_tune:
            print("  Tuning per-node thresholds on training fold …")
            node_thresholds = tune_node_thresholds(clf, X_tr, y_tr, meta_tr, node_ids)
        else:
            node_thresholds = {nid: fixed_threshold for nid in node_ids}

        # ── Test fold ────────────────────────────────────────────────────────
        held_data = {held_out: session_data[held_out]}
        X_raw_te, y_te, meta_te = build_pooled_features(held_data, node_ids, window)
        X_te = _apply_shared_scaler(X_raw_te, scaler, fit=False)

        X_te_df = pd.DataFrame(X_te.copy(), columns=POOLED_FEATURE_NAMES)
        X_te_df[POOLED_FEATURE_NAMES[-1]] = X_te_df[POOLED_FEATURE_NAMES[-1]].astype(int)
        proba_te = clf.predict_proba(X_te_df)[:, 1]

        # ── Store held-out probabilities for calibration ─────────────────────
        fold_calib_data[held_out] = {
            "proba": proba_te,
            "y":     y_te,
            "meta":  meta_te,
        }

        # ── Session-level top-k accuracy ─────────────────────────────────────
        topk_correct, topk_total = _top_k_from_pooled(proba_te, y_te, meta_te, top_k)
        topk_acc = topk_correct / topk_total if topk_total > 0 else 0.0

        # ── Per-node metrics ─────────────────────────────────────────────────
        node_strs = []
        for node_idx, nid in enumerate(node_ids):
            mask = meta_te["node_id"].values == nid
            if not mask.any():
                continue

            thresh = node_thresholds.get(nid, fixed_threshold)
            p   = proba_te[mask]
            y   = y_te[mask]
            gt  = int(y.sum())
            pred = (p >= thresh).astype(int)

            prec, rec, f1, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0
            )
            node_strs.append(f"n{nid}=F1:{f1:.2f}/R:{rec:.2f}")
            all_rows.append({
                "held_out":       held_out,
                "node_id":        nid,
                "recall":         float(rec),
                "precision":      float(prec),
                "f1":             float(f1),
                "gt_steps":       gt,
                "threshold":      float(thresh),
                "top_k_accuracy": topk_acc,
                "top_k_k":        top_k,
            })

        print(f"    " + "  ".join(node_strs))

    df = pd.DataFrame(all_rows)

    print("\n  Mean LOSO recall per node (pooled):")
    for nid in node_ids:
        sub = df[df["node_id"] == nid]
        if sub.empty:
            continue
        recalls = sub["recall"].values
        print(f"    node {nid:>2}: {recalls.mean():.3f}  "
              f"({', '.join(f'{r:.2f}' for r in recalls)})")

    print(f"\n  LOSO top-{top_k} accuracy per held-out fold "
          f"(GT in top-{top_k} by probability):")
    fold_accs = []
    for sess in sessions:
        sub = df[df["held_out"] == sess]
        if sub.empty:
            continue
        acc   = sub["top_k_accuracy"].iloc[0]
        total = sub["gt_steps"].sum()           # total GPS-covered steps in fold
        fold_accs.append(acc)
        print(f"    {sess}: {acc:.3f}  ({int(round(acc * total))}/{total} steps)")
    if fold_accs:
        print(f"    overall mean: {np.mean(fold_accs):.3f}")
    print("=" * 68)

    return df, fold_calib_data


# ===========================================================================
# Comparison table: pooled vs v3 baseline
# ===========================================================================

def compare_vs_baseline(
    pooled_df:    pd.DataFrame,
    v3_results:   dict,
    node_ids:     list[int],
    sessions:     list[str],
) -> None:
    """
    Print side-by-side LOSO comparison: pooled vs per-node v3 baseline.

    v3_results: output of loso_per_node() — {held_out: {nid: {recall, f1, ...}}}
    """
    k = int(pooled_df["top_k_k"].iloc[0]) if "top_k_k" in pooled_df.columns else 3

    print("\n" + "=" * 80)
    print("  POOLED vs PER-NODE v3  —  Mean LOSO recall across all held-out folds")
    print("=" * 80)
    print(f"  {'Node':<6} {'GT':>5}  "
          f"{'Pool-R':>8} {'Top-'+str(k):>6} {'Pool-F1':>8}  "
          f"{'Base-R':>8} {'Base-F1':>8}  {'ΔRecall':>8}")
    print("  " + "-" * 76)

    for nid in node_ids:
        sub = pooled_df[pooled_df["node_id"] == nid]
        gt_mean = sub["gt_steps"].mean() if not sub.empty else 0
        pr  = sub["recall"].mean()       if not sub.empty else 0.0
        pf  = sub["f1"].mean()           if not sub.empty else 0.0
        ptk = sub["top_k_accuracy"].mean() if ("top_k_accuracy" in sub.columns and not sub.empty) else float("nan")

        br_list = [v3_results[s][nid]["recall"] for s in sessions if nid in v3_results.get(s, {})]
        bf_list = [v3_results[s][nid]["f1"]     for s in sessions if nid in v3_results.get(s, {})]
        br = float(np.mean(br_list)) if br_list else 0.0
        bf = float(np.mean(bf_list)) if bf_list else 0.0

        delta = pr - br
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
        tk_str = f"{ptk:.3f}" if ptk == ptk else "  n/a"
        print(f"  {nid:<6} {gt_mean:>5.0f}  "
              f"{pr:>8.3f} {tk_str:>6} {pf:>8.3f}  "
              f"{br:>8.3f} {bf:>8.3f}  "
              f"{arrow}{abs(delta):>6.3f}")

    # Summary: mean across all nodes
    all_pool_r  = pooled_df.groupby("node_id")["recall"].mean().mean()
    all_pool_tk = pooled_df.groupby("held_out")["top_k_accuracy"].first().mean() \
                  if "top_k_accuracy" in pooled_df.columns else float("nan")
    all_base_r = np.mean([
        v3_results[s][nid]["recall"]
        for s in sessions
        for nid in node_ids
        if nid in v3_results.get(s, {})
    ])
    tk_all_str = f"{all_pool_tk:.3f}" if all_pool_tk == all_pool_tk else "  n/a"
    print("  " + "-" * 76)
    print(f"  {'ALL':<6} {'':>5}  "
          f"{all_pool_r:>8.3f} {tk_all_str:>6} {'':>8}  "
          f"{all_base_r:>8.3f} {'':>8}  "
          f"{'↑' if all_pool_r > all_base_r else '↓'}"
          f"{abs(all_pool_r - all_base_r):>6.3f}")
    print("=" * 80)


# ===========================================================================
# Production training
# ===========================================================================

def train_production_model(
    session_data:    dict,
    node_ids:        list[int],
    out_dir:         Path,
    ts:              str,
    window:          int  = WINDOW_STEPS,
    do_tune:         bool = False,
    fold_calib_data: dict = None,
) -> tuple:
    """
    Train the final production pooled classifier on ALL sessions.

    Saves:
        pooled_clf_{ts}.pkl
        pooled_scaler_{ts}.pkl
        pooled_thresholds_{ts}.json      (only if do_tune=True)
        pooled_calibrators_{ts}.pkl      (always — uses fold_calib_data if provided,
                                          else fits on training data directly)
        pooled_report_{ts}.txt

    Returns (clf, scaler, node_thresholds, calibrators)
    """
    import joblib
    from sklearn.preprocessing import StandardScaler

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Building pooled training dataset (all sessions) …")
    X_raw, y_all, meta = build_pooled_features(session_data, node_ids, window)

    print("  Fitting shared StandardScaler …")
    scaler = StandardScaler()
    X_scaled = _apply_shared_scaler(X_raw, scaler, fit=True)

    print("  Training production classifier …")
    clf = train_pooled_clf(X_scaled, y_all)

    # Per-node threshold tuning (on all data — production estimate only)
    if do_tune:
        print("  Tuning per-node thresholds …")
        node_thresholds = tune_node_thresholds(clf, X_scaled, y_all, meta, node_ids)
    else:
        node_thresholds = {nid: DEFAULT_THRESHOLD for nid in node_ids}

    # ── Fit calibrators ───────────────────────────────────────────────────────
    if fold_calib_data is not None:
        print("  Fitting isotonic calibrators from LOSO held-out data …")
        calibrators = fit_node_calibrators(fold_calib_data, node_ids)
    else:
        # Fallback: calibrate on training data (optimistic — use LOSO if available)
        print("  Fitting isotonic calibrators on training data (no LOSO data provided) …")
        X_df_all = pd.DataFrame(X_scaled.copy(), columns=POOLED_FEATURE_NAMES)
        X_df_all[POOLED_FEATURE_NAMES[-1]] = X_df_all[POOLED_FEATURE_NAMES[-1]].astype(int)
        proba_train = clf.predict_proba(X_df_all)[:, 1]
        fold_calib_train = {"all_sessions": {"proba": proba_train, "y": y_all, "meta": meta}}
        calibrators = fit_node_calibrators(fold_calib_train, node_ids)

    # Save artefacts
    clf_path    = out_dir / f"pooled_clf_{ts}.pkl"
    scaler_path = out_dir / f"pooled_scaler_{ts}.pkl"
    calib_path  = out_dir / f"pooled_calibrators_{ts}.pkl"
    joblib.dump(clf,         clf_path)
    joblib.dump(scaler,      scaler_path)
    joblib.dump(calibrators, calib_path)
    print(f"  Saved: {clf_path.name}")
    print(f"  Saved: {scaler_path.name}")
    print(f"  Saved: {calib_path.name}")

    if do_tune:
        thresh_path = out_dir / f"pooled_thresholds_{ts}.json"
        with open(thresh_path, "w") as f:
            json.dump({str(k): v for k, v in node_thresholds.items()}, f, indent=2)
        print(f"  Saved: {thresh_path.name}")

    # Report
    lines = [
        "=" * 68,
        "  POOLED NODE CLASSIFIER — 10-NODE IoBT",
        "=" * 68,
        f"  Created    : {datetime.now().isoformat()}",
        f"  Features   : 21 (20 acoustic v3 + node_id categorical)",
        f"  Scaler     : shared StandardScaler (acoustic cols 0–19)",
        f"  Classifier : LightGBM (n_estimators=300, num_leaves=63, "
        f"scale_pos_weight={compute_scale_pos_weight(y_all):.1f})",
        f"  Sessions   : {sorted(session_data.keys())}",
        f"  Total rows : {len(y_all):,}  positive: {int(y_all.sum()):,} "
        f"({100*y_all.mean():.2f}%)",
        "",
        "  Per-node positive counts (across all sessions):",
    ]
    for nid in node_ids:
        sub = meta[meta["node_id"] == nid]
        n_tot = len(sub)
        n_pos = int(y_all[sub.index].sum()) if n_tot > 0 else 0
        lines.append(f"    node {nid:>2}: {n_pos:>4} pos / {n_tot:>5} total "
                     f"({100*n_pos/max(n_tot,1):.1f}%)  "
                     f"thresh={node_thresholds.get(nid, DEFAULT_THRESHOLD):.2f}")

    report = "\n".join(lines)
    print("\n" + report)
    rpt_path = out_dir / f"pooled_report_{ts}.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"\n  Report → {rpt_path}")

    return clf, scaler, node_thresholds, calibrators


# ===========================================================================
# Inference helper
# ===========================================================================

def build_p_audio_pooled(
    clf,
    scaler,
    session_dir:  Path,
    node_ids:     list[int],
    step_s:       float = DEFAULT_STEP_S,
    window:       int   = WINDOW_STEPS,
    n_timesteps:  int   = None,
) -> np.ndarray:
    """
    Build (T, N) audio probability matrix using the shared pooled classifier.

    For each node k:
      1. build_node_features(node_k_flac)  → raw (T, 10)
      2. scaler.transform                  → scaled (T, 10)
      3. aggregate_rolling                 → rolled (T, 20)
      4. append node_idx column            → (T, 21)
      5. predict_proba                     → P_audio[:, k]

    Per-node isolation is preserved: column k is computed using only node k's
    own audio features. node_id is the only cross-node information.

    Returns
    -------
    P_audio : (T, N) float32
    """
    # Determine T from first available FLAC
    T = None
    raw_cache = {}
    for node_idx, nid in enumerate(node_ids):
        flac_path = session_dir / f"node{nid}_respeaker.flac"
        if not flac_path.exists():
            continue
        raw, T_node = build_node_features(flac_path, step_s)
        raw_cache[(node_idx, nid)] = (raw, T_node)
        if T is None:
            T = T_node

    if T is None:
        raise ValueError(f"No FLAC files found in {session_dir}")
    if n_timesteps is not None:
        T = min(T, n_timesteps)

    N       = len(node_ids)
    P_audio = np.zeros((T, N), dtype=np.float32)

    for node_idx, nid in enumerate(node_ids):
        if (node_idx, nid) not in raw_cache:
            continue
        raw, _ = raw_cache[(node_idx, nid)]
        raw_T  = raw[:T]                                  # (T, 10)

        # Order must match train-time: roll first, then scale.
        # The shared scaler was fitted on 20-col rolled features (not raw 10).
        rolled = aggregate_rolling(raw_T, window)         # (T, 20)
        scaled = scaler.transform(rolled)                 # (T, 20)  shared scaler

        node_col = np.full((T, 1), node_idx, dtype=np.float64)
        X21 = np.hstack([rolled, node_col])               # (T, 21)
        X21[:, NODE_ID_COL] = X21[:, NODE_ID_COL].astype(int)

        proba = clf.predict_proba(
            pd.DataFrame(X21, columns=POOLED_FEATURE_NAMES)
        )[:, 1]
        P_audio[:, node_idx] = proba.astype(np.float32)

    return P_audio


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pooled cross-node audio classifier (10-node IoBT).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # LOSO eval + comparison vs per-node v3 baseline
  python collection/train_node_classifiers_pooled.py \\
      --data-dir iobt_data_10 --out-dir results/pooled \\
      --loso-only --compare-baseline

  # Full training with threshold tuning
  python collection/train_node_classifiers_pooled.py \\
      --data-dir iobt_data_10 --out-dir results/pooled \\
      --tune-thresholds

  # Exclude node 2 from pooled model (train separately with v3)
  python collection/train_node_classifiers_pooled.py \\
      --data-dir iobt_data_10 --out-dir results/pooled \\
      --loso-only --compare-baseline --exclude-nodes 2
        """,
    )
    _DEFAULT_DATA = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iobt_data_10"
    )
    parser.add_argument("--data-dir",         type=Path, default=Path(_DEFAULT_DATA))
    parser.add_argument("--out-dir",          type=Path, default=Path("results/pooled"))
    parser.add_argument("--step",             type=float, default=DEFAULT_STEP_S)
    parser.add_argument("--threshold",        type=float, default=DEFAULT_THRESHOLD,
                        help=f"Fixed probability threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--validate-only",    action="store_true")
    parser.add_argument("--loso-only",        action="store_true",
                        help="Run LOSO only; skip saving production classifier")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Also run v3 per-node LOSO and print side-by-side ΔRecall")
    parser.add_argument("--tune-thresholds",  action="store_true",
                        help="Tune per-node thresholds by F1 on training fold")
    parser.add_argument("--top-k",            type=int, default=3,
                        help="k for top-k accuracy in LOSO (default: 3)")
    parser.add_argument("--exclude-nodes",    type=int, nargs="+", default=[],
                        metavar="N",
                        help="Exclude these node IDs from pooled training; "
                             "train them with v3 per-node clf instead")
    args = parser.parse_args()

    print("=" * 68)
    print("  POOLED CROSS-NODE CLASSIFIER — 10-NODE IoBT")
    print("=" * 68)
    print(f"  Data dir          : {args.data_dir}")
    print(f"  Out dir           : {args.out_dir}")
    print(f"  Step size         : {args.step}s")
    print(f"  Fixed threshold   : {args.threshold}")
    print(f"  Tune thresholds   : {args.tune_thresholds}")
    print(f"  Compare baseline  : {args.compare_baseline}")
    print(f"  Excluded nodes    : {args.exclude_nodes or 'none'}")

    pooled_node_ids = [n for n in EXPECTED_NODE_IDS if n not in args.exclude_nodes]
    excluded_ids    = [n for n in EXPECTED_NODE_IDS if n in args.exclude_nodes]
    print(f"  Pooled nodes      : {pooled_node_ids}")
    if excluded_ids:
        print(f"  Separate v3 nodes : {excluded_ids}")

    # [0] Discover sessions
    print(f"\n[0/4] Discovering valid sessions in {args.data_dir} …")
    sessions = discover_valid_sessions_10(args.data_dir)
    if not sessions:
        print("[ERROR] No valid sessions found.")
        sys.exit(1)
    print(f"  Found {len(sessions)} session(s): {sessions}")

    # [1] Validate
    print(f"\n[1/4] Validating sessions …")
    all_ok = True
    for sess in sessions:
        ok = validate_session_data(args.data_dir / sess)
        if not ok:
            all_ok = False
    if not all_ok:
        print("\n[ERROR] One or more sessions failed validation.")
        sys.exit(1)
    if args.validate_only:
        print("\n[--validate-only] Done.")
        sys.exit(0)

    # [2] Load raw features
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[2/4] Loading per-node raw features …")
    session_data: dict = {}
    for sess in sessions:
        print(f"\n  Session: {sess}")
        session_data[sess] = load_session_raw_per_node_10(
            args.data_dir / sess, args.step
        )

    # [3] LOSO evaluation
    print(f"\n[3/4] Running LOSO cross-validation …")

    fold_calib_data: dict = {}

    if len(sessions) < 2:
        print("  [WARN] Need ≥ 2 sessions for LOSO. Skipping.")
        loso_df    = pd.DataFrame()
        v3_results = {}
    else:
        loso_df, fold_calib_data = loso_pooled(
            session_data, pooled_node_ids, sessions,
            window=WINDOW_STEPS,
            do_tune=args.tune_thresholds,
            fixed_threshold=args.threshold,
            top_k=args.top_k,
        )

        if args.compare_baseline:
            print(f"\n  Running v3 per-node LOSO for comparison …")
            v3_results = loso_per_node(
                session_data, EXPECTED_NODE_IDS, sessions, window=WINDOW_STEPS
            )
            compare_vs_baseline(loso_df, v3_results, pooled_node_ids, sessions)
        else:
            v3_results = {}

    # [4] Production training (or calibrator-only save if --loso-only)
    if args.loso_only:
        print(f"\n[4/4] --loso-only: skipping production classifier training.")
        # Still fit and save calibrators from LOSO held-out probabilities
        if fold_calib_data:
            import joblib
            calibrators = fit_node_calibrators(fold_calib_data, pooled_node_ids)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            calib_path = args.out_dir / f"pooled_calibrators_{ts}.pkl"
            joblib.dump(calibrators, calib_path)
            print(f"  Saved: {calib_path.name}")
        else:
            print("  [WARN] No fold_calib_data — calibrators not saved.")
    else:
        print(f"\n[4/4] Training production classifier on all sessions …")
        clf, scaler, node_thresholds, calibrators = train_production_model(
            session_data, pooled_node_ids, args.out_dir, ts,
            window=WINDOW_STEPS, do_tune=args.tune_thresholds,
            fold_calib_data=fold_calib_data if fold_calib_data else None,
        )

        # If any nodes are excluded, train separate v3 per-node classifiers
        if excluded_ids:
            print(f"\n  Training separate v3 classifiers for excluded nodes: {excluded_ids}")
            excluded_data = {
                s: {nid: session_data[s][nid] for nid in excluded_ids if nid in session_data[s]}
                for s in sessions
            }
            excluded_clfs = train_final_classifiers(
                excluded_data, excluded_ids, sessions, window=WINDOW_STEPS
            )
            import joblib
            for nid, (clf_v3, scaler_v3) in excluded_clfs.items():
                cp = args.out_dir / f"excluded_node{nid}_clf_{ts}.pkl"
                sp = args.out_dir / f"excluded_node{nid}_scaler_{ts}.pkl"
                joblib.dump(clf_v3,    cp)
                joblib.dump(scaler_v3, sp)
                print(f"  Saved: {cp.name}  (v3 clf for node {nid})")
                print(f"  Saved: {sp.name}")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)

    if not args.loso_only:
        print(f"\n  Load for inference:")
        print(f"    import joblib")
        print(f"    from collection.train_node_classifiers_pooled import build_p_audio_pooled")
        print(f"    clf         = joblib.load('results/pooled/pooled_clf_{ts}.pkl')")
        print(f"    scaler      = joblib.load('results/pooled/pooled_scaler_{ts}.pkl')")
        print(f"    calibrators = joblib.load('results/pooled/pooled_calibrators_{ts}.pkl')")
        print(f"    P_audio = build_p_audio_pooled(clf, scaler, session_dir, node_ids)")


if __name__ == "__main__":
    main()
