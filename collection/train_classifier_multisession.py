#!/usr/bin/env python3
"""
train_classifier_multisession.py — Multi-session per-node binary classifiers.

Trains one binary RandomForest classifier per acoustic node using GPS data
pooled across all available sessions.  Each classifier answers "is the person
at my node right now?" using only that node's own z-score-normalised audio
features (rolling mean + std of dB power).

Per-session z-score normalisation removes the session-specific absolute level
and scale from each node's signal, making the 2 features transferable across
recording days with different weather, placement, and background noise.

Session discovery is automatic: drop new {session}_*.flac (×6 nodes) and the
matching {session}_gps2_gps.csv into --flac-dir and re-run — no code changes
needed.

Outputs (saved to --out-dir):
    node_clf_N{nid}_multisession_{TIMESTAMP}.pkl   (6 files — one per node)
    node_clf_ms_report_{TIMESTAMP}.txt             (LOSO report)

The pkl filenames match the glob used by finetune_deterministic.py
--node-clfs-dir, so they are drop-in compatible.

Usage
-----
    python collection/train_classifier_multisession.py \\
        --flac-dir  iobt_data \\
        --nodes-txt iobt_data/node_positions.txt \\
        --window    5 \\
        --out-dir   results/multisession

    # LOSO evaluation only, skip saving final classifiers:
    python collection/train_classifier_multisession.py \\
        --flac-dir  iobt_data \\
        --nodes-txt iobt_data/node_positions.txt \\
        --loso-only
"""

import argparse
import re
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
)
from collection.flac_to_trackmdp import (
    discover_flac_files,
    compute_power_per_step,
    load_node_positions_ordered,
    load_gps_track,
    _parse_session_start,
)

# Node IDs required for a valid session (orin_11 through orin_16)
EXPECTED_NODE_IDS = list(range(11, 17))   # [11, 12, 13, 14, 15, 16]
DEFAULT_STEP_S    = 0.5


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def discover_valid_sessions(flac_dir: Path) -> list[str]:
    """
    Return sorted list of session IDs that have:
      - All 6 FLAC files for nodes 11–16  (pattern: orin_<N>)
      - Matching GPS CSV: {session}_gps2_gps.csv

    Session ID = first 15 chars of the FLAC filename (YYYYMMDD_HHMMSS).
    """
    session_pattern = re.compile(r'^(\d{8}_\d{6})')
    node_pattern    = re.compile(r'orin_(\d+)', re.IGNORECASE)

    session_nodes: dict[str, set[int]] = {}
    for fpath in sorted(flac_dir.glob("*.flac")):
        sm = session_pattern.match(fpath.name)
        nm = node_pattern.search(fpath.name)
        if sm and nm:
            session_nodes.setdefault(sm.group(1), set()).add(int(nm.group(1)))

    valid = []
    for sess, nodes in sorted(session_nodes.items()):
        missing = [n for n in EXPECTED_NODE_IDS if n not in nodes]
        if missing:
            print(f"  [SKIP] {sess}: missing nodes {missing}")
            continue
        if not (flac_dir / f"{sess}_gps2_gps.csv").exists():
            print(f"  [SKIP] {sess}: GPS CSV not found")
            continue
        valid.append(sess)

    return valid


# ---------------------------------------------------------------------------
# Feature engineering — per-node, per-session z-scored
# ---------------------------------------------------------------------------

def build_node_features_normalized(power_array: np.ndarray, window: int) -> np.ndarray:
    """
    Build a (T, 2) feature matrix for a single node using only that node's
    own signal, with per-session z-score normalisation applied to the raw
    dB series before computing rolling statistics.

    Z-scoring removes the session-specific absolute level and scale, making
    the rolling mean and std transferable across sessions with different
    acoustic environments.

    Features:
        col 0 — causal rolling mean of z-scored dB power
        col 1 — causal rolling std  of z-scored dB power

    Parameters
    ----------
    power_array : (T,) float array of linear acoustic power values
    window      : rolling-window length in timesteps

    Returns
    -------
    (T, 2) float64 array
    """
    db      = 10.0 * np.log10(np.maximum(power_array, 1e-12))
    db_norm = (db - db.mean()) / (db.std() + 1e-9)   # per-session z-score
    X       = db_norm.reshape(-1, 1)
    mean    = apply_rolling_mean(X, window)            # (T, 1)
    std     = apply_rolling_std(X, window)             # (T, 1)
    return np.concatenate([mean, std], axis=1)         # (T, 2)


# ---------------------------------------------------------------------------
# Per-session data loading
# ---------------------------------------------------------------------------

def load_session_per_node(
    session:   str,
    flac_dir:  Path,
    nodes_txt: Path,
    window:    int,
    step_s:    float = DEFAULT_STEP_S,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Load one session and return per-node binary datasets.

    Returns
    -------
    dict mapping nid -> (X_valid, y_valid) where:
        X_valid : (M, 2) z-scored features for GPS-labelled steps only
        y_valid : (M,) int32 binary labels — 1 if person at nid, 0 otherwise
    """
    file_map = discover_flac_files(flac_dir, session=session)
    node_ids = sorted(file_map.keys())

    power_arrays: dict[int, np.ndarray] = {}
    for nid, fpath in file_map.items():
        power_arrays[nid] = compute_power_per_step(fpath, step_s)

    T          = min(len(a) for a in power_arrays.values())
    node_xy    = load_node_positions_ordered(nodes_txt, node_ids)
    gps_df     = load_gps_track(flac_dir / f"{session}_gps2_gps.csv")
    flac_start = _parse_session_start(session)
    gt         = build_ground_truth_sequence(
        gps_df, node_xy, node_ids, flac_start, step_s, T
    )

    valid    = gt > 0
    n_valid  = int(valid.sum())
    print(f"    {session}: T={T}  GPS-labelled={n_valid} ({n_valid/T*100:.1f}%)")

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nid in node_ids:
        X = build_node_features_normalized(power_arrays[nid][:T], window)
        y = (gt == nid).astype(np.int32)
        result[nid] = (X[valid], y[valid])

    return result


# ---------------------------------------------------------------------------
# LOSO cross-validation — per node
# ---------------------------------------------------------------------------

def loso_per_node(
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    node_ids:     list[int],
    sessions:     list[str],
) -> dict[str, dict[int, dict]]:
    """
    Leave-one-session-out cross-validation for each binary per-node classifier.

    For each held-out session, for each node:
      - Pool training data from all other sessions
      - Train binary RF
      - Evaluate precision / recall / F1 on the held-out session

    Returns
    -------
    dict: held_out_session -> nid -> {precision, recall, f1, support, accuracy}
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support

    results: dict[str, dict[int, dict]] = {}

    print("\n" + "=" * 68)
    print("  LEAVE-ONE-SESSION-OUT EVALUATION  (per node, binary)")
    print("=" * 68)

    for held_out in sessions:
        train_sessions = [s for s in sessions if s != held_out]
        results[held_out] = {}

        node_metrics: list[str] = []
        for nid in node_ids:
            # Pool training data across other sessions
            X_tr = np.vstack([session_data[s][nid][0] for s in train_sessions])
            y_tr = np.concatenate([session_data[s][nid][1] for s in train_sessions])
            X_te, y_te = session_data[held_out][nid]

            n_pos_train = int(y_tr.sum())
            n_pos_test  = int(y_te.sum())

            if n_pos_train == 0:
                # Node never visited in training data — cannot train
                results[held_out][nid] = dict(
                    precision=0.0, recall=0.0, f1=0.0,
                    support=n_pos_test, accuracy=float((y_te == 0).mean()),
                )
                node_metrics.append(f"n{nid}=--- (no train pos)")
                continue

            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, y_pred, average="binary", zero_division=0
            )
            acc = float((y_pred == y_te).mean())

            results[held_out][nid] = dict(
                precision=float(prec), recall=float(rec),
                f1=float(f1), support=n_pos_test, accuracy=acc,
            )
            node_metrics.append(f"n{nid}=F1:{f1:.2f}/R:{rec:.2f}")

        print(f"  held-out {held_out}")
        print(f"    " + "  ".join(node_metrics))

    # Summary: mean LOSO recall per node
    print("\n  Mean LOSO recall per node:")
    for nid in node_ids:
        recalls = [results[s][nid]["recall"] for s in sessions]
        print(f"    node {nid}: {np.mean(recalls):.3f}  "
              f"({', '.join(f'{r:.2f}' for r in recalls)})")
    print("=" * 68)

    return results


# ---------------------------------------------------------------------------
# Final classifiers — trained on all sessions pooled
# ---------------------------------------------------------------------------

def train_final_classifiers(
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    node_ids:     list[int],
    sessions:     list[str],
) -> dict[int, object]:
    """
    Pool all sessions per node, train final binary RF classifier.

    Returns dict: nid -> fitted RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    clfs: dict[int, object] = {}
    for nid in node_ids:
        X_all = np.vstack([session_data[s][nid][0] for s in sessions])
        y_all = np.concatenate([session_data[s][nid][1] for s in sessions])

        pos_rate = y_all.mean()
        print(f"  Node {nid}: {int(y_all.sum())} positive / {len(y_all)} total "
              f"({pos_rate*100:.1f}%)")

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_all, y_all)
        clfs[nid] = clf

    return clfs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_classifiers(
    clfs:         dict[int, object],
    loso_results: dict[str, dict[int, dict]],
    node_ids:     list[int],
    sessions:     list[str],
    total_steps:  int,
    window:       int,
    out_dir:      Path,
    ts:           str,
) -> list[Path]:
    import joblib

    saved_paths = []
    for nid, clf in clfs.items():
        pkl_path = out_dir / f"node_clf_N{nid}_multisession_{ts}.pkl"
        joblib.dump(clf, pkl_path)
        saved_paths.append(pkl_path)
        print(f"  Saved: {pkl_path.name}")

    # Report
    lines = [
        "=" * 68,
        "  MULTI-SESSION PER-NODE BINARY CLASSIFIER REPORT",
        "=" * 68,
        f"  Created    : {datetime.now().isoformat()}",
        f"  Window     : {window} steps ({window * DEFAULT_STEP_S:.1f}s)",
        f"  Norm       : per-session z-score per node",
        f"  Sessions   : {len(sessions)}  →  {sessions}",
        f"  Total GPS steps : {total_steps}",
        "",
        "  LOSO RESULTS  (F1 / Recall per node per held-out session)",
        "  " + "-" * 64,
        "  held-out session     " + "  ".join(f" n{nid:>2}" for nid in node_ids),
        "  " + "-" * 64,
    ]
    for sess in sorted(loso_results.keys()):
        row = f"  {sess}  "
        for nid in node_ids:
            m = loso_results[sess][nid]
            row += f"  {m['f1']:4.2f}/{m['recall']:4.2f}"
        lines.append(row)

    lines.append("  " + "-" * 64)
    lines.append("  Mean LOSO recall per node:")
    for nid in node_ids:
        recalls = [loso_results[s][nid]["recall"] for s in sessions]
        lines.append(f"    node {nid}: {np.mean(recalls):.3f}")
    lines.append("=" * 68)

    report = "\n".join(lines)
    print("\n" + report)

    rpt_path = out_dir / f"node_clf_ms_report_{ts}.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"\n  Report → {rpt_path}")

    return saved_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-session per-node binary RF classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python collection/train_classifier_multisession.py \\
      --flac-dir  iobt_data \\
      --nodes-txt iobt_data/node_positions.txt \\
      --window 5 \\
      --out-dir results/multisession
        """,
    )
    _D = os.path.join(project_root, "iobt_data")
    parser.add_argument("--flac-dir",  type=Path, default=Path(_D),
                        help="Directory containing FLAC files and GPS CSVs")
    parser.add_argument("--nodes-txt", type=Path,
                        default=Path(os.path.join(_D, "node_positions.txt")),
                        help="Tab-delimited node_positions.txt")
    parser.add_argument("--window",    type=int,  default=5,
                        help="Causal rolling-window length in timesteps (default: 5)")
    parser.add_argument("--step",      type=float, default=DEFAULT_STEP_S,
                        help=f"FLAC step size in seconds (default: {DEFAULT_STEP_S})")
    parser.add_argument("--out-dir",   type=Path, default=Path("results/multisession"),
                        help="Output directory (default: results/multisession)")
    parser.add_argument("--loso-only", action="store_true", default=False,
                        help="Run LOSO evaluation only; skip saving final classifiers")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 68)
    print("  MULTI-SESSION PER-NODE BINARY CLASSIFIER TRAINING")
    print("=" * 68)
    print(f"  FLAC dir   : {args.flac_dir}")
    print(f"  Nodes txt  : {args.nodes_txt}")
    print(f"  Window     : {args.window} steps ({args.window * args.step:.1f}s)")
    print(f"  Step size  : {args.step}s")
    print(f"  Out dir    : {args.out_dir}")
    print(f"  LOSO only  : {args.loso_only}")

    # ── [1/4] Discover sessions ───────────────────────────────────────────────
    print(f"\n[1/4] Discovering valid sessions in {args.flac_dir} …")
    sessions = discover_valid_sessions(args.flac_dir)
    if not sessions:
        print("[ERROR] No valid sessions found. Each session needs:\n"
              "  - 6 FLAC files for nodes 11–16 (orin_NN in filename)\n"
              "  - {session}_gps2_gps.csv in the same directory")
        sys.exit(1)
    print(f"  Found {len(sessions)} valid session(s): {sessions}")
    if len(sessions) < 2:
        print("[WARN] Only 1 session — LOSO requires ≥2 sessions. "
              "Falling back to 70/30 chronological split.")

    # ── [2/4] Load all sessions ───────────────────────────────────────────────
    print(f"\n[2/4] Loading per-node features for all sessions …")
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    total_steps = 0
    for sess in sessions:
        node_data = load_session_per_node(
            sess, args.flac_dir, args.nodes_txt, args.window, args.step
        )
        session_data[sess] = node_data
        # Count GPS-labelled steps via node 11 (all nodes have same valid mask)
        total_steps += len(node_data[EXPECTED_NODE_IDS[0]][0])

    print(f"\n  Total GPS-labelled steps : {total_steps}")
    print(f"  Node IDs                 : {EXPECTED_NODE_IDS}")
    print(f"  Features per node        : 2  (z-scored rolling mean + std)")

    # ── [3/4] LOSO cross-validation ──────────────────────────────────────────
    print(f"\n[3/4] Running LOSO cross-validation …")
    if len(sessions) >= 2:
        loso_results = loso_per_node(session_data, EXPECTED_NODE_IDS, sessions)
    else:
        # Single session: 70/30 split fallback
        print("  [single session] Using 70/30 chronological split …")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_recall_fscore_support
        sess = sessions[0]
        loso_results = {sess: {}}
        for nid in EXPECTED_NODE_IDS:
            X_all, y_all = session_data[sess][nid]
            split = int(len(y_all) * 0.70)
            X_tr, y_tr = X_all[:split], y_all[:split]
            X_te, y_te = X_all[split:],  y_all[split:]
            clf = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=42,
            )
            if y_tr.sum() == 0:
                loso_results[sess][nid] = dict(
                    precision=0.0, recall=0.0, f1=0.0,
                    support=int(y_te.sum()), accuracy=float((y_te == 0).mean()),
                )
                continue
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, y_pred, average="binary", zero_division=0
            )
            loso_results[sess][nid] = dict(
                precision=float(prec), recall=float(rec), f1=float(f1),
                support=int(y_te.sum()), accuracy=float((y_pred == y_te).mean()),
            )
            print(f"  Node {nid}: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    # ── [4/4] Final classifiers ────────────────────────────────────────────────
    if not args.loso_only:
        print(f"\n[4/4] Training final classifiers on all {total_steps} steps …")
        clfs = train_final_classifiers(session_data, EXPECTED_NODE_IDS, sessions)
        saved = save_classifiers(
            clfs, loso_results, EXPECTED_NODE_IDS, sessions,
            total_steps, args.window, args.out_dir, ts,
        )
        print("\n" + "=" * 68)
        print("  DONE")
        print("=" * 68)
        print("\n  Use with finetune_deterministic.py:")
        print(f"    --node-clfs-dir {args.out_dir}")
    else:
        print(f"\n[4/4] --loso-only: skipping final classifier training.")
        print("\n" + "=" * 68)
        print("  DONE")
        print("=" * 68)


if __name__ == "__main__":
    main()
