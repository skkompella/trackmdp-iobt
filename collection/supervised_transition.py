#!/usr/bin/env python3
"""
supervised_transition.py — GPS-supervised classifier for Track-MDP node ranking.

Replaces the Bayesian path-loss estimator with a supervised classifier
trained directly from GPS ground-truth labels, bypassing path-loss parameters
that can be poorly fitted (muted nodes).

Pipeline
--------
1. Load FLAC files → per-node RSSI dB vectors at every 0.5s step.
2. Load GPS track → linearly interpolate position at each FLAC step →
   nearest-node label per step (ground truth).
3. Train classifier on (RSSI, label) pairs.
4. Apply classifier to all steps → supervised rankings.
5. Build transition matrix from rankings.

Usage
-----
    python collection/supervised_transition.py \\
        --flac-dir  iobt_data \\
        --nodes-txt iobt_data/node_positions.txt \\
        --gps-csv   iobt_data/20260417_100634_gps2_gps.csv \\
        --session   20260417_100634 \\
        --out-dir   results/20260417_100634 \\
        --model     rf

Outputs
-------
  ground_truth_SESSION.npy              int32 (T,) — node ID per FLAC step, 0=no GPS
  ground_truth_SESSION_meta.json
  classifier_SESSION.pkl                sklearn model (joblib)
  classifier_SESSION.json               centroid params (centroid model only)
  rankings_supervised_SESSION.npy       int32 (T,) supervised rankings
  rankings_supervised_SESSION_meta.json
  transition_matrix_SESSION.npy         float64 (N,N)
  transition_matrix_raw_SESSION.npy     int64   (N,N)
  dominant_cycle_SESSION.json
  transition_report_SESSION.txt
  transition_analysis_SESSION.png       (if matplotlib available)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import joblib
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

# ---------------------------------------------------------------------------
# Reuse from flac_to_trackmdp
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from flac_to_trackmdp import (
    _parse_session_start,
    build_and_save_transition,
    compute_power_per_step,
    discover_flac_files,
    fit_supervised_classifier,
    load_gps_track,
    load_node_positions_ordered,
)

DEFAULT_STEP_S    = 0.5
DEFAULT_MIN_DWELL = 2
DEFAULT_MIN_PROB  = 0.05


# ---------------------------------------------------------------------------
# Ground truth: GPS position → nearest node at every FLAC step
# ---------------------------------------------------------------------------

def build_ground_truth_sequence(
    gps_df,
    node_xy:    np.ndarray,
    node_ids:   list[int],
    flac_start: datetime,
    step_s:     float,
    T:          int,
) -> np.ndarray:
    """
    Interpolate GPS position at each FLAC timestep and assign nearest node ID.

    For each FLAC step t (wall-clock = flac_start + t*step_s):
      - Linearly interpolate UTM position between the two bracketing GPS samples.
      - Find the nearest node via argmin Euclidean distance.

    Steps outside the GPS window are labelled 0 (invalid).

    Returns int32 (T,) of node IDs (1-indexed), 0 = no GPS coverage.
    """
    gps_times = gps_df['timestamp'].values            # numpy datetime64
    gps_x     = gps_df['x'].values.astype(np.float64)
    gps_y     = gps_df['y'].values.astype(np.float64)

    # Convert flac_start to numpy datetime64[ns] for comparison
    t0_ns = np.datetime64(flac_start.replace(tzinfo=None), 'ns')

    gt = np.zeros(T, dtype=np.int32)

    for t in range(T):
        t_ns = t0_ns + np.timedelta64(int(t * step_s * 1e9), 'ns')

        # Find bracketing GPS indices
        idx = np.searchsorted(gps_times, t_ns)  # first GPS sample >= t_ns

        if idx == 0:
            if gps_times[0] == t_ns:
                pos_x, pos_y = gps_x[0], gps_y[0]
            else:
                continue  # before first GPS sample
        elif idx >= len(gps_times):
            continue      # after last GPS sample
        else:
            # Linear interpolation between idx-1 and idx
            t_lo = gps_times[idx - 1]
            t_hi = gps_times[idx]
            span = float((t_hi - t_lo) / np.timedelta64(1, 's'))
            frac = float((t_ns  - t_lo) / np.timedelta64(1, 's')) / max(span, 1e-9)
            pos_x = gps_x[idx - 1] + frac * (gps_x[idx] - gps_x[idx - 1])
            pos_y = gps_y[idx - 1] + frac * (gps_y[idx] - gps_y[idx - 1])

        dx   = pos_x - node_xy[:, 0]
        dy   = pos_y - node_xy[:, 1]
        near = int(np.argmin(dx ** 2 + dy ** 2))
        gt[t] = node_ids[near]

    return gt


# ---------------------------------------------------------------------------
# RSSI feature matrix
# ---------------------------------------------------------------------------

def apply_rolling_mean(X: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean along the time axis. Returns same shape as X."""
    if window <= 1:
        return X
    T, N = X.shape
    out    = np.empty_like(X)
    cumsum = np.zeros((T + 1, N), dtype=np.float64)
    cumsum[1:] = np.cumsum(X, axis=0)
    for t in range(T):
        start   = max(0, t - window + 1)
        count   = t - start + 1
        out[t]  = (cumsum[t + 1] - cumsum[start]) / count
    return out


def apply_rolling_std(X: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling standard deviation along the time axis."""
    if window <= 1:
        return np.zeros_like(X)
    mean = apply_rolling_mean(X, window)
    sq_mean = apply_rolling_mean(X ** 2, window)
    variance = np.maximum(sq_mean - mean ** 2, 0.0)
    return np.sqrt(variance)


def build_rssi_matrix(
    power_arrays: dict,
    node_ids:     list[int],
    T:            int,
    rssi_window:  int = 1,
) -> np.ndarray:
    """Build feature matrix from per-node RSSI dB.

    Features per timestep (concatenated):
      - N rolling-mean dB values
      - N rolling-std dB values (zero if rssi_window <= 1)
      - N*(N-1)/2 pairwise dB differences (X[i] - X[j], i < j)

    Returns (T, F) float64 where F = 2*N + N*(N-1)/2.
    """
    N = len(node_ids)
    X_raw = np.zeros((T, N), dtype=np.float64)
    for j, nid in enumerate(node_ids):
        pwr = power_arrays[nid][:T]
        X_raw[:, j] = 10.0 * np.log10(np.maximum(pwr, 1e-12))

    X_mean = apply_rolling_mean(X_raw, rssi_window)
    X_std  = apply_rolling_std(X_raw, rssi_window)

    # Pairwise differences on the smoothed means
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    X_diff = np.stack([X_mean[:, i] - X_mean[:, j] for i, j in pairs], axis=1)

    return np.concatenate([X_mean, X_std, X_diff], axis=1)


def _apply_centroid_classifier(X: np.ndarray, classifier: dict) -> np.ndarray:
    """Vectorized nearest-centroid inference. Returns 0-indexed class indices (T,)."""
    mean  = np.array(classifier["scaler_mean"])
    std   = np.array(classifier["scaler_std"])
    C     = np.array(classifier["centroids"])
    Z     = (X - mean) / std
    diffs = Z[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return np.argmin(dists, axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Sklearn-based classifiers
# ---------------------------------------------------------------------------

SKLEARN_MODELS = ("rf", "svm", "mlp")


def _build_sklearn_model(model_type: str):
    """Construct an unfitted sklearn estimator."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, n_jobs=-1, random_state=42,
        )
    if model_type == "svm":
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                           decision_function_shape="ovr")),
        ])
    if model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp",    MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu",
                max_iter=500, random_state=42, early_stopping=True,
                n_iter_no_change=20,
            )),
        ])
    raise ValueError(f"Unknown model type: {model_type!r}")


def train_sklearn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    node_ids: list[int],
    model_type: str,
):
    """Train a sklearn classifier and return (fitted_model, label_order).

    label_order maps clf.classes_ indices back to node IDs.
    """
    clf = _build_sklearn_model(model_type)
    clf.fit(X_train, y_train)
    return clf


def apply_sklearn_classifier(
    X: np.ndarray,
    clf,
    node_ids: list[int],
) -> np.ndarray:
    """Run inference; returns int32 array of node IDs (T,)."""
    return clf.predict(X).astype(np.int32)


# ---------------------------------------------------------------------------
# Save ground truth
# ---------------------------------------------------------------------------

def save_ground_truth(gt, node_ids, T, step_s, out_dir, session):
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path  = out_dir / f"ground_truth_{session}.npy"
    meta_path = out_dir / f"ground_truth_{session}_meta.json"

    np.save(npy_path, gt)

    valid     = gt[gt > 0]
    coverage  = 100.0 * len(valid) / T if T > 0 else 0.0
    counts    = {nid: int((gt == nid).sum()) for nid in node_ids}

    meta = {
        "session":          session,
        "created_at_utc":   datetime.now(timezone.utc).isoformat(),
        "step_s":           step_s,
        "total_steps":      T,
        "valid_steps":      int(len(valid)),
        "coverage_pct":     round(coverage, 2),
        "node_ids":         node_ids,
        "node_counts":      counts,
        "description":      "gt[t] = node ID of nearest node at FLAC step t (0 = no GPS coverage)",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Ground truth coverage : {coverage:.1f}%  ({len(valid)}/{T} steps valid)")
    print("  Per-node occupancy:")
    max_count = max(counts.values()) if counts else 1
    for nid in node_ids:
        bar = "█" * int(counts[nid] / max_count * 20)
        print(f"    node {nid}: {counts[nid]:6d} steps  {bar}")
    print(f"\n  ground_truth .npy  → {npy_path}")
    print(f"  ground_truth .json → {meta_path}")


# ---------------------------------------------------------------------------
# Save supervised rankings
# ---------------------------------------------------------------------------

def save_rankings(rankings, node_ids, T, step_s, out_dir, session,
                  model_type="centroid", rssi_window=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path  = out_dir / f"rankings_supervised_{session}.npy"
    meta_path = out_dir / f"rankings_supervised_{session}_meta.json"

    np.save(npy_path, rankings)

    meta = {
        "schema_version":    1,
        "created_at_utc":    datetime.now(timezone.utc).isoformat(),
        "method":            f"supervised_{model_type}",
        "model_type":        model_type,
        "rssi_window":       rssi_window,
        "step_s":            step_s,
        "total_steps":       T,
        "node_ids":          node_ids,
        "array_shape":       list(rankings.shape),
        "array_dtype":       str(rankings.dtype),
        "unique_nodes_seen": sorted(int(n) for n in set(rankings.tolist()) if n > 0),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  rankings_supervised .npy  → {npy_path}")
    print(f"  rankings_supervised .json → {meta_path}")
    print(f"  Unique nodes seen         : {meta['unique_nodes_seen']}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    flac_dir:    Path,
    nodes_txt:   Path,
    gps_csv:     Path,
    session:     str,
    step_s:      float,
    rssi_window: int,
    min_dwell:   int,
    min_prob:    float,
    no_plot:     bool,
    out_dir:     Path,
    model_type:  str = "rf",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── [1/6] FLAC discovery & power extraction ───────────────────────────
    print(f"\n[1/6] Scanning {flac_dir} for FLAC files (session={session}) …")
    file_map = discover_flac_files(flac_dir, session=session)
    if not file_map:
        print("[ERROR] No FLAC files found.")
        sys.exit(1)
    node_ids = sorted(file_map.keys())
    print(f"       Found {len(node_ids)} nodes: {node_ids}")
    for nid, fp in file_map.items():
        print(f"         node {nid:3d} → {fp.name}")

    print(f"\n       Computing power (step={step_s}s) …")
    power_arrays: dict[int, np.ndarray] = {}
    for nid, fpath in file_map.items():
        pwr = compute_power_per_step(fpath, step_s)
        power_arrays[nid] = pwr
        print(f"         node {nid:3d}: {len(pwr)} steps")
    T = min(len(a) for a in power_arrays.values())
    print(f"       Using T={T} steps (shortest file).")

    # ── [2/6] Node positions & GPS ────────────────────────────────────────
    print(f"\n[2/6] Loading node positions from {nodes_txt.name} …")
    node_xy = load_node_positions_ordered(nodes_txt, node_ids)
    for i, nid in enumerate(node_ids):
        print(f"         node {nid:3d}: x={node_xy[i,0]:.1f}  y={node_xy[i,1]:.1f}")

    print(f"\n       Loading GPS track from {gps_csv.name} …")
    gps_df = load_gps_track(gps_csv)
    print(f"       {len(gps_df)} GPS samples  "
          f"({gps_df['timestamp'].iloc[0]} → {gps_df['timestamp'].iloc[-1]})")

    # ── [3/6] Ground truth ────────────────────────────────────────────────
    print(f"\n[3/6] Building ground truth (interpolating GPS → FLAC steps) …")
    flac_start = _parse_session_start(session)
    print(f"       FLAC t=0 = {flac_start.isoformat()}")
    gt = build_ground_truth_sequence(gps_df, node_xy, node_ids, flac_start, step_s, T)

    print("\n" + "=" * 60)
    print("  GROUND TRUTH")
    print("=" * 60)
    save_ground_truth(gt, node_ids, T, step_s, out_dir, ts)

    # ── [4/6] Build RSSI matrix & train classifier ────────────────────────
    X = build_rssi_matrix(power_arrays, node_ids, T, rssi_window=rssi_window)
    print(f"\n[4/6] Built RSSI feature matrix "
          f"(T={T}, N={len(node_ids)}, rssi_window={rssi_window}, "
          f"features={X.shape[1]}, model={model_type}) …")

    valid_mask  = gt > 0
    valid_idxs  = np.where(valid_mask)[0]              # timestep indices with GPS
    n_valid     = len(valid_idxs)

    # Chronological 70/30 split for held-out evaluation
    split       = int(n_valid * 0.70)
    train_idxs  = valid_idxs[:split]
    test_idxs   = valid_idxs[split:]

    X_train_cv  = X[train_idxs]
    y_train_cv  = gt[train_idxs]
    X_test_cv   = X[test_idxs]
    y_test_cv   = gt[test_idxs]

    print(f"       Hold-out split (70/30): {len(train_idxs)} train / "
          f"{len(test_idxs)} test steps …")

    def _fit_and_eval(X_tr, y_tr, X_te, y_te):
        if model_type == "centroid":
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            y_idx = np.array([id_to_idx[nid] for nid in y_tr], dtype=np.int32)
            dist  = np.ones((len(y_tr), len(node_ids)), dtype=np.float64)
            dist[np.arange(len(y_tr)), y_idx] = 0.0
            clf_cv = fit_supervised_classifier(X_tr, dist, node_ids)
            cell_cv = _apply_centroid_classifier(X_te, clf_cv)
            pred_cv = np.array([node_ids[c] for c in cell_cv], dtype=np.int32)
        else:
            clf_cv  = train_sklearn_classifier(X_tr, y_tr, node_ids, model_type)
            pred_cv = apply_sklearn_classifier(X_te, clf_cv, node_ids)
        acc = float((pred_cv == y_te).sum()) / len(y_te)
        return acc, pred_cv

    ho_acc, ho_pred = _fit_and_eval(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    print(f"\n  Hold-out accuracy (last 30%): {ho_acc * 100:.1f}%  "
          f"({int(ho_acc * len(test_idxs))}/{len(test_idxs)} steps)")
    # Per-node breakdown on held-out set
    print("  Per-node hold-out accuracy:")
    for nid in node_ids:
        mask_n = y_test_cv == nid
        if mask_n.sum() == 0:
            continue
        acc_n = float((ho_pred[mask_n] == nid).sum()) / mask_n.sum()
        print(f"    node {nid:3d}: {acc_n * 100:5.1f}%  ({mask_n.sum()} steps)")

    # Retrain on ALL labeled data for final predictions & transition matrix
    X_all = X[valid_idxs]
    y_all = gt[valid_idxs]
    print(f"\n       Retraining on all {n_valid} labelled steps for final model …")

    if model_type == "centroid":
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        y_idx     = np.array([id_to_idx[nid] for nid in y_all], dtype=np.int32)
        distances = np.ones((n_valid, len(node_ids)), dtype=np.float64)
        distances[np.arange(n_valid), y_idx] = 0.0
        classifier = fit_supervised_classifier(X_all, distances, node_ids)
        classifier["rssi_window"] = rssi_window
        clf_path = out_dir / f"classifier_{ts}.json"
        with open(clf_path, "w") as f:
            json.dump(classifier, f, indent=2)
        print(f"\n  classifier .json → {clf_path}")
    else:
        if not _JOBLIB:
            print("[ERROR] joblib is required for sklearn models. "
                  "Run: pip install scikit-learn")
            sys.exit(1)
        clf_sklearn = train_sklearn_classifier(X_all, y_all, node_ids, model_type)
        clf_path = out_dir / f"classifier_{ts}.pkl"
        joblib.dump(clf_sklearn, clf_path)
        print(f"\n  classifier .pkl → {clf_path}")

    # ── [5/6] Supervised inference → rankings ─────────────────────────────
    print(f"\n[5/6] Applying classifier to {T} steps …")
    if model_type == "centroid":
        cell_seq = _apply_centroid_classifier(X, classifier)
        rankings = np.array([node_ids[c] for c in cell_seq], dtype=np.int32)
    else:
        rankings = apply_sklearn_classifier(X, clf_sklearn, node_ids)

    print("\n" + "=" * 60)
    print("  SUPERVISED RANKINGS")
    print("=" * 60)
    save_rankings(rankings, node_ids, T, step_s, out_dir, ts,
                  model_type=model_type, rssi_window=rssi_window)

    # ── [6/6] Transition matrix ────────────────────────────────────────────
    build_and_save_transition(
        rankings  = rankings,
        node_ids  = node_ids,
        node_xy   = node_xy,
        min_dwell = min_dwell,
        min_prob  = min_prob,
        no_plot   = no_plot,
        out_dir   = out_dir,
        session   = ts,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPS-supervised node ranking and transition matrix from ReSpeaker FLAC files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python collection/supervised_transition.py \\
      --flac-dir  iobt_data \\
      --nodes-txt iobt_data/node_positions.txt \\
      --gps-csv   iobt_data/20260417_100634_gps2_gps.csv \\
      --session   20260417_100634 \\
      --out-dir   results/20260417_100634
        """
    )
    parser.add_argument("--flac-dir",  required=True, type=Path,
                        help="Folder containing per-node *.flac files")
    parser.add_argument("--nodes-txt", required=True, type=Path,
                        help="Tab-delimited node_positions.txt (Latitude, Longitude, ordered by node ID)")
    parser.add_argument("--gps-csv",   required=True, type=Path,
                        help="GPS ground-truth CSV (Timestamp, Latitude, Longitude, …)")
    parser.add_argument("--session",   required=True, type=str,
                        help="Session prefix to filter FLAC files and anchor FLAC t=0 "
                             "(e.g. 20260417_100634)")
    parser.add_argument("--out-dir",   type=Path, default=Path("./results"),
                        help="Output directory (default: ./results)")
    parser.add_argument("--step",        type=float, default=DEFAULT_STEP_S,
                        help=f"FLAC window size in seconds (default: {DEFAULT_STEP_S})")
    parser.add_argument("--rssi-window", type=int,   default=10,
                        help="Causal rolling mean window over RSSI dB features "
                             "(default: 10 steps = 5s). Set to 1 to disable.")
    parser.add_argument("--model", type=str, default="rf",
                        choices=["centroid", "rf", "svm", "mlp"],
                        help="Classifier type: centroid (nearest-centroid), "
                             "rf (RandomForest), svm (RBF-SVM), mlp (MLP). "
                             "Default: rf")
    parser.add_argument("--min-dwell", type=int,   default=DEFAULT_MIN_DWELL,
                        help=f"Transition min-dwell filter (default: {DEFAULT_MIN_DWELL})")
    parser.add_argument("--min-prob",  type=float, default=DEFAULT_MIN_PROB,
                        help=f"Min transition probability for cycle extraction "
                             f"(default: {DEFAULT_MIN_PROB})")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip matplotlib visualisation")

    args = parser.parse_args()

    for p, name in [(args.flac_dir,  "--flac-dir"),
                    (args.nodes_txt, "--nodes-txt"),
                    (args.gps_csv,   "--gps-csv")]:
        if not p.exists():
            print(f"[ERROR] {name} not found: {p}")
            sys.exit(1)

    print("=" * 60)
    print("  SUPERVISED TRANSITION PIPELINE")
    print("=" * 60)
    print(f"  FLAC dir   : {args.flac_dir}")
    print(f"  Nodes txt  : {args.nodes_txt}")
    print(f"  GPS CSV    : {args.gps_csv}")
    print(f"  Session    : {args.session}")
    print(f"  Step size  : {args.step} s")
    print(f"  Min dwell  : {args.min_dwell}")
    print(f"  Min prob   : {args.min_prob}")
    print(f"  Model      : {args.model}")
    print(f"  Output dir : {args.out_dir}")

    run(
        flac_dir    = args.flac_dir,
        nodes_txt   = args.nodes_txt,
        gps_csv     = args.gps_csv,
        session     = args.session,
        step_s      = args.step,
        rssi_window = args.rssi_window,
        min_dwell   = args.min_dwell,
        min_prob    = args.min_prob,
        no_plot     = args.no_plot,
        out_dir     = args.out_dir,
        model_type  = args.model,
    )

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
