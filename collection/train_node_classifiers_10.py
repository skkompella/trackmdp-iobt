#!/usr/bin/env python3
"""
train_node_classifiers_10.py — Per-node binary classifiers + YOLO camera fusion
for the 10-node IoBT environment (IOBT_MAP nodes 1–10).

Adapts the proven v3 amplitude-preserving audio pipeline from
train_node_classifiers_pcen.py to the iobt_data_10/ session layout and adds:
  1. YOLO camera pipeline   → P_cam  (T, 10) float32
  2. OR fusion              → B      (T, 10) bool

Run order
---------
  Phase 0 — validate all session data (always runs first)
  Phase 1 — audio: LOSO cross-validation + final classifier training
  Phase 2 — camera: load YOLO detections → static filter → bin → P_cam
  Phase 3 — fuse: P_audio OR P_cam → B

Data layout (per session)
-------------------------
  iobt_data_10/<YYYYMMDD_HHMMSS>/
      meta_data.json              node positions (X, Y local-metric)
      gt_tracks.csv               GPS ground truth (DateTime, ElapsedTime, x, y)
      nodeN_respeaker.flac        audio, N=1..10 (16 kHz, 6 ch, int16)
      nodeN_zed_yolo.json         YOLO detections, nested list-of-lists

Outputs (to --out-dir)
----------------------
  node_clf_N{1..10}_multisession_v3_{TS}.pkl
  node_scaler_N{1..10}_multisession_v3_{TS}.pkl
  node_clf_v3_report_{TS}.txt
  fused_B_{session}_{TS}.npy       (T, 10) bool         [only with --fuse]
  p_cam_{session}_{TS}.npy         (T, 10) float32      [only with --fuse]
  p_audio_{session}_{TS}.npy       (T, 10) float32      [only with --fuse]

Usage
-----
  python collection/train_node_classifiers_10.py \\
      --data-dir  ~/trackmdp-iobt/iobt_data_10 \\
      --out-dir   results/v3_10node

  python collection/train_node_classifiers_10.py \\
      --data-dir  iobt_data_10 --validate-only

  python collection/train_node_classifiers_10.py \\
      --data-dir  iobt_data_10 --out-dir results/v3_10node --fuse
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import kurtosis as scipy_kurtosis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_NODE_IDS = list(range(1, 11))   # [1, 2, ..., 10]
DEFAULT_STEP_S    = 0.5
POWER_CHANNELS    = slice(1, 5)           # ReSpeaker body mics (0-indexed channels 1–4)
N_FFT_SUBBAND     = 512
HOP_SUBBAND       = 256
WINDOW_STEPS      = 5                     # causal rolling window in timesteps

_RAW_NAMES = [
    'db_mean', 'db_std',
    'band0_db', 'band1_db', 'band2_db', 'band3_db',
    'crest', 'kurtosis', 'dyn_range', 'mod_energy',
]
FEATURE_NAMES = [f'{n}_rmean' for n in _RAW_NAMES] + [f'{n}_rstd' for n in _RAW_NAMES]
N_RAW_FEATURES = len(_RAW_NAMES)    # 10
N_FEATURES     = len(FEATURE_NAMES) # 20

SESSION_RE = re.compile(r'^\d{8}_\d{6}$')
_YOLO_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
_SESSION_FMT  = "%Y%m%d_%H%M%S"


# ===========================================================================
# Phase 0 — Data validation
# ===========================================================================

def validate_session_data(session_dir: Path) -> bool:
    """
    Validate all data files for one session. Prints a pass/fail table.
    Returns True if all critical checks pass, False otherwise.
    """
    session_id = session_dir.name
    ok = True

    print(f"\n{'─'*60}")
    print(f"  Validating session: {session_id}")
    print(f"{'─'*60}")

    # ── meta_data.json ───────────────────────────────────────────────────────
    meta_path = session_dir / "meta_data.json"
    try:
        # Use the same position loader used during training (includes respeaker fallback)
        positions = load_node_positions_from_meta(meta_path)
        n_pos = len(positions)
        missing_nodes = [n for n in EXPECTED_NODE_IDS if n not in positions]
        if missing_nodes:
            print(f"  [FAIL] meta_data.json: no position for nodes {missing_nodes}")
            ok = False
        else:
            print(f"  [OK  ] meta_data.json: all 10 node positions resolved")
    except Exception as e:
        print(f"  [FAIL] meta_data.json: {e}")
        ok = False

    # ── gt_tracks.csv ────────────────────────────────────────────────────────
    gt_path = session_dir / "gt_tracks.csv"
    try:
        import pandas as _pd
        gt_df = _pd.read_csv(gt_path)
        required_cols = {"DateTime", "ElapsedTime", "x", "y"}
        missing_cols  = required_cols - set(gt_df.columns)
        if missing_cols:
            print(f"  [FAIL] gt_tracks.csv: missing columns {missing_cols}")
            ok = False
        elif gt_df[["x", "y"]].isnull().any().any():
            n_nan = int(gt_df[["x", "y"]].isnull().sum().sum())
            print(f"  [WARN] gt_tracks.csv: {n_nan} NaN values in x/y")
        elif len(gt_df) < 10:
            print(f"  [FAIL] gt_tracks.csv: only {len(gt_df)} rows")
            ok = False
        else:
            t_range = f"{gt_df['ElapsedTime'].min():.1f}–{gt_df['ElapsedTime'].max():.1f} s"
            print(f"  [OK  ] gt_tracks.csv: {len(gt_df)} rows, elapsed {t_range}")
    except Exception as e:
        print(f"  [FAIL] gt_tracks.csv: {e}")
        ok = False

    # ── FLAC files ───────────────────────────────────────────────────────────
    flac_ok = True
    for nid in EXPECTED_NODE_IDS:
        fpath = session_dir / f"node{nid}_respeaker.flac"
        if not fpath.exists():
            print(f"  [FAIL] node{nid}_respeaker.flac: not found")
            ok = flac_ok = False
            continue
        try:
            info = sf.info(str(fpath))
            if info.samplerate != 16000:
                print(f"  [WARN] node{nid}_respeaker.flac: sr={info.samplerate} (expected 16000)")
            if info.channels != 6:
                print(f"  [WARN] node{nid}_respeaker.flac: channels={info.channels} (expected 6)")
            if info.duration < 5.0:
                print(f"  [FAIL] node{nid}_respeaker.flac: duration={info.duration:.1f}s < 5s")
                ok = flac_ok = False
        except Exception as e:
            print(f"  [FAIL] node{nid}_respeaker.flac: {e}")
            ok = flac_ok = False
    if flac_ok:
        durations = []
        for nid in EXPECTED_NODE_IDS:
            fpath = session_dir / f"node{nid}_respeaker.flac"
            if fpath.exists():
                durations.append(sf.info(str(fpath)).duration)
        print(f"  [OK  ] FLAC files: all 10 present, "
              f"durations {min(durations):.1f}–{max(durations):.1f} s")

    # ── YOLO JSON files (optional — only needed for camera/fusion modes) ───────
    session_start_dt = _parse_session_start_10(session_id)
    n_yolo_missing = sum(
        1 for nid in EXPECTED_NODE_IDS
        if not (session_dir / f"node{nid}_zed_yolo.json").exists()
    )
    if n_yolo_missing == len(EXPECTED_NODE_IDS):
        print(f"  [WARN] No YOLO JSON files found (audio-only session — OK for audio training)")
    elif n_yolo_missing > 0:
        print(f"  [WARN] {n_yolo_missing}/10 YOLO JSON files missing")
    for nid in EXPECTED_NODE_IDS:
        jpath = session_dir / f"node{nid}_zed_yolo.json"
        if not jpath.exists():
            continue  # warned above; not a hard failure
            continue
        try:
            with open(jpath) as f:
                frames = json.load(f)
            if not isinstance(frames, list):
                print(f"  [FAIL] node{nid}_zed_yolo.json: top-level not a list")
                ok = False
                continue

            total_dets = 0; car_dets = 0; depth40 = 0
            bad_ts = 0; bad_world = 0
            t_min = float('inf'); t_max = float('-inf')
            conf_vals = []

            for frame in frames:
                if not isinstance(frame, list):
                    continue
                for det in frame:
                    if not isinstance(det, dict):
                        continue
                    total_dets += 1
                    if det.get("class") == "car":
                        car_dets += 1
                    if det.get("depth", 0) >= 39.9:
                        depth40 += 1
                    world = det.get("world", [])
                    if not (isinstance(world, list) and len(world) >= 2):
                        bad_world += 1
                    t_str = det.get("t", "")
                    try:
                        t_unix = datetime.strptime(t_str, _YOLO_TS_FMT).timestamp()
                        t_min = min(t_min, t_unix)
                        t_max = max(t_max, t_unix)
                    except Exception:
                        bad_ts += 1
                    conf = det.get("conf", 0.0)
                    conf_vals.append(conf)

            # Temporal alignment check
            t_align_warn = ""
            if total_dets > 0 and t_min != float('inf'):
                sess_unix = session_start_dt.timestamp()
                if t_min < sess_unix - 5 or t_max > sess_unix + 400:
                    t_align_warn = f" ⚠ timestamps outside session window"

            frac40 = depth40 / total_dets if total_dets > 0 else 0.0
            mean_conf = np.mean(conf_vals) if conf_vals else 0.0
            issues = []
            if bad_ts:
                issues.append(f"{bad_ts} bad-ts")
            if bad_world:
                issues.append(f"{bad_world} bad-world")
            issue_str = f" ⚠ {', '.join(issues)}" if issues else ""
            print(f"  [OK  ] node{nid}_zed_yolo.json: {total_dets} dets "
                  f"({car_dets} car), depth=40: {frac40:.0%}, "
                  f"conf_mean={mean_conf:.2f}{issue_str}{t_align_warn}")
            # bad-world / bad-ts are skipped silently during load; don't fail session

        except json.JSONDecodeError as e:
            print(f"  [FAIL] node{nid}_zed_yolo.json: JSON parse error — {e}")
            ok = False
        except Exception as e:
            print(f"  [FAIL] node{nid}_zed_yolo.json: {e}")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"\n  Session {session_id}: {status}")
    return ok


# ===========================================================================
# Audio feature extraction (verbatim from train_node_classifiers_pcen.py)
# ===========================================================================

def build_node_features(
    flac_path: Path,
    step_s:    float = DEFAULT_STEP_S,
) -> tuple[np.ndarray, int]:
    """
    Extract 10 raw features per timestep from one node's FLAC file.

    Returns
    -------
    raw_features : (T, 10) float64
    T            : number of steps
    """
    data, sr = sf.read(str(flac_path), dtype='int16', always_2d=True)
    win_samples = int(round(step_s * sr))
    T           = len(data) // win_samples

    stft_freqs = np.fft.rfftfreq(N_FFT_SUBBAND, d=1.0 / sr)
    band_edges = [(0, 500), (500, 2000), (2000, 4000), (4000, sr // 2)]
    band_masks = []
    for lo, hi in band_edges:
        mask = (stft_freqs >= lo) & (stft_freqs < hi)
        band_masks.append(mask if mask.any() else np.zeros(len(stft_freqs), dtype=bool))

    mod_lo = 2.0
    mod_hi = 1.0 / step_s

    hop = HOP_SUBBAND
    out = np.zeros((T, N_RAW_FEATURES), dtype=np.float64)

    for t in range(T):
        chunk = data[t * win_samples : (t + 1) * win_samples]
        x = chunk[:, POWER_CHANNELS].mean(axis=1).astype(np.float64)
        x -= x.mean()

        n_hop_frames = max(1, win_samples // hop)
        frame_powers = np.array([
            np.mean(x[i*hop : i*hop + hop] ** 2) + 1e-12
            for i in range(n_hop_frames)
        ])
        frame_db = 10.0 * np.log10(frame_powers)

        out[t, 0] = float(frame_db.mean())
        out[t, 1] = float(frame_db.std()) if len(frame_db) > 1 else 0.0

        S       = np.abs(np.fft.rfft(x, n=N_FFT_SUBBAND))
        S_power = S ** 2
        for b, mask in enumerate(band_masks):
            if mask.any():
                out[t, 2 + b] = 10.0 * np.log10(S_power[mask].mean() + 1e-10)

        rms  = np.sqrt(np.mean(x ** 2)) + 1e-10
        peak = np.max(np.abs(x))
        out[t, 6] = float(peak / rms)
        out[t, 7] = float(scipy_kurtosis(x, fisher=False))
        out[t, 8] = float(np.percentile(frame_db, 90) - np.percentile(frame_db, 10))

        if len(frame_db) >= 4:
            env_fft   = np.abs(np.fft.rfft(frame_db))
            env_freqs = np.fft.rfftfreq(len(frame_db), d=step_s / n_hop_frames)
            mod_mask  = (env_freqs >= mod_lo) & (env_freqs <= mod_hi)
            out[t, 9] = float(np.mean(env_fft[mod_mask] ** 2)) if mod_mask.any() else 0.0

    return out, T


def aggregate_rolling(raw_scaled: np.ndarray, window: int = WINDOW_STEPS) -> np.ndarray:
    """Causal rolling mean + std → (T, 20)."""
    import pandas as _pd
    df        = _pd.DataFrame(raw_scaled)
    roll_mean = df.rolling(window=window, min_periods=1).mean().values
    roll_std  = df.rolling(window=window, min_periods=1).std().fillna(0).values
    return np.hstack([roll_mean, roll_std])


def make_classifier():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def smooth_predictions(
    proba:     np.ndarray,
    window:    int   = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Rolling-mean smoothing on classifier probabilities."""
    import pandas as _pd
    smoothed = _pd.Series(proba).rolling(window=window, min_periods=1).mean()
    return (smoothed.values >= threshold).astype(np.int32)


# ===========================================================================
# 10-node data loading
# ===========================================================================

def _parse_session_start_10(session_id: str) -> datetime:
    """Parse 'YYYYMMDD_HHMMSS' → naive datetime (local timezone)."""
    return datetime.strptime(session_id, _SESSION_FMT)


def load_node_positions_from_meta(meta_json_path: Path) -> dict[int, np.ndarray]:
    """
    Extract node positions from meta_data.json sensors section.

    Returns {node_id: np.array([X, Y])} for node IDs 1–10.
    Uses 'nodeN_zed_left' camera sensors; X, Y are in local metric frame
    (same coordinate system as gt_tracks.csv x, y).
    """
    with open(meta_json_path) as f:
        meta = json.load(f)

    positions: dict[int, np.ndarray] = {}
    node_re = re.compile(r'node(\d+)_zed_left', re.IGNORECASE)

    for key, sensor in meta.get("sensors", {}).items():
        m = node_re.match(key)
        if m and sensor.get("type") == "camera":
            nid = int(m.group(1))
            positions[nid] = np.array([float(sensor["X"]), float(sensor["Y"])])

    # Fallback: fill any gaps from respeaker sensors (positions are identical,
    # but some sessions omit certain zed_left entries from meta_data.json).
    respeaker_re = re.compile(r'node(\d+)_respeaker', re.IGNORECASE)
    for key, sensor in meta.get("sensors", {}).items():
        m = respeaker_re.match(key)
        if m and sensor.get("type") == "audio":
            nid = int(m.group(1))
            if nid not in positions:
                positions[nid] = np.array([float(sensor["X"]), float(sensor["Y"])])

    return positions


def load_gps_track_10(gt_tracks_csv: Path):
    """
    Load gt_tracks.csv.

    Returns DataFrame with columns: ['elapsed_s', 'x', 'y']
    where elapsed_s is seconds from session start (from ElapsedTime column).
    """
    import pandas as _pd
    df = _pd.read_csv(gt_tracks_csv)
    df = df.rename(columns={"ElapsedTime": "elapsed_s"})
    df = df.dropna(subset=["elapsed_s", "x", "y"])
    return df[["elapsed_s", "x", "y"]].reset_index(drop=True)


def _discover_session_flac_10(session_dir: Path) -> dict[int, Path]:
    """Discover node{N}_respeaker.flac files in session_dir. Returns {nid: Path}."""
    file_map: dict[int, Path] = {}
    node_re = re.compile(r'^node(\d+)_respeaker\.flac$', re.IGNORECASE)
    for fpath in sorted(session_dir.glob("node*_respeaker.flac")):
        m = node_re.match(fpath.name)
        if m:
            file_map[int(m.group(1))] = fpath
    return file_map


def discover_valid_sessions_10(data_dir: Path) -> list[str]:
    """
    Scan data_dir for subdirectories matching YYYYMMDD_HHMMSS.
    Validates each has: all 10 FLACs, meta_data.json, gt_tracks.csv.
    Returns sorted list of valid session ID strings.
    """
    valid = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or not SESSION_RE.match(sub.name):
            continue
        sess = sub.name
        # Check required files
        missing = []
        for nid in EXPECTED_NODE_IDS:
            if not (sub / f"node{nid}_respeaker.flac").exists():
                missing.append(f"node{nid}_respeaker.flac")
        if not (sub / "meta_data.json").exists():
            missing.append("meta_data.json")
        if not (sub / "gt_tracks.csv").exists():
            missing.append("gt_tracks.csv")
        if missing:
            print(f"  [SKIP] {sess}: missing {missing[:3]}"
                  f"{'...' if len(missing) > 3 else ''}")
            continue
        valid.append(sess)

    return valid


def build_ground_truth_sequence_10(
    gt_df,
    node_positions: dict[int, np.ndarray],
    step_s:         float,
    T:              int,
) -> np.ndarray:
    """
    Build binary ground-truth node assignment for each FLAC bin.

    Uses gt_df['elapsed_s'] (seconds from session start) to locate the car
    at each 0.5s bin, then assigns to the nearest node.

    Parameters
    ----------
    gt_df         : DataFrame with 'elapsed_s', 'x', 'y' (sorted by elapsed_s)
    node_positions: {nid: [X, Y]} in same coordinate frame as gt_df x, y
    step_s        : bin size in seconds
    T             : number of bins

    Returns
    -------
    (T,) int32 — node IDs (1–10) for bins with GPS coverage; 0 = no coverage
    """
    elapsed = gt_df["elapsed_s"].values.astype(np.float64)
    gps_x   = gt_df["x"].values.astype(np.float64)
    gps_y   = gt_df["y"].values.astype(np.float64)

    node_ids  = sorted(node_positions.keys())
    node_xy   = np.array([node_positions[nid] for nid in node_ids])  # (N, 2)

    gt = np.zeros(T, dtype=np.int32)

    for t in range(T):
        t_s = t * step_s  # elapsed seconds for this bin

        # Find bracketing GPS samples by elapsed time
        idx = np.searchsorted(elapsed, t_s)

        if idx == 0:
            if elapsed[0] == t_s:
                px, py = gps_x[0], gps_y[0]
            else:
                continue  # before GPS coverage
        elif idx >= len(elapsed):
            continue      # after GPS coverage
        else:
            t0, t1 = elapsed[idx - 1], elapsed[idx]
            span   = t1 - t0
            frac   = (t_s - t0) / max(span, 1e-9)
            px     = gps_x[idx-1] + frac * (gps_x[idx] - gps_x[idx-1])
            py     = gps_y[idx-1] + frac * (gps_y[idx] - gps_y[idx-1])

        dx   = px - node_xy[:, 0]
        dy   = py - node_xy[:, 1]
        near = int(np.argmin(dx**2 + dy**2))
        gt[t] = node_ids[near]

    return gt


def load_session_raw_per_node_10(
    session_dir: Path,
    step_s:      float = DEFAULT_STEP_S,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Load one session and return per-node raw (un-z-scored) feature matrices.

    Returns
    -------
    {nid: (raw_features, y_binary)}
        raw_features : (M, 10) GPS-labelled steps only
        y_binary     : (M,) int32 — 1 if car at nid, 0 otherwise
    """
    file_map = _discover_session_flac_10(session_dir)
    if not file_map:
        raise ValueError(f"No FLAC files found in {session_dir}")

    node_ids = sorted(file_map.keys())

    # Extract raw features for all nodes; record per-node T
    node_raw: dict[int, tuple[np.ndarray, int]] = {}
    for nid in node_ids:
        print(f"    node{nid}: extracting features …", flush=True)
        raw, T = build_node_features(file_map[nid], step_s)
        node_raw[nid] = (raw, T)

    T_min = min(info[1] for info in node_raw.values())

    # Load node positions and GPS ground truth
    node_positions = load_node_positions_from_meta(session_dir / "meta_data.json")
    gt_df          = load_gps_track_10(session_dir / "gt_tracks.csv")
    gt             = build_ground_truth_sequence_10(gt_df, node_positions, step_s, T_min)

    valid   = gt > 0
    n_valid = int(valid.sum())
    print(f"    {session_dir.name}: T={T_min}  GPS-labelled={n_valid} "
          f"({n_valid/T_min*100:.1f}%)")

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nid in node_ids:
        raw, _ = node_raw[nid]
        y_bin  = (gt == nid).astype(np.int32)
        result[nid] = (raw[:T_min][valid], y_bin[valid])

    return result


# ===========================================================================
# LOSO cross-validation (verbatim logic from pcen.py, adapted for 10 nodes)
# ===========================================================================

def loso_per_node(
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    node_ids:     list[int],
    sessions:     list[str],
    window:       int = WINDOW_STEPS,
) -> dict[str, dict[int, dict]]:
    """Leave-one-session-out cross-validation. Scaler fitted on training fold only."""
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
    import pandas as _pd

    results: dict[str, dict[int, dict]] = {}

    print("\n" + "=" * 68)
    print("  LEAVE-ONE-SESSION-OUT EVALUATION  (per node, binary)")
    print("=" * 68)

    for held_out in sessions:
        train_sessions = [s for s in sessions if s != held_out]
        results[held_out] = {}
        node_metrics: list[str] = []

        for nid in node_ids:
            X_tr_raw = np.vstack([session_data[s][nid][0] for s in train_sessions])
            y_tr     = np.concatenate([session_data[s][nid][1] for s in train_sessions])
            X_te_raw, y_te = session_data[held_out][nid]

            n_pos_train = int(y_tr.sum())
            n_pos_test  = int(y_te.sum())

            if n_pos_train == 0:
                results[held_out][nid] = dict(
                    precision=0.0, recall=0.0, f1=0.0,
                    support=n_pos_test, accuracy=float((y_te == 0).mean()),
                )
                node_metrics.append(f"n{nid}=--- (no train pos)")
                continue

            scaler = StandardScaler()
            X_tr   = aggregate_rolling(scaler.fit_transform(X_tr_raw), window)
            X_te   = aggregate_rolling(scaler.transform(X_te_raw), window)

            clf = make_classifier()
            clf.fit(_pd.DataFrame(X_tr, columns=FEATURE_NAMES), y_tr)
            y_pred = clf.predict(_pd.DataFrame(X_te, columns=FEATURE_NAMES))

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, y_pred, average="binary", zero_division=0,
            )
            results[held_out][nid] = dict(
                precision=float(prec), recall=float(rec),
                f1=float(f1), support=n_pos_test,
                accuracy=float((y_pred == y_te).mean()),
            )
            node_metrics.append(f"n{nid}=F1:{f1:.2f}/R:{rec:.2f}")

        print(f"  held-out {held_out}")
        print(f"    " + "  ".join(node_metrics))

    print("\n  Mean LOSO recall per node:")
    for nid in node_ids:
        recalls = [results[s][nid]["recall"] for s in sessions]
        print(f"    node {nid:>2}: {np.mean(recalls):.3f}  "
              f"({', '.join(f'{r:.2f}' for r in recalls)})")
    print("=" * 68)

    return results


# ===========================================================================
# Final classifier training (all sessions pooled)
# ===========================================================================

def train_final_classifiers(
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    node_ids:     list[int],
    sessions:     list[str],
    window:       int = WINDOW_STEPS,
) -> dict[int, tuple[object, object]]:
    """Returns {nid: (fitted_clf, fitted_scaler)}."""
    from sklearn.preprocessing import StandardScaler
    import pandas as _pd

    clfs: dict[int, tuple[object, object]] = {}
    for nid in node_ids:
        X_raw = np.vstack([session_data[s][nid][0] for s in sessions])
        y_all = np.concatenate([session_data[s][nid][1] for s in sessions])
        pos_rate = y_all.mean()
        print(f"  Node {nid:>2}: {int(y_all.sum())} positive / {len(y_all)} total "
              f"({pos_rate*100:.1f}%)")

        scaler = StandardScaler()
        X_all  = aggregate_rolling(scaler.fit_transform(X_raw), window)

        clf = make_classifier()
        clf.fit(_pd.DataFrame(X_all, columns=FEATURE_NAMES), y_all)
        clfs[nid] = (clf, scaler)

    return clfs


def save_classifiers(
    clfs:         dict[int, tuple[object, object]],
    loso_results: dict[str, dict[int, dict]],
    node_ids:     list[int],
    sessions:     list[str],
    total_steps:  int,
    out_dir:      Path,
    ts:           str,
) -> list[Path]:
    import joblib

    saved_paths = []
    for nid, (clf, scaler) in clfs.items():
        clf_path    = out_dir / f"node_clf_N{nid}_multisession_v3_{ts}.pkl"
        scaler_path = out_dir / f"node_scaler_N{nid}_multisession_v3_{ts}.pkl"
        joblib.dump(clf,    clf_path)
        joblib.dump(scaler, scaler_path)
        saved_paths.append(clf_path)
        print(f"  Saved: {clf_path.name}")
        print(f"  Saved: {scaler_path.name}")

    lines = [
        "=" * 68,
        "  PER-NODE BINARY CLASSIFIER v3 — 10-NODE IoBT",
        "=" * 68,
        f"  Created    : {datetime.now().isoformat()}",
        f"  Features   : 20 (dB power + sub-band + crest/kurt + dyn range + modulation)",
        f"  Norm       : per-session z-score (leakage-free; scaler per node)",
        f"  Rolling    : mean + std over {WINDOW_STEPS} steps",
        f"  Classifier : LightGBM (n_estimators=200, num_leaves=31, balanced)",
        f"  Sessions   : {len(sessions)}  →  {sessions}",
        f"  Total GPS steps : {total_steps}",
        "",
        "  LOSO RESULTS  (F1 / Recall per node per held-out session)",
        "  " + "-" * 64,
        "  held-out session     " + "  ".join(f"n{nid:>2}" for nid in node_ids),
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
        lines.append(f"    node {nid:>2}: {np.mean(recalls):.3f}")
    lines.append("=" * 68)

    report = "\n".join(lines)
    print("\n" + report)

    rpt_path = out_dir / f"node_clf_v3_report_{ts}.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"\n  Report → {rpt_path}")

    return saved_paths


# ===========================================================================
# Phase 2 — YOLO camera pipeline
# ===========================================================================

def load_yolo_json(
    json_path:    Path,
    target_class: str   = 'car',
    min_conf:     float = 0.5,
    max_depth:    float = 35.0,
):
    """
    Load one node's YOLO JSON file into a flat DataFrame.

    JSON structure: [[det_dict, ...], ...]  (list of frames, each a list of dets)

    Filters applied:
      - class == target_class
      - conf >= min_conf
      - depth < max_depth  (depth=40.0 is ZED clipping artefact)

    Returns
    -------
    DataFrame: t_unix, x, y, z, conf, depth   (sorted by t_unix)
    """
    import pandas as _pd

    try:
        with open(json_path) as f:
            frames = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        import warnings as _warnings
        _warnings.warn(f"load_yolo_json: skipping malformed JSON {json_path}: {e}")
        return _pd.DataFrame(columns=["t_unix", "x", "y", "z", "conf", "depth"])

    records = []
    for frame in frames:
        if not isinstance(frame, list):
            continue
        for det in frame:
            if not isinstance(det, dict):
                continue
            if det.get("class") != target_class:
                continue
            conf = float(det.get("conf", 0.0))
            if conf < min_conf:
                continue
            depth = float(det.get("depth", 0.0))
            if depth >= max_depth:
                continue
            t_str = det.get("t", "")
            try:
                t_unix = datetime.strptime(t_str, _YOLO_TS_FMT).timestamp()
            except Exception:
                continue
            world = det.get("world", [0, 0, 0])
            if not (isinstance(world, list) and len(world) >= 2):
                continue
            records.append({
                "t_unix": t_unix,
                "x":      float(world[0]),
                "y":      float(world[1]),
                "z":      float(world[2]) if len(world) > 2 else 0.0,
                "conf":   conf,
                "depth":  depth,
            })

    if not records:
        return _pd.DataFrame(columns=["t_unix", "x", "y", "z", "conf", "depth"])

    df = _pd.DataFrame(records).sort_values("t_unix").reset_index(drop=True)
    return df


def find_static_objects(
    dets_df,
    bucket_size: float = 1.0,
    min_frames:  int   = 20,
    max_std:     float = 0.08,
) -> list[np.ndarray]:
    """
    Find world positions that remain stationary throughout the session.
    Returns list of (x, y) centroids to blacklist (e.g., parked cars).

    max_std=0.08m matches ZED depth noise at ~12 m range (~0.05–0.10 m typical).
    """
    if dets_df.empty:
        return []

    df = dets_df.copy()
    df["xb"] = (df["x"] / bucket_size).round() * bucket_size
    df["yb"] = (df["y"] / bucket_size).round() * bucket_size

    statics = []
    for (xb, yb), group in df.groupby(["xb", "yb"]):
        if len(group) < min_frames:
            continue
        if group["x"].std() < max_std and group["y"].std() < max_std:
            statics.append(np.array([group["x"].mean(), group["y"].mean()]))

    return statics


def filter_static_objects(
    dets_df,
    static_centroids:  list[np.ndarray],
    exclusion_radius:  float = 1.5,
):
    """Remove detections within exclusion_radius of any static centroid."""
    if not static_centroids or dets_df.empty:
        return dets_df

    positions = dets_df[["x", "y"]].values  # (M, 2)
    keep      = np.ones(len(dets_df), dtype=bool)

    for centroid in static_centroids:
        dist = np.sqrt(((positions - centroid) ** 2).sum(axis=1))
        keep &= (dist >= exclusion_radius)

    return dets_df[keep].reset_index(drop=True)


def build_p_cam_10(
    session_dir:        Path,
    node_ids:           list[int],
    session_start_unix: float,
    n_timesteps:        int,
    bin_size_s:         float = 0.5,
    target_class:       str   = 'car',
    min_conf:           float = 0.5,
    max_depth:          float = 35.0,
) -> np.ndarray:
    """
    Build the (T, N) camera probability matrix with strict per-node isolation.

    Each column k is populated ONLY from node k's own YOLO JSON file.
    P_cam[t, k] = max detection confidence in bin t for node k (0 if none).

    Parameters
    ----------
    session_dir         : session directory containing nodeN_zed_yolo.json files
    node_ids            : ordered list of node IDs (determines column index)
    session_start_unix  : session start as Unix timestamp (float seconds)
    n_timesteps         : T — length of output array
    bin_size_s          : bin width in seconds (must match audio step_s)

    Returns
    -------
    P_cam : (T, N) float32
    """
    N      = len(node_ids)
    P_cam  = np.zeros((n_timesteps, N), dtype=np.float32)

    for col_idx, nid in enumerate(node_ids):
        json_path = session_dir / f"node{nid}_zed_yolo.json"
        if not json_path.exists():
            print(f"  [camera] node{nid}: no JSON → P_cam[:,{col_idx}] = 0")
            continue

        dets = load_yolo_json(json_path, target_class, min_conf, max_depth)
        if dets.empty:
            print(f"  [camera] node{nid}: 0 filtered detections")
            continue

        # Remove static objects (e.g., parked car)
        statics = find_static_objects(dets)
        if statics:
            print(f"  [camera] node{nid}: removing {len(statics)} static object(s)")
        dets = filter_static_objects(dets, statics)
        if dets.empty:
            print(f"  [camera] node{nid}: 0 detections after static filter")
            continue

        # Bin detections to timesteps
        elapsed   = dets["t_unix"].values - session_start_unix
        bin_idxs  = np.floor(elapsed / bin_size_s).astype(int)

        for i, t in enumerate(bin_idxs):
            if 0 <= t < n_timesteps:
                conf = float(dets.iloc[i]["conf"])
                if conf > P_cam[t, col_idx]:
                    P_cam[t, col_idx] = conf

        pos_rate = (P_cam[:, col_idx] > 0).mean()
        n_active = int((P_cam[:, col_idx] > 0).sum())
        print(f"  [camera] node{nid}: {n_active}/{n_timesteps} active bins "
              f"({pos_rate:.1%} positive rate)")

    return P_cam


# ===========================================================================
# Phase 3 — OR fusion
# ===========================================================================

def get_audio_probabilities_10(
    clfs_dict:          dict[int, tuple],
    session_dir:        Path,
    n_timesteps:        int,
    step_s:             float = DEFAULT_STEP_S,
    window:             int   = WINDOW_STEPS,
) -> np.ndarray:
    """
    Run each trained classifier's predict_proba on its node's FLAC file.

    Parameters
    ----------
    clfs_dict    : {nid: (clf, scaler)} from train_final_classifiers()
    session_dir  : session directory containing nodeN_respeaker.flac files
    n_timesteps  : T — truncate output to this length

    Returns
    -------
    P_audio : (T, N) float32
    """
    import pandas as _pd

    node_ids = sorted(clfs_dict.keys())
    N        = len(node_ids)
    P_audio  = np.zeros((n_timesteps, N), dtype=np.float32)

    for col_idx, nid in enumerate(node_ids):
        flac_path = session_dir / f"node{nid}_respeaker.flac"
        if not flac_path.exists():
            print(f"  [audio] node{nid}: FLAC not found → P_audio[:,{col_idx}] = 0")
            continue

        clf, scaler = clfs_dict[nid]

        raw, T_flac = build_node_features(flac_path, step_s)
        T_use = min(T_flac, n_timesteps)

        X_scaled = scaler.transform(raw[:T_use])
        X_agg    = aggregate_rolling(X_scaled, window)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(
                _pd.DataFrame(X_agg, columns=FEATURE_NAMES)
            )[:, 1].astype(np.float32)
        else:
            proba = clf.predict(
                _pd.DataFrame(X_agg, columns=FEATURE_NAMES)
            ).astype(np.float32)

        P_audio[:T_use, col_idx] = proba

    return P_audio


def fuse_or(
    P_audio:         np.ndarray,
    P_cam:           np.ndarray,
    audio_threshold: float = 0.6,
    cam_threshold:   float = 0.3,
) -> np.ndarray:
    """
    OR fusion: B[t,k] = 1 if EITHER audio OR camera fires.

    Per-node isolation is preserved:
      P_audio[:,k] comes only from node k's audio classifier
      P_cam[:,k]   comes only from node k's camera JSON

    Parameters
    ----------
    audio_threshold : higher → fewer audio false positives (default 0.6)
    cam_threshold   : lower → trust camera at moderate confidence (default 0.3)

    Returns
    -------
    B : (T, N) bool
    """
    assert P_audio.shape == P_cam.shape, (
        f"Shape mismatch: P_audio {P_audio.shape} vs P_cam {P_cam.shape}"
    )
    return ((P_audio >= audio_threshold) | (P_cam >= cam_threshold))


def build_fused_detection_matrix_10(
    session_dir:     Path,
    clfs_dict:       dict[int, tuple],
    node_ids:        list[int],
    step_s:          float = DEFAULT_STEP_S,
    audio_threshold: float = 0.6,
    cam_threshold:   float = 0.3,
    target_class:    str   = 'car',
    min_conf_cam:    float = 0.5,
    max_depth:       float = 35.0,
) -> tuple[np.ndarray, int]:
    """
    End-to-end fused detection matrix for one session.

    Returns
    -------
    B           : (T, N) bool
    n_timesteps : T (= min FLAC duration in bins)
    """
    # Compute n_timesteps = min across all node FLACs
    file_map = _discover_session_flac_10(session_dir)
    T_list = []
    for nid in node_ids:
        fpath = file_map.get(nid)
        if fpath and fpath.exists():
            info = sf.info(str(fpath))
            T_list.append(int(info.duration / step_s))
    if not T_list:
        raise ValueError(f"No FLAC files found in {session_dir}")
    n_timesteps = min(T_list)

    session_start_dt   = _parse_session_start_10(session_dir.name)
    session_start_unix = session_start_dt.timestamp()

    print(f"\n[fusion] session={session_dir.name}  T={n_timesteps} bins "
          f"({n_timesteps * step_s:.1f} s)")

    # Audio probabilities
    print("\n[fusion] Building P_audio …")
    P_audio = get_audio_probabilities_10(clfs_dict, session_dir, n_timesteps, step_s)

    # Camera probabilities
    print("\n[fusion] Building P_cam …")
    P_cam = build_p_cam_10(
        session_dir, node_ids, session_start_unix, n_timesteps,
        bin_size_s=step_s, target_class=target_class,
        min_conf=min_conf_cam, max_depth=max_depth,
    )

    # Fuse
    B = fuse_or(P_audio, P_cam, audio_threshold, cam_threshold)

    # Summary
    audio_pos_rate = float((P_audio >= audio_threshold).mean())
    cam_pos_rate   = float((P_cam   >= cam_threshold  ).mean())
    fused_pos_rate = float(B.mean())
    print(f"\n[fusion summary]  T={n_timesteps} steps × {len(node_ids)} nodes")
    print(f"  Audio positive rate  (>={audio_threshold}): {audio_pos_rate:.3f}")
    print(f"  Camera positive rate (>={cam_threshold}): {cam_pos_rate:.3f}")
    print(f"  Fused positive rate  (OR):              {fused_pos_rate:.3f}")
    assert fused_pos_rate >= audio_pos_rate - 1e-6, \
        "OR fusion should never reduce positive rate vs audio-only"

    return B, n_timesteps


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-node binary classifier v3 (10-node IoBT) + YOLO camera fusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Validate data only
  python collection/train_node_classifiers_10.py \\
      --data-dir iobt_data_10 --validate-only

  # LOSO eval only (no saving)
  python collection/train_node_classifiers_10.py \\
      --data-dir iobt_data_10 --loso-only

  # Full train + YOLO fusion
  python collection/train_node_classifiers_10.py \\
      --data-dir iobt_data_10 --out-dir results/v3_10node --fuse
        """,
    )
    _DEFAULT_DATA = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iobt_data_10"
    )
    parser.add_argument("--data-dir",   type=Path, default=Path(_DEFAULT_DATA),
                        help="Directory containing session subdirs (default: iobt_data_10/)")
    parser.add_argument("--out-dir",    type=Path, default=Path("results/v3_10node"),
                        help="Output directory for classifiers and reports")
    parser.add_argument("--step",       type=float, default=DEFAULT_STEP_S,
                        help=f"Bin size in seconds (default: {DEFAULT_STEP_S})")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run data validation only; skip training")
    parser.add_argument("--loso-only",  action="store_true",
                        help="Run LOSO evaluation only; skip saving final classifiers")
    parser.add_argument("--fuse",       action="store_true",
                        help="After training, also run YOLO + OR fusion and save B matrix")
    parser.add_argument("--audio-threshold", type=float, default=0.6,
                        help="Audio classifier threshold for OR fusion (default: 0.6)")
    parser.add_argument("--cam-threshold",   type=float, default=0.3,
                        help="Camera confidence threshold for OR fusion (default: 0.3)")
    args = parser.parse_args()

    print("=" * 68)
    print("  PER-NODE BINARY CLASSIFIER v3 — 10-NODE IoBT")
    print("=" * 68)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Out dir    : {args.out_dir}")
    print(f"  Step size  : {args.step}s")
    print(f"  Window     : {WINDOW_STEPS} steps ({WINDOW_STEPS * args.step:.1f}s)")
    print(f"  Node IDs   : {EXPECTED_NODE_IDS}")

    # [0/4] Discover sessions
    print(f"\n[0/4] Discovering valid sessions in {args.data_dir} …")
    sessions = discover_valid_sessions_10(args.data_dir)
    if not sessions:
        print("[ERROR] No valid sessions found. Each session subdirectory needs:\n"
              "  - node{1..10}_respeaker.flac\n"
              "  - node{1..10}_zed_yolo.json\n"
              "  - meta_data.json\n"
              "  - gt_tracks.csv")
        sys.exit(1)
    print(f"  Found {len(sessions)} session(s): {sessions}")

    # [1/4] Validate all sessions
    print(f"\n[1/4] Validating session data …")
    all_valid = True
    for sess in sessions:
        ok = validate_session_data(args.data_dir / sess)
        if not ok:
            all_valid = False

    if not all_valid:
        print("\n[ERROR] One or more sessions failed validation. Fix issues before training.")
        sys.exit(1)

    if args.validate_only:
        print("\n[--validate-only] All sessions validated. Exiting.")
        sys.exit(0)

    # [2/4] Load per-node raw features
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[2/4] Loading per-node raw features …")
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    total_steps = 0
    for sess in sessions:
        print(f"\n  Session: {sess}")
        node_data = load_session_raw_per_node_10(args.data_dir / sess, args.step)
        session_data[sess] = node_data
        # Total GPS-labelled steps (use node 1 as reference)
        n1_steps = len(node_data[EXPECTED_NODE_IDS[0]][0])
        total_steps += n1_steps

    print(f"\n  Total GPS-labelled steps : {total_steps}")
    print(f"  Raw features per node    : {N_RAW_FEATURES}")
    print(f"  Features after rolling   : {N_FEATURES}")

    # [3/4] LOSO cross-validation
    print(f"\n[3/4] Running LOSO cross-validation …")
    if len(sessions) >= 2:
        loso_results = loso_per_node(session_data, EXPECTED_NODE_IDS, sessions)
    else:
        # Single session: 70/30 chronological split
        print("  [single session] Using 70/30 chronological split …")
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.preprocessing import StandardScaler
        import pandas as _pd

        sess = sessions[0]
        loso_results = {sess: {}}
        for nid in EXPECTED_NODE_IDS:
            X_raw, y_all = session_data[sess][nid]
            split = int(len(y_all) * 0.70)
            X_tr_raw, y_tr = X_raw[:split], y_all[:split]
            X_te_raw, y_te = X_raw[split:],  y_all[split:]
            if y_tr.sum() == 0:
                loso_results[sess][nid] = dict(
                    precision=0.0, recall=0.0, f1=0.0,
                    support=int(y_te.sum()), accuracy=float((y_te == 0).mean()),
                )
                continue
            scaler = StandardScaler()
            X_tr   = aggregate_rolling(scaler.fit_transform(X_tr_raw))
            X_te   = aggregate_rolling(scaler.transform(X_te_raw))
            clf    = make_classifier()
            clf.fit(_pd.DataFrame(X_tr, columns=FEATURE_NAMES), y_tr)
            y_pred = clf.predict(_pd.DataFrame(X_te, columns=FEATURE_NAMES))
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, y_pred, average="binary", zero_division=0,
            )
            loso_results[sess][nid] = dict(
                precision=float(prec), recall=float(rec), f1=float(f1),
                support=int(y_te.sum()), accuracy=float((y_pred == y_te).mean()),
            )
            print(f"  Node {nid:>2}: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    # [4/4] Final classifiers (+ optional fusion)
    if args.loso_only:
        print(f"\n[4/4] --loso-only: skipping final classifier training.")
    else:
        print(f"\n[4/4] Training final classifiers on all {total_steps} steps …")
        clfs = train_final_classifiers(session_data, EXPECTED_NODE_IDS, sessions)
        save_classifiers(
            clfs, loso_results, EXPECTED_NODE_IDS, sessions,
            total_steps, args.out_dir, ts,
        )

        # Optional: YOLO fusion
        if args.fuse:
            print(f"\n[FUSION] Building fused detection matrices …")
            for sess in sessions:
                sess_dir = args.data_dir / sess
                B, n_ts = build_fused_detection_matrix_10(
                    sess_dir, clfs, EXPECTED_NODE_IDS,
                    step_s=args.step,
                    audio_threshold=args.audio_threshold,
                    cam_threshold=args.cam_threshold,
                )

                # Also grab P_audio and P_cam for inspection
                session_start_unix = _parse_session_start_10(sess).timestamp()
                P_audio = get_audio_probabilities_10(
                    clfs, sess_dir, n_ts, args.step,
                )
                P_cam = build_p_cam_10(
                    sess_dir, EXPECTED_NODE_IDS, session_start_unix, n_ts,
                    bin_size_s=args.step,
                )

                # Shape assertions
                assert B.shape == P_audio.shape == P_cam.shape == (n_ts, len(EXPECTED_NODE_IDS)), \
                    f"Shape mismatch: B={B.shape} P_audio={P_audio.shape} P_cam={P_cam.shape}"

                # Save
                b_path = args.out_dir / f"fused_B_{sess}_{ts}.npy"
                pa_path = args.out_dir / f"p_audio_{sess}_{ts}.npy"
                pc_path = args.out_dir / f"p_cam_{sess}_{ts}.npy"
                np.save(b_path,  B)
                np.save(pa_path, P_audio)
                np.save(pc_path, P_cam)
                print(f"  Saved: {b_path.name}  shape={B.shape}")
                print(f"  Saved: {pa_path.name}")
                print(f"  Saved: {pc_path.name}")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)

    if not args.loso_only:
        print(f"\n  Use with finetune_deterministic.py:")
        print(f"    --node-clfs-dir {args.out_dir}")
        print(f"\n  Load classifiers:")
        print(f"    clf    = joblib.load('node_clf_N1_..._v3_*.pkl')")
        print(f"    scaler = joblib.load('node_scaler_N1_..._v3_*.pkl')")
        print(f"    X_feat = aggregate_rolling(scaler.transform(raw_10_features))")
        print(f"    pred   = clf.predict(pd.DataFrame(X_feat, columns=FEATURE_NAMES))")


if __name__ == "__main__":
    main()
