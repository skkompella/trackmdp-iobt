#!/usr/bin/env python3
"""
train_node_classifiers_pcen.py — Per-node binary classifiers v3.

v1 (original): 2 features — rolling mean+std of dB power, RandomForest.
  LOSO recall: node 11 ~0.50, nodes 13–16 <0.20.
v2 (PCEN attempt): 58 features — PCEN+MFCCs+deltas, LightGBM.
  LOSO recall: ALL NODES WORSE. PCEN normalizes away the amplitude that IS the signal.
v3 (this file): 20 features — amplitude-preserving feature set, LightGBM.
  Keeps dB power as primary signal; adds sub-band, impulsive, and cadence features
  that complement amplitude without removing it.

Feature pipeline per node per step:
  Raw FLAC → channels 1–4 averaged → DC-remove per step window
  → dB mean + std (amplitude)
  → sub-band dB in 4 frequency bands (amplitude-preserving spectral)
  → crest factor + kurtosis (impulsive event detection)
  → dynamic range (p90–p10 dB spread)
  → modulation energy 2–4 Hz (walking cadence)
  = 10 raw features
  → per-session z-score (leakage-free: fitted on training folds only)
  → causal rolling mean + std over 5 steps
  = 20 features per timestep

Outputs (to --out-dir):
    node_clf_N{nid}_multisession_v3_{TIMESTAMP}.pkl    (6 classifiers)
    node_scaler_N{nid}_multisession_v3_{TIMESTAMP}.pkl (6 scalers)
    node_clf_v3_report_{TIMESTAMP}.txt

Usage
-----
    python collection/train_node_classifiers_pcen.py \\
        --flac-dir  iobt_data \\
        --nodes-txt iobt_data/node_positions.txt \\
        --out-dir   results/v3

    python collection/train_node_classifiers_pcen.py \\
        --flac-dir  iobt_data \\
        --nodes-txt iobt_data/node_positions.txt \\
        --loso-only
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import kurtosis as scipy_kurtosis

EXPECTED_NODE_IDS = list(range(11, 17))   # [11, 12, 13, 14, 15, 16]
DEFAULT_STEP_S    = 0.5
POWER_CHANNELS    = slice(1, 5)           # ReSpeaker body mics, 0-indexed
EPSG_PROJ         = 32618                 # UTM zone 18N (Camp Buckner)
N_FFT_SUBBAND     = 512
HOP_SUBBAND       = 256
WINDOW_STEPS      = 5                     # causal rolling window in timesteps

_RAW_NAMES = [
    'db_mean', 'db_std',
    'band0_db', 'band1_db', 'band2_db', 'band3_db',
    'crest', 'kurtosis', 'dyn_range', 'mod_energy',
]
FEATURE_NAMES = [f'{n}_rmean' for n in _RAW_NAMES] + [f'{n}_rstd' for n in _RAW_NAMES]
N_RAW_FEATURES = len(_RAW_NAMES)   # 10
N_FEATURES     = len(FEATURE_NAMES)  # 20


# ---------------------------------------------------------------------------
# Geospatial utilities  (self-contained; geopandas optional)
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


def _latlon_to_utm(lats, lons, epsg: int = EPSG_PROJ) -> np.ndarray:
    if _GEO_AVAILABLE:
        gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in zip(lats, lons)],
            crs="EPSG:4326",
        ).to_crs(epsg=epsg)
        return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

    a   = 6_378_137.0
    f   = 1 / 298.257_223_563
    b   = a * (1 - f)
    e2  = 1 - (b / a) ** 2
    ep2 = e2 / (1 - e2)
    k0  = 0.9996
    E0  = 500_000.0
    lon0 = np.radians(-75.0)

    phi = np.radians(np.asarray(lats, dtype=np.float64))
    lam = np.radians(np.asarray(lons, dtype=np.float64))
    sin_phi = np.sin(phi); cos_phi = np.cos(phi); tan_phi = np.tan(phi)
    N_r = a / np.sqrt(1 - e2 * sin_phi ** 2)
    T   = tan_phi ** 2
    C   = ep2 * cos_phi ** 2
    A   = cos_phi * (lam - lon0)
    M = a * (
        (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*phi)
        + (15*e2**2/256 + 45*e2**3/1024)         * np.sin(4*phi)
        - (35*e2**3/3072)                         * np.sin(6*phi)
    )
    easting = k0 * N_r * (
        A + (1 - T + C) * A**3 / 6
        + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120
    ) + E0
    northing = k0 * (
        M + N_r * tan_phi * (
            A**2 / 2
            + (5 - T + 9*C + 4*C**2)               * A**4 / 24
            + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720
        )
    )
    return np.column_stack([easting, northing])


def load_node_positions_ordered(txt_path: Path, node_ids: list[int]) -> np.ndarray:
    import pandas as _pd
    df = _pd.read_csv(txt_path, sep='\t')
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    lat_col = col_map.get('latitude') or col_map.get('lat')
    lon_col = col_map.get('longitude') or col_map.get('lon')
    lats = df[lat_col].values.astype(float)
    lons = df[lon_col].values.astype(float)
    if len(lats) != len(node_ids):
        raise ValueError(
            f"{txt_path.name} has {len(lats)} rows but {len(node_ids)} node IDs given"
        )
    return _latlon_to_utm(lats, lons)


def load_gps_track(gps_csv: Path):
    import pandas as _pd
    df = _pd.read_csv(gps_csv)
    df['timestamp'] = _pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    xy = _latlon_to_utm(df['Latitude'].values, df['Longitude'].values)
    df['x'] = xy[:, 0]
    df['y'] = xy[:, 1]
    return df[['timestamp', 'x', 'y']].reset_index(drop=True)


def _parse_session_start(session: str) -> datetime:
    return datetime.strptime(session, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def build_ground_truth_sequence(
    gps_df,
    node_xy:    np.ndarray,
    node_ids:   list[int],
    flac_start: datetime,
    step_s:     float,
    T:          int,
) -> np.ndarray:
    gps_times = gps_df['timestamp'].values
    gps_x     = gps_df['x'].values.astype(np.float64)
    gps_y     = gps_df['y'].values.astype(np.float64)
    t0_ns     = np.datetime64(flac_start.replace(tzinfo=None), 'ns')
    gt        = np.zeros(T, dtype=np.int32)

    for t in range(T):
        t_ns = t0_ns + np.timedelta64(int(t * step_s * 1e9), 'ns')
        idx  = np.searchsorted(gps_times, t_ns)

        if idx == 0:
            if gps_times[0] == t_ns:
                pos_x, pos_y = gps_x[0], gps_y[0]
            else:
                continue
        elif idx >= len(gps_times):
            continue
        else:
            t_lo = gps_times[idx - 1]; t_hi = gps_times[idx]
            span = float((t_hi - t_lo) / np.timedelta64(1, 's'))
            frac = float((t_ns  - t_lo) / np.timedelta64(1, 's')) / max(span, 1e-9)
            pos_x = gps_x[idx-1] + frac * (gps_x[idx] - gps_x[idx-1])
            pos_y = gps_y[idx-1] + frac * (gps_y[idx] - gps_y[idx-1])

        dx   = pos_x - node_xy[:, 0]
        dy   = pos_y - node_xy[:, 1]
        near = int(np.argmin(dx**2 + dy**2))
        gt[t] = node_ids[near]

    return gt


def _discover_session_flac(flac_dir: Path, session: str) -> dict[int, Path]:
    pattern  = re.compile(r'orin_(\d+)', re.IGNORECASE)
    file_map: dict[int, Path] = {}
    for fpath in sorted(flac_dir.glob(f"{session}*.flac")):
        m = pattern.search(fpath.name)
        if m:
            file_map[int(m.group(1))] = fpath
    return file_map


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def discover_valid_sessions(flac_dir: Path) -> list[str]:
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
# Feature extraction — v3 amplitude-preserving pipeline
# ---------------------------------------------------------------------------

def build_node_features(
    flac_path: Path,
    step_s:    float = DEFAULT_STEP_S,
) -> tuple[np.ndarray, int]:
    """
    Extract 10 raw features per GPS timestep from a single node's FLAC file.

    All features preserve or complement the acoustic amplitude signal:
      0  db_mean    — mean dB power per step (primary proximity signal)
      1  db_std     — std of per-frame dB within step
      2  band0_db   — mean dB in  0–500 Hz (footstep/vehicle fundamental)
      3  band1_db   — mean dB in 500–2000 Hz (harmonics)
      4  band2_db   — mean dB in 2000–4000 Hz (surface texture)
      5  band3_db   — mean dB in 4000–sr/2 Hz (background/wind)
      6  crest      — peak / rms (footsteps are impulsive: high crest)
      7  kurtosis   — 4th moment peakedness (Pearson; random noise ≈ 3, impulses >> 3)
      8  dyn_range  — dB p90 – p10 (spread of energy fluctuation)
      9  mod_energy — RMS-envelope spectrum energy in 2–(1/step_s) Hz band (walking cadence)

    Returns
    -------
    raw_features : (T, 10) float64
    T            : number of steps (= total_samples // win_samples)
    """
    data, sr = sf.read(str(flac_path), dtype='int16', always_2d=True)
    win_samples = int(round(step_s * sr))
    T           = len(data) // win_samples

    # Sub-band setup from actual sr
    stft_freqs  = np.fft.rfftfreq(N_FFT_SUBBAND, d=1.0 / sr)
    band_edges  = [(0, 500), (500, 2000), (2000, 4000), (4000, sr // 2)]
    band_masks  = []
    for lo, hi in band_edges:
        mask = (stft_freqs >= lo) & (stft_freqs < hi)
        band_masks.append(mask if mask.any() else np.zeros(len(stft_freqs), dtype=bool))

    # Modulation band: 2 Hz to Nyquist of envelope (1/step_s)
    mod_lo = 2.0
    mod_hi = 1.0 / step_s  # Nyquist of 0.5s step = 1 Hz, but we use 2-4 Hz

    hop   = HOP_SUBBAND
    out   = np.zeros((T, N_RAW_FEATURES), dtype=np.float64)

    for t in range(T):
        chunk = data[t * win_samples : (t + 1) * win_samples]   # (win, 6)
        # Average body mics, DC-remove
        x = chunk[:, POWER_CHANNELS].mean(axis=1).astype(np.float64)
        x -= x.mean()

        # --- Amplitude features ---
        n_hop_frames  = max(1, win_samples // hop)
        frame_powers  = np.array([
            np.mean(x[i*hop : i*hop + hop] ** 2) + 1e-12
            for i in range(n_hop_frames)
        ])
        frame_db = 10.0 * np.log10(frame_powers)

        out[t, 0] = float(frame_db.mean())                    # db_mean
        out[t, 1] = float(frame_db.std()) if len(frame_db) > 1 else 0.0   # db_std

        # --- Sub-band dB ---
        S = np.abs(np.fft.rfft(x, n=N_FFT_SUBBAND))          # single FFT of full chunk
        S_power = S ** 2
        for b, mask in enumerate(band_masks):
            if mask.any():
                out[t, 2 + b] = 10.0 * np.log10(S_power[mask].mean() + 1e-10)

        # --- Impulsive features ---
        rms  = np.sqrt(np.mean(x ** 2)) + 1e-10
        peak = np.max(np.abs(x))
        out[t, 6] = float(peak / rms)                         # crest factor
        out[t, 7] = float(scipy_kurtosis(x, fisher=False))    # Pearson kurtosis

        # --- Dynamic range ---
        out[t, 8] = float(np.percentile(frame_db, 90) - np.percentile(frame_db, 10))

        # --- Modulation energy ---
        if len(frame_db) >= 4:
            env_fft   = np.abs(np.fft.rfft(frame_db))
            env_freqs = np.fft.rfftfreq(len(frame_db), d=step_s / n_hop_frames)
            mod_mask  = (env_freqs >= mod_lo) & (env_freqs <= mod_hi)
            out[t, 9] = float(np.mean(env_fft[mod_mask] ** 2)) if mod_mask.any() else 0.0

    return out, T


# ---------------------------------------------------------------------------
# Rolling aggregation (operates on already z-scored raw features)
# ---------------------------------------------------------------------------

def aggregate_rolling(raw_scaled: np.ndarray, window: int = WINDOW_STEPS) -> np.ndarray:
    """
    Causal rolling mean + std over a window of timesteps.

    Parameters
    ----------
    raw_scaled : (T, 10) z-scored raw features
    window     : rolling window in timesteps

    Returns
    -------
    (T, 20) float64
    """
    import pandas as _pd
    df       = _pd.DataFrame(raw_scaled)
    roll_mean = df.rolling(window=window, min_periods=1).mean().values
    roll_std  = df.rolling(window=window, min_periods=1).std().fillna(0).values
    return np.hstack([roll_mean, roll_std])


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference utility
# ---------------------------------------------------------------------------

def smooth_predictions(
    proba: np.ndarray,
    window: int = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Rolling mean smoothing on classifier probabilities."""
    import pandas as _pd
    smoothed = _pd.Series(proba).rolling(window=window, min_periods=1).mean()
    return (smoothed.values >= threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# Per-session raw data loading  (no z-score — done inside LOSO/final loops)
# ---------------------------------------------------------------------------

def load_session_raw_per_node(
    session:   str,
    flac_dir:  Path,
    nodes_txt: Path,
    step_s:    float = DEFAULT_STEP_S,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Load one session and return per-node raw (un-z-scored) feature matrices.

    Z-scoring is deferred to the LOSO/final-training loops to prevent data
    leakage (scaler must be fitted on training folds only).

    Returns
    -------
    dict: nid -> (raw_features, y_binary)
        raw_features : (M, 10) GPS-labelled steps only, un-normalised
        y_binary     : (M,) int32 — 1 if person at nid, 0 otherwise
    """
    file_map = _discover_session_flac(flac_dir, session)
    node_ids = sorted(file_map.keys())

    # Extract raw features per node; also record T (common across nodes)
    node_raw: dict[int, tuple[np.ndarray, int]] = {}
    for nid, fpath in file_map.items():
        print(f"    {session} node {nid}: extracting features …", flush=True)
        raw, T = build_node_features(fpath, step_s)
        node_raw[nid] = (raw, T)

    T_min = min(info[1] for info in node_raw.values())

    node_xy    = load_node_positions_ordered(nodes_txt, node_ids)
    gps_df     = load_gps_track(flac_dir / f"{session}_gps2_gps.csv")
    flac_start = _parse_session_start(session)
    gt         = build_ground_truth_sequence(
        gps_df, node_xy, node_ids, flac_start, step_s, T_min,
    )

    valid   = gt > 0
    n_valid = int(valid.sum())
    print(f"    {session}: T={T_min}  GPS-labelled={n_valid} ({n_valid/T_min*100:.1f}%)")

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nid in node_ids:
        raw, _ = node_raw[nid]
        y_bin  = (gt == nid).astype(np.int32)
        result[nid] = (raw[:T_min][valid], y_bin[valid])

    return result


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

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

            scaler   = StandardScaler()
            X_tr     = aggregate_rolling(scaler.fit_transform(X_tr_raw), window)
            X_te     = aggregate_rolling(scaler.transform(X_te_raw), window)

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
        print(f"    node {nid}: {np.mean(recalls):.3f}  "
              f"({', '.join(f'{r:.2f}' for r in recalls)})")
    print("=" * 68)

    return results


# ---------------------------------------------------------------------------
# Final classifiers (trained on all sessions pooled)
# ---------------------------------------------------------------------------

def train_final_classifiers(
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    node_ids:     list[int],
    sessions:     list[str],
    window:       int = WINDOW_STEPS,
) -> dict[int, tuple[object, object]]:
    """Returns dict: nid -> (fitted_clf, fitted_scaler)."""
    from sklearn.preprocessing import StandardScaler
    import pandas as _pd

    clfs: dict[int, tuple[object, object]] = {}
    for nid in node_ids:
        X_raw = np.vstack([session_data[s][nid][0] for s in sessions])
        y_all = np.concatenate([session_data[s][nid][1] for s in sessions])
        pos_rate = y_all.mean()
        print(f"  Node {nid}: {int(y_all.sum())} positive / {len(y_all)} total "
              f"({pos_rate*100:.1f}%)")

        scaler = StandardScaler()
        X_all  = aggregate_rolling(scaler.fit_transform(X_raw), window)

        clf = make_classifier()
        clf.fit(_pd.DataFrame(X_all, columns=FEATURE_NAMES), y_all)
        clfs[nid] = (clf, scaler)

    return clfs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

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
        "  PER-NODE BINARY CLASSIFIER v3 — AMPLITUDE-PRESERVING FEATURES",
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

    rpt_path = out_dir / f"node_clf_v3_report_{ts}.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"\n  Report → {rpt_path}")

    return saved_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-node binary classifier v3 — amplitude-preserving features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python collection/train_node_classifiers_pcen.py \\
      --flac-dir  iobt_data \\
      --nodes-txt iobt_data/node_positions.txt \\
      --out-dir   results/v3
        """,
    )
    _D = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iobt_data"
    )
    parser.add_argument("--flac-dir",  type=Path, default=Path(_D),
                        help="Directory containing FLAC files and GPS CSVs")
    parser.add_argument("--nodes-txt", type=Path,
                        default=Path(os.path.join(_D, "node_positions.txt")),
                        help="Tab-delimited node_positions.txt")
    parser.add_argument("--step",      type=float, default=DEFAULT_STEP_S,
                        help=f"FLAC step size in seconds (default: {DEFAULT_STEP_S})")
    parser.add_argument("--out-dir",   type=Path, default=Path("results/v3"),
                        help="Output directory (default: results/v3)")
    parser.add_argument("--loso-only", action="store_true", default=False,
                        help="Run LOSO evaluation only; skip saving final classifiers")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 68)
    print("  PER-NODE BINARY CLASSIFIER v3 TRAINING")
    print("=" * 68)
    print(f"  FLAC dir   : {args.flac_dir}")
    print(f"  Nodes txt  : {args.nodes_txt}")
    print(f"  Step size  : {args.step}s")
    print(f"  Window     : {WINDOW_STEPS} steps ({WINDOW_STEPS * args.step:.1f}s)")
    print(f"  Out dir    : {args.out_dir}")
    print(f"  LOSO only  : {args.loso_only}")
    print(f"  Features   : {N_FEATURES} (rolling mean+std of {N_RAW_FEATURES} raw)")

    # [1/4] Discover sessions
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

    # [2/4] Load all sessions (raw features, no z-score)
    print(f"\n[2/4] Loading per-node raw features for all sessions …")
    session_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    total_steps = 0
    for sess in sessions:
        node_data = load_session_raw_per_node(
            sess, args.flac_dir, args.nodes_txt, args.step,
        )
        session_data[sess] = node_data
        total_steps += len(node_data[EXPECTED_NODE_IDS[0]][0])

    print(f"\n  Total GPS-labelled steps : {total_steps}")
    print(f"  Node IDs                 : {EXPECTED_NODE_IDS}")
    print(f"  Raw features per node    : {N_RAW_FEATURES}")
    print(f"  Features after rolling   : {N_FEATURES}")

    # [3/4] LOSO cross-validation
    print(f"\n[3/4] Running LOSO cross-validation …")
    if len(sessions) >= 2:
        loso_results = loso_per_node(session_data, EXPECTED_NODE_IDS, sessions)
    else:
        # Single session: 70/30 chronological split fallback
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
            print(f"  Node {nid}: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    # [4/4] Final classifiers
    if not args.loso_only:
        print(f"\n[4/4] Training final classifiers on all {total_steps} steps …")
        clfs = train_final_classifiers(session_data, EXPECTED_NODE_IDS, sessions)
        save_classifiers(
            clfs, loso_results, EXPECTED_NODE_IDS, sessions,
            total_steps, args.out_dir, ts,
        )
        print("\n" + "=" * 68)
        print("  DONE")
        print("=" * 68)
        print("\n  Use with finetune_deterministic.py:")
        print(f"    --node-clfs-dir {args.out_dir}")
        print(f"\n  Note: each node also has a scaler pkl — load both for inference:")
        print(f"    clf    = joblib.load('node_clf_N11_..._v3_*.pkl')")
        print(f"    scaler = joblib.load('node_scaler_N11_..._v3_*.pkl')")
        print(f"    X_feat = aggregate_rolling(scaler.transform(raw_10_features))")
        print(f"    pred   = clf.predict(pd.DataFrame(X_feat, columns=FEATURE_NAMES))")
    else:
        print(f"\n[4/4] --loso-only: skipping final classifier training.")
        print("\n" + "=" * 68)
        print("  DONE")
        print("=" * 68)


if __name__ == "__main__":
    main()
