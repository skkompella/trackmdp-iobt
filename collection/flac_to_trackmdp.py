#!/usr/bin/env python3
"""
flac_to_trackmdp.py — Convert per-node ReSpeaker FLAC files into a ranked
node array suitable as Track-MDP input.

Signal chain (mirrors respeaker_custom.py + RealTimeGUI exactly)
-----------------------------------------------------------------
1. Load each node's FLAC (16 kHz, 6-channel int16).
2. Chunk into non-overlapping windows of `step_s` seconds.
3. Per window per node:
     a. Extract channels 1–4 (indices 1:5), average across mics.
     b. DC-remove (subtract mean).
     c. Compute mean squared power (linear).
     d. Convert to dB:  db = 10 * log10(max(power, 1e-12))
4. Apply rolling mean of `rssi_window` consecutive dB readings per node.
5. For each timestep run the Gaussian path-loss Bayesian posterior over the
   hypothesis grid (same vectorised computation as RealTimeGUI.compute_posterior).
6. The node nearest the top-1 posterior hypothesis is the output for that step.

Output
------
  rankings.npy      int32 (T,)   — 1-indexed node ID of #1 ranked source each step
  rankings_meta.json             — step_s, nodes found, total steps, timestamps

Usage
-----
    python flac_to_trackmdp.py --flac-dir ./data/audio  \\
                               --nodes-csv ./data/nodes_gps.csv \\
                               --path-loss ./data/TRUCK_path_loss_params.json

    python flac_to_trackmdp.py --flac-dir ./data/audio \\
                               --nodes-csv ./data/nodes_gps.csv \\
                               --path-loss ./data/ATV_path_loss_params.json \\
                               --step 0.2 --rssi-window 5 \\
                               --out-dir ./results
"""

import argparse
import json
import os
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Optional geospatial imports (needed for node XY positions)
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False
    print("[WARN] geopandas / shapely not installed. "
          "Node positions will be loaded from CSV x/y columns directly.")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

RESPEAKER_RATE     = 16000
RESPEAKER_CHANNELS = 6
POWER_CHANNELS     = slice(1, 5)   # channels 1,2,3,4 (0-indexed)

CANDIDATE_OFFSETS = np.arange(-20, 20)   # 20×20 hypothesis grid
SPACING           = 4                    # metres (projected CRS)
EPSG_PROJ         = 32618                # UTM zone 18N (Camp Buckner)

DEFAULT_STEP_S      = 0.5
DEFAULT_RSSI_WINDOW = 5


# ---------------------------------------------------------------------------
# File discovery — find FLAC files and infer node IDs from names
# ---------------------------------------------------------------------------

def discover_flac_files(flac_dir: Path,
                        session: str | None = None) -> dict[int, Path]:
    """
    Scan flac_dir for *.flac files and map each to a node ID.

    Node ID is parsed from the filename by looking for:
      - "node<N>"          e.g. node3_respeaker_20250812.flac  -> 3
      - "orin_<N>"         e.g. dvpg_gq_orin_11_respeaker.flac -> 11
    If multiple files map to the same node, the lexicographically last
    (most recent) is kept.

    If `session` is given (e.g. "20260416_154037"), only files whose name
    starts with that prefix are considered.
    """
    pattern = re.compile(r'(?:node|orin_)(\d+)', re.IGNORECASE)
    found: dict[int, Path] = {}
    for fpath in sorted(flac_dir.glob("*.flac")):
        if session and not fpath.name.startswith(session):
            continue
        m = pattern.search(fpath.name)
        if m:
            nid = int(m.group(1))
            found[nid] = fpath
        else:
            print(f"  [WARN] Cannot infer node ID from {fpath.name} — skipping.")
    return found


# ---------------------------------------------------------------------------
# Node positions
# ---------------------------------------------------------------------------

def load_node_positions(nodes_csv: Path, node_ids: list[int]) -> np.ndarray:
    """
    Load node (x, y) positions in projected metres from nodes_gps.csv.

    Returns NODE_XY: float64 (N, 2) aligned to node_ids order.

    If geopandas is available, reprojects from EPSG:4326 to EPSG_PROJ.
    Otherwise expects columns 'X' and 'Y' already in metres.
    """
    if _GEO_AVAILABLE:
        df = pd.read_csv(nodes_csv)
        # Normalise column names (case-insensitive)
        df.columns = [c.strip() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}
        node_col = col_map.get('node #') or col_map.get('node#') or col_map.get('node_id') or 'Node #'
        lat_col  = col_map.get('lat') or 'Lat'
        lon_col  = col_map.get('lon') or 'Lon'

        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lat, lon in zip(df[lat_col], df[lon_col])],
            crs="EPSG:4326"
        ).to_crs(epsg=EPSG_PROJ)

        gdf = gdf.rename(columns={node_col: 'node_id'})
        gdf['node_id'] = gdf['node_id'].astype(int)

        xy = []
        missing = []
        for nid in node_ids:
            row = gdf[gdf['node_id'] == nid]
            if row.empty:
                missing.append(nid)
                xy.append([np.nan, np.nan])
            else:
                xy.append([row.iloc[0].geometry.x, row.iloc[0].geometry.y])
        if missing:
            print(f"  [WARN] Nodes not found in CSV: {missing}")
        return np.array(xy, dtype=np.float64)

    else:
        # Fallback: expect X, Y columns already in metres
        import csv
        rows = {}
        with open(nodes_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = int(row.get('Node #') or row.get('node_id'))
                rows[nid] = (float(row['X']), float(row['Y']))
        return np.array([[rows[n][0], rows[n][1]] for n in node_ids], dtype=np.float64)


# ---------------------------------------------------------------------------
# Path-loss parameters
# ---------------------------------------------------------------------------

def load_path_loss(json_path: Path, node_ids: list[int]):
    """
    Load path-loss JSON and return (P0, ETA, SIG2, MASK) arrays aligned to node_ids.

    JSON format: {"1": [p0, eta, sig2], "2": [...], ...}
    """
    with open(json_path) as f:
        pl = {int(k): v for k, v in json.load(f).items()}

    P0, ETA, SIG2, MASK = [], [], [], []
    for nid in node_ids:
        if nid in pl:
            p0, eta, s2 = pl[nid]
            P0.append(float(p0))
            ETA.append(float(eta))
            SIG2.append(max(float(s2), 1e-6))
            MASK.append(True)
        else:
            print(f"  [WARN] Node {nid} missing in path-loss JSON — masked out.")
            P0.append(0.0); ETA.append(2.0); SIG2.append(1.0); MASK.append(False)

    return (np.array(P0), np.array(ETA), np.array(SIG2), np.array(MASK, dtype=bool))


# ---------------------------------------------------------------------------
# Training helpers — lat/lon projection, GPS track, path-loss fitting
# ---------------------------------------------------------------------------

def _latlon_to_utm(lats, lons, epsg: int = EPSG_PROJ) -> np.ndarray:
    """
    Project lat/lon (WGS84) arrays to UTM metres. Returns (N, 2) float64.

    Uses geopandas when available; otherwise falls back to a pure-numpy
    Helmert-series Transverse Mercator accurate to <1 m for the region.
    The fallback hard-codes UTM zone 18N (EPSG:32618, central meridian -75°),
    which covers the Camp Buckner / NJ area.  Pass a different epsg only when
    geopandas is installed.
    """
    if _GEO_AVAILABLE:
        gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in zip(lats, lons)],
            crs="EPSG:4326",
        ).to_crs(epsg=epsg)
        return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

    # ── Pure-numpy fallback (UTM zone 18N / EPSG:32618) ──────────────────
    # WGS84 ellipsoid
    a   = 6_378_137.0
    f   = 1 / 298.257_223_563
    b   = a * (1 - f)
    e2  = 1 - (b / a) ** 2
    ep2 = e2 / (1 - e2)
    k0  = 0.9996
    E0  = 500_000.0
    lon0 = np.radians(-75.0)   # central meridian for zone 18

    phi = np.radians(np.asarray(lats, dtype=np.float64))
    lam = np.radians(np.asarray(lons, dtype=np.float64))

    sin_phi = np.sin(phi);  cos_phi = np.cos(phi);  tan_phi = np.tan(phi)
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
            + (5 - T + 9*C + 4*C**2)           * A**4 / 24
            + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720
        )
    )
    return np.column_stack([easting, northing])


def load_node_positions_ordered(txt_path: Path, node_ids: list[int]) -> np.ndarray:
    """
    Load node positions from a tab-delimited file (Latitude, Longitude header,
    no node-ID column). Row i maps to node_ids[i].
    Returns UTM (N, 2) float64.
    """
    import pandas as _pd
    df = _pd.read_csv(txt_path, sep='\t')
    df.columns = [c.strip() for c in df.columns]
    col_map   = {c.lower(): c for c in df.columns}
    lat_col   = col_map.get('latitude') or col_map.get('lat')
    lon_col   = col_map.get('longitude') or col_map.get('lon')
    lats = df[lat_col].values.astype(float)
    lons = df[lon_col].values.astype(float)
    if len(lats) != len(node_ids):
        raise ValueError(
            f"{txt_path.name} has {len(lats)} rows but {len(node_ids)} node IDs given"
        )
    return _latlon_to_utm(lats, lons)


def load_gps_track(gps_csv: Path):
    """
    Parse a GPS ground-truth CSV (Timestamp, Latitude, Longitude, …).
    Timestamp format: '2026/04/16 15:40:38.068'
    Returns a pandas DataFrame with columns: timestamp (UTC datetime), x, y (UTM metres).
    """
    import pandas as _pd
    df = _pd.read_csv(gps_csv)
    df['timestamp'] = _pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    xy = _latlon_to_utm(df['Latitude'].values, df['Longitude'].values)
    df['x'] = xy[:, 0]
    df['y'] = xy[:, 1]
    return df[['timestamp', 'x', 'y']].reset_index(drop=True)


def _parse_session_start(session: str) -> datetime:
    """'20260416_154037' -> timezone-aware UTC datetime."""
    return datetime.strptime(session, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def build_training_pairs(
    gps_df,
    node_xy:      np.ndarray,
    power_arrays: dict,
    node_ids:     list[int],
    flac_start:   datetime,
    step_s:       float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align GPS timestamps with FLAC power windows.

    For each GPS row, find the corresponding FLAC step and compute:
      - Euclidean distances from GPS position to every node (UTM metres)
      - Instantaneous dB power at each node (no rolling average)

    Returns:
      distances (M, N) float64
      rssi_db   (M, N) float64
      gps_xy    (M, 2) float64  — GPS positions in UTM for the M valid samples
    """
    T = min(len(a) for a in power_arrays.values())
    dist_rows, rssi_rows, xy_rows = [], [], []
    skipped = 0
    for _, row in gps_df.iterrows():
        t_offset = (row['timestamp'] - flac_start).total_seconds()
        step_idx = int(t_offset / step_s)
        if step_idx < 0 or step_idx >= T:
            skipped += 1
            continue
        dx    = row['x'] - node_xy[:, 0]
        dy    = row['y'] - node_xy[:, 1]
        dists = np.clip(np.sqrt(dx ** 2 + dy ** 2), 1e-3, None)
        rssi_vec = np.array([
            10.0 * np.log10(max(float(power_arrays[nid][step_idx]), 1e-12))
            for nid in node_ids
        ])
        dist_rows.append(dists)
        rssi_rows.append(rssi_vec)
        xy_rows.append([row['x'], row['y']])

    if skipped:
        print(f"  [WARN] {skipped} GPS samples outside FLAC time range — skipped.")

    return (
        np.array(dist_rows,  dtype=np.float64),
        np.array(rssi_rows,  dtype=np.float64),
        np.array(xy_rows,    dtype=np.float64),
    )


_R2_MASK_THRESHOLD = 0.05   # mute node in posterior if constrained R² < this


def fit_path_loss(
    distances:     np.ndarray,
    rssi_db:       np.ndarray,
    node_ids:      list[int],
    r2_threshold:  float = _R2_MASK_THRESHOLD,
) -> dict:
    """
    Fit RSSI = P0 - 10*eta*log10(d) per node with eta >= 0 (physical constraint).

    Unconstrained lstsq can yield negative eta (RSSI increasing with distance),
    which is physically impossible and breaks the Bayesian posterior.
    scipy.optimize.lsq_linear is used to enforce eta >= 0.

    Nodes whose constrained R² < r2_threshold are unreliable: their sigma2 is set
    to 1000 so the posterior treats them as uninformative (flat log-likelihood).

    Returns {"node_id": [P0, eta, sigma2], …} (string keys for load_path_loss compat).
    """
    from scipy.optimize import lsq_linear

    params = {}
    for j, nid in enumerate(node_ids):
        d = distances[:, j]
        r = rssi_db[:, j]
        valid = d >= 0.1
        d, r = d[valid], r[valid]
        if len(d) < 3:
            print(f"  [WARN] node {nid}: only {len(d)} valid samples — using defaults.")
            params[str(nid)] = [-30.0, 2.0, 1.0]
            continue

        # Design matrix: x = [P0, eta], columns = [1, -10*log10(d)]
        # RSSI = P0 - 10*eta*log10(d);  constraint: eta >= 0
        A = np.column_stack([np.ones(len(d)), -10.0 * np.log10(d)])
        res    = lsq_linear(A, r, bounds=([-np.inf, 0.0], [np.inf, np.inf]))
        P0, eta = float(res.x[0]), float(res.x[1])

        pred   = A @ res.x
        resid  = r - pred
        sigma2 = float(max(np.var(resid), 1e-4))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        reliable = r2 >= r2_threshold
        if not reliable:
            # High sigma2 makes this node's log-likelihood flat across all hypotheses
            # so it contributes nothing to the posterior — effectively masked.
            sigma2 = 1000.0
            print(f"  node {nid:3d}: P0={P0:+.2f} dB  η={eta:.3f}  R²={r2:.3f}  "
                  f"→ MUTED (R² < {r2_threshold})")
        else:
            print(f"  node {nid:3d}: P0={P0:+.2f} dB  η={eta:.3f}  σ²={sigma2:.4f}  "
                  f"R²={r2:.3f}  residual_σ={np.sqrt(sigma2):.3f} dB  (N={len(d)})")

        params[str(nid)] = [P0, eta, sigma2]
    return params


def fit_supervised_classifier(
    rssi_db:   np.ndarray,
    distances: np.ndarray,
    node_ids:  list[int],
) -> dict:
    """
    Nearest-centroid classifier trained from GPS-labelled RSSI vectors.

    Labels: y[i] = argmin(distances[i]) — index of nearest node (0-based).
    Features are z-scored per node before centroid computation.

    Returns a JSON-serialisable dict with keys:
        method, node_ids, scaler_mean, scaler_std, centroids, class_counts.
    Pass the returned dict directly to predict_supervised_sequence().
    """
    N      = len(node_ids)
    labels = np.argmin(distances, axis=1)       # (M,) nearest node index

    mean = rssi_db.mean(axis=0)                 # (N,) per-node dB mean
    std  = rssi_db.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)       # avoid divide-by-zero on flat channels
    X    = (rssi_db - mean) / std               # (M, N) z-scored

    n_features   = rssi_db.shape[1]
    centroids    = np.zeros((N, n_features))
    class_counts = np.zeros(N, dtype=int)
    for k in range(N):
        idx = np.where(labels == k)[0]
        class_counts[k] = len(idx)
        if len(idx) > 0:
            centroids[k] = X[idx].mean(axis=0)
        print(f"  cell {k} (node {node_ids[k]}): {len(idx)} training samples")

    return {
        "method":       "nearest_centroid",
        "node_ids":     node_ids,
        "scaler_mean":  mean.tolist(),
        "scaler_std":   std.tolist(),
        "centroids":    centroids.tolist(),
        "class_counts": class_counts.tolist(),
    }


def predict_supervised_sequence(
    power_arrays: dict,
    node_ids:     list[int],
    classifier:   dict,
    step_s:       float,
) -> np.ndarray:
    """
    Apply the nearest-centroid classifier to every FLAC timestep.
    Returns int32 (T,) of cell indices 0-(N-1).
    """
    T    = min(len(power_arrays[nid]) for nid in node_ids)
    mean = np.array(classifier["scaler_mean"])
    std  = np.array(classifier["scaler_std"])
    C    = np.array(classifier["centroids"])    # (N_classes, N_nodes)

    positions = np.zeros(T, dtype=np.int32)
    for t in range(T):
        rssi = np.array([
            10.0 * np.log10(max(float(power_arrays[nid][t]), 1e-12))
            for nid in node_ids
        ])
        z             = (rssi - mean) / std
        dists         = np.linalg.norm(C - z, axis=1)  # (N_classes,)
        positions[t]  = int(np.argmin(dists))

    return positions


def evaluate_fit(
    params_dict: dict,
    node_ids:    list[int],
    node_xy:     np.ndarray,
    gps_xy:      np.ndarray,
    distances:   np.ndarray,
    rssi_db:     np.ndarray,
) -> None:
    """
    Evaluate fitted params on training pairs.
    Reports nearest-node classification accuracy and mean position error (m).
    """
    P0_a, ETA_a, SIG2_a, MASK_a = [], [], [], []
    for nid in node_ids:
        entry = params_dict.get(str(nid))
        if entry:
            P0_a.append(entry[0]); ETA_a.append(entry[1])
            SIG2_a.append(max(entry[2], 1e-6)); MASK_a.append(True)
        else:
            P0_a.append(0.0); ETA_a.append(2.0); SIG2_a.append(1.0); MASK_a.append(False)
    P0_a   = np.array(P0_a);   ETA_a  = np.array(ETA_a)
    SIG2_a = np.array(SIG2_a); MASK_a = np.array(MASK_a, dtype=bool)

    hypos, dist_h, nearest_idx, _, _ = build_hypothesis_grid(node_xy)

    correct, pos_errors = 0, []
    M = len(distances)
    for i in range(M):
        rss_map   = {nid: float(rssi_db[i, j]) for j, nid in enumerate(node_ids)}
        pred_node = compute_posterior_top1(
            rss_map, node_ids, dist_h, nearest_idx,
            P0_a, ETA_a, SIG2_a, MASK_a, hypos,
        )
        true_node = node_ids[int(np.argmin(distances[i]))]
        if pred_node == true_node:
            correct += 1
        if pred_node is not None:
            pred_node_xy = node_xy[node_ids.index(pred_node)]
            pos_errors.append(float(np.linalg.norm(pred_node_xy - gps_xy[i])))

    acc      = 100.0 * correct / M if M > 0 else 0.0
    mean_err = float(np.mean(pos_errors))   if pos_errors else float('nan')
    med_err  = float(np.median(pos_errors)) if pos_errors else float('nan')
    print(f"  Nearest-node accuracy  : {acc:.1f}%  ({correct}/{M})")
    print(f"  Mean  position error   : {mean_err:.1f} m")
    print(f"  Median position error  : {med_err:.1f} m")


def run_train(
    flac_dir:   Path,
    nodes_txt:  Path,
    gps_csv:    Path,
    step_s:     float,
    session:    str | None,
    out_params: Path,
) -> None:
    """Orchestrate path-loss training: FLAC + GPS → fitted JSON."""
    print("=" * 60)
    print("  BAYESIAN PATH-LOSS TRAINING")
    print("=" * 60)

    # 1. FLAC discovery
    suffix = f" (session={session})" if session else ""
    print(f"\n[1/6] Scanning {flac_dir} for FLAC files{suffix} …")
    file_map = discover_flac_files(flac_dir, session=session)
    if not file_map:
        print("[ERROR] No FLAC files found.")
        sys.exit(1)
    node_ids = sorted(file_map.keys())
    print(f"       Found {len(node_ids)} nodes: {node_ids}")
    for nid, fp in file_map.items():
        print(f"         node {nid:3d} → {fp.name}")

    # 2. Node positions (row-ordered)
    print(f"\n[2/6] Loading node positions from {nodes_txt.name} …")
    node_xy = load_node_positions_ordered(nodes_txt, node_ids)
    for i, nid in enumerate(node_ids):
        print(f"         node {nid:3d}: x={node_xy[i,0]:.1f}  y={node_xy[i,1]:.1f}")

    # 3. GPS track
    print(f"\n[3/6] Loading GPS track from {gps_csv.name} …")
    gps_df = load_gps_track(gps_csv)
    print(f"       {len(gps_df)} samples  "
          f"({gps_df['timestamp'].iloc[0]} → {gps_df['timestamp'].iloc[-1]})")

    # 4. FLAC start time
    flac_start = _parse_session_start(session) if session else gps_df['timestamp'].iloc[0]
    print(f"\n[4/6] FLAC t=0 anchored to {flac_start.isoformat()}")

    # 5. Audio power
    print(f"\n[5/6] Computing audio power (step={step_s}s) …")
    power_arrays: dict[int, np.ndarray] = {}
    for nid, fpath in file_map.items():
        print(f"       node {nid}: {fpath.name}")
        pwr = compute_power_per_step(fpath, step_s)
        power_arrays[nid] = pwr
        print(f"         → {len(pwr)} steps")

    # 6. Align GPS ↔ FLAC
    print(f"\n[6/6] Building training pairs …")
    distances, rssi_db, gps_xy = build_training_pairs(
        gps_df, node_xy, power_arrays, node_ids, flac_start, step_s
    )
    print(f"       {len(distances)} aligned (GPS, RSSI) pairs")
    if len(distances) == 0:
        print("[ERROR] No training pairs — check GPS timestamps vs FLAC session.")
        sys.exit(1)

    # 7. Fit
    print(f"\n{'='*60}")
    print("  FITTING PATH-LOSS PARAMETERS")
    print(f"{'='*60}")
    params = fit_path_loss(distances, rssi_db, node_ids)

    # 8. Evaluate
    print(f"\n{'='*60}")
    print("  EVALUATION (training set)")
    print(f"{'='*60}")
    evaluate_fit(params, node_ids, node_xy, gps_xy, distances, rssi_db)

    # 9. Save
    out_params.parent.mkdir(parents=True, exist_ok=True)
    with open(out_params, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n  Saved → {out_params}")


# ---------------------------------------------------------------------------
# Audio power extraction
# ---------------------------------------------------------------------------

def compute_power_per_step(flac_path: Path, step_s: float) -> np.ndarray:
    """
    Load a FLAC file and compute mean-squared power for each non-overlapping
    window of `step_s` seconds, matching respeaker_custom.send_power().

    Steps per the original code:
      1. Read channels 1:5, average across mic axis.
      2. DC-remove (subtract window mean).
      3. mean(x²) -> linear power.

    Returns float64 array of shape (T,) where T = floor(total_samples / win_samples).
    """
    data, sr = sf.read(str(flac_path), dtype='int16', always_2d=True)
    # data: (total_samples, channels)
    if sr != RESPEAKER_RATE:
        print(f"    [WARN] {flac_path.name}: sample rate {sr} != {RESPEAKER_RATE}; "
              f"timing may be off.")

    win_samples = int(round(step_s * sr))
    if win_samples <= 0:
        raise ValueError(f"step_s={step_s} too small for sample rate {sr}")

    n_steps = len(data) // win_samples
    powers  = np.zeros(n_steps, dtype=np.float64)

    for t in range(n_steps):
        chunk = data[t * win_samples : (t + 1) * win_samples]   # (win, 6)
        # Average across the 4 respeaker body mics (channels 1–4)
        mic_avg = chunk[:, POWER_CHANNELS].mean(axis=1).astype(np.float32)
        mic_avg -= mic_avg.mean()                                 # DC removal
        powers[t] = float(np.mean(mic_avg ** 2))                 # mean squared

    return powers


# ---------------------------------------------------------------------------
# Hypothesis grid + posterior
# ---------------------------------------------------------------------------

def build_hypothesis_grid(node_xy: np.ndarray):
    """Build the 20×20 candidate grid centred on the node cloud centroid."""
    cx, cy = node_xy.mean(axis=0)
    gx = cx + CANDIDATE_OFFSETS * SPACING
    gy = cy + CANDIDATE_OFFSETS * SPACING
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    H, W    = GX.shape
    hypos   = np.column_stack([GX.ravel(), GY.ravel()])   # (H*W, 2)

    # Precompute distances from each hypothesis to each node
    dx   = hypos[:, [0]] - node_xy[:, 0].reshape(1, -1)
    dy   = hypos[:, [1]] - node_xy[:, 1].reshape(1, -1)
    dist = np.clip(np.sqrt(dx**2 + dy**2), 1e-3, None)

    # Pre-sort nearest nodes per hypothesis
    nearest_idx = np.argsort(dist, axis=1)   # (H*W, N)

    return hypos, dist, nearest_idx, H, W


def compute_posterior_top1(rss_map: dict, node_ids: list[int],
                           dist: np.ndarray, nearest_idx: np.ndarray,
                           P0, ETA, SIG2, MASK,
                           hypos: np.ndarray) -> int | None:
    """
    Run the Bayesian path-loss posterior and return the 1-indexed node ID
    of the node nearest the highest-probability hypothesis.

    Mirrors RealTimeGUI.compute_posterior() exactly.

    Returns None if no usable readings are available.
    """
    have = np.array([nid in rss_map and np.isfinite(rss_map[nid]) for nid in node_ids])
    use  = MASK & have
    if not np.any(use):
        return None

    cols     = np.where(use)[0]
    rss_vec  = np.array([rss_map[nid] for nid in np.array(node_ids)[cols]])  # (M,)

    d_use    = dist[:, cols]                                    # (H*W, M)
    p0_use   = P0[cols][None, :]
    eta_use  = ETA[cols][None, :]
    sig2_use = SIG2[cols][None, :]

    pred     = p0_use - 10.0 * eta_use * np.log10(d_use)      # (H*W, M)
    resid    = rss_vec[None, :] - pred

    ll       = -0.5 * np.log(2 * np.pi * sig2_use) \
               - 0.5 * (resid ** 2) / sig2_use
    log_post = np.sum(ll, axis=1)                              # (H*W,)

    # Stable softmax
    log_post -= log_post.max()
    post      = np.exp(log_post)
    post     /= post.sum()

    # Top-1 hypothesis
    top1_idx   = int(np.argmax(post))
    # Nearest node to top-1 hypothesis (1 node = the #1 recommendation)
    nearest_col = nearest_idx[top1_idx, 0]
    return node_ids[nearest_col]


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process(
    flac_dir:    Path,
    nodes_csv:   Path,
    pl_json:     Path,
    step_s:      float,
    rssi_window: int,
    out_dir:     Path,
    session:     str | None = None,
) -> np.ndarray:
    """
    Full pipeline: FLAC → power → dB → rolling avg → posterior → rankings.

    Returns int32 (T,) array of 1-indexed node IDs (top-1 each timestep).
    """
    # ── Discover files ───────────────────────────────────────────────────
    suffix = f" (session={session})" if session else ""
    print(f"\n[1/5] Scanning {flac_dir} for FLAC files{suffix} …")
    file_map = discover_flac_files(flac_dir, session=session)
    if not file_map:
        print("[ERROR] No FLAC files with recognisable node IDs found.")
        sys.exit(1)
    node_ids = sorted(file_map.keys())
    print(f"       Found {len(node_ids)} nodes: {node_ids}")
    for nid, fp in file_map.items():
        print(f"         node {nid:3d} → {fp.name}")

    # ── Load node positions ───────────────────────────────────────────────
    print(f"\n[2/5] Loading node positions from {nodes_csv.name} …")
    node_xy = load_node_positions(nodes_csv, node_ids)
    for i, nid in enumerate(node_ids):
        print(f"         node {nid:3d}: x={node_xy[i,0]:.1f}  y={node_xy[i,1]:.1f}")

    # ── Load path-loss params ─────────────────────────────────────────────
    print(f"\n[3/5] Loading path-loss params from {pl_json.name} …")
    P0, ETA, SIG2, MASK = load_path_loss(pl_json, node_ids)
    for i, nid in enumerate(node_ids):
        print(f"         node {nid:3d}: P0={P0[i]:.1f}  η={ETA[i]:.2f}  "
              f"σ²={SIG2[i]:.4f}  active={MASK[i]}")

    # ── Build hypothesis grid ─────────────────────────────────────────────
    print(f"\n[4/5] Building hypothesis grid (20×20, {SPACING}m spacing) …")
    hypos, dist, nearest_idx, H, W = build_hypothesis_grid(node_xy)
    print(f"       Grid: {H}×{W} = {H*W} hypotheses")

    # ── Compute power per timestep for each node ──────────────────────────
    print(f"\n[5/5] Computing power (step={step_s}s, rssi_window={rssi_window}) …")

    # First pass: find the minimum number of timesteps across all files
    power_arrays: dict[int, np.ndarray] = {}
    for nid, fpath in file_map.items():
        print(f"       Processing node {nid}: {fpath.name}")
        pwr = compute_power_per_step(fpath, step_s)
        power_arrays[nid] = pwr
        print(f"         → {len(pwr)} timesteps")

    T = min(len(a) for a in power_arrays.values())
    print(f"\n       Using {T} timesteps (shortest file).")

    # ── Rolling dB windows (one deque per node, as in the GUI) ───────────
    rssi_windows = {nid: deque(maxlen=rssi_window) for nid in node_ids}

    rankings = np.zeros(T, dtype=np.int32)

    for t in range(T):
        # Update rolling dB for every node at this timestep
        rss_map: dict[int, float] = {}
        for nid in node_ids:
            pwr = power_arrays[nid][t]
            db  = 10.0 * np.log10(max(float(pwr), 1e-12))
            rssi_windows[nid].append(db)
            rss_map[nid] = float(np.mean(rssi_windows[nid]))

        # Posterior → top-1 node
        top1 = compute_posterior_top1(
            rss_map, node_ids, dist, nearest_idx,
            P0, ETA, SIG2, MASK, hypos
        )
        rankings[t] = top1 if top1 is not None else 0

        if t % 50 == 0 or t == T - 1:
            print(f"       t={t:5d}/{T}  rss={[f'{rss_map[n]:.1f}' for n in node_ids]}  "
                  f"top1={top1}")

    return rankings, node_ids, T, node_xy


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(rankings: np.ndarray, node_ids: list, T: int,
                 step_s: float, pl_json: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")

    arr_path  = out_dir / f"rankings_{session}.npy"
    meta_path = out_dir / f"rankings_{session}_meta.json"

    np.save(arr_path, rankings)

    # Human-readable timestamp per step (relative to t=0)
    ts_strings = [f"t={i*step_s:.3f}s" for i in range(T)]

    meta = {
        "schema_version":   1,
        "created_at_utc":   datetime.now(timezone.utc).isoformat(),
        "step_s":           step_s,
        "total_steps":      T,
        "node_ids":         node_ids,
        "path_loss_file":   str(pl_json),
        "array_shape":      list(rankings.shape),
        "array_dtype":      str(rankings.dtype),
        "description":      "rankings[t] = 1-indexed node ID of top-1 audio source at timestep t. "
                            "0 = no valid posterior (insufficient RSSI readings).",
        "ts_strings":       ts_strings,
        "unique_nodes_seen": sorted(int(n) for n in set(rankings.tolist()) if n > 0),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  rankings .npy  → {arr_path}")
    print(f"  metadata .json → {meta_path}")
    print(f"\n  Shape          : {rankings.shape}  dtype={rankings.dtype}")
    print(f"  Unique top-1   : {meta['unique_nodes_seen']}")
    print(f"  Zero steps     : {int((rankings == 0).sum())} "
          f"(posterior unavailable — not enough window fill)")
    return session


# ---------------------------------------------------------------------------
# Transition matrix (ported / adapted from respeaker_transition.py)
# ---------------------------------------------------------------------------

def _tm_apply_min_dwell(positions: np.ndarray, min_dwell: int) -> np.ndarray:
    if min_dwell <= 1:
        return positions
    filtered = positions.copy()
    i = 0
    while i < len(filtered):
        cell = filtered[i]
        run = 1
        while i + run < len(filtered) and filtered[i + run] == cell:
            run += 1
        if run < min_dwell and i > 0:
            filtered[i:i + run] = filtered[i - 1]
        i += run
    n_changed = int((filtered != positions).sum())
    print(f"  Min-dwell filter (dwell={min_dwell}): "
          f"replaced {n_changed} timesteps ({n_changed / len(positions) * 100:.1f}%)")
    return filtered


def _tm_build_matrix(positions: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    T_count = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(positions[:-1], positions[1:]):
        T_count[a, b] += 1
    T_prob = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = T_count[i].astype(float)
        total = row.sum()
        if total > 0:
            T_prob[i] = row / total
    return T_prob, T_count


def _tm_greedy_cycle(T_prob: np.ndarray, n: int, min_prob: float) -> list[int]:
    # Normalise off-diagonal only so self-loop probabilities don't swamp the
    # cycle extraction (the stored matrix retains self-loops).
    T_off = T_prob.copy()
    np.fill_diagonal(T_off, 0.0)
    row_sums = T_off.sum(axis=1, keepdims=True)
    T_norm = np.where(row_sums > 0, T_off / row_sums, T_off)

    start = int(T_norm.sum(axis=1).argmax())
    cycle, visited, current = [start], {start}, start
    for _ in range(n - 1):
        row = T_norm[current].copy()
        row[list(visited)] = 0.0
        if row.max() < min_prob:
            break
        nxt = int(row.argmax())
        cycle.append(nxt)
        visited.add(nxt)
        current = nxt
    return cycle


def _tm_coverage(cycle: list[int], T_count: np.ndarray) -> float:
    pairs = set(zip(cycle, cycle[1:] + [cycle[0]]))
    total = int(T_count.sum())
    if total == 0:
        return 0.0
    return sum(int(T_count[i, j]) for i, j in pairs) / total


def _tm_plot(T_prob, T_count, positions, cycle_idx, node_ids, node_xy, out_dir, session):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plot.")
        return

    n = len(node_ids)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Transition Analysis", fontsize=13, fontweight="bold")
    labels = [str(nid) for nid in node_ids]

    # Panel 1: probability heatmap
    ax = axes[0]
    im = ax.imshow(T_prob, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Transition Probability P[i→j]\n(self-loops included)")
    ax.set_xlabel("To node j"); ax.set_ylabel("From node i")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{T_prob[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if T_prob[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 2: raw counts
    ax = axes[1]
    im2 = ax.imshow(T_count, cmap="Oranges")
    ax.set_title("Raw Transition Counts")
    ax.set_xlabel("To node j"); ax.set_ylabel("From node i")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(T_count[i, j])), ha="center", va="center", fontsize=7)
    plt.colorbar(im2, ax=ax, fraction=0.046)

    # Panel 3: node scatter map with cycle arrows
    ax = axes[2]
    cycle_nids = [node_ids[i] for i in cycle_idx]
    ax.set_title(f"Dominant Cycle: {cycle_nids}")
    counts = np.bincount(positions, minlength=n)
    sc = ax.scatter(node_xy[:, 0], node_xy[:, 1],
                    c=counts / max(counts.max(), 1), cmap="Greens", s=400,
                    edgecolors="black", linewidths=1.5, zorder=3, vmin=0.1, vmax=1.0)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="Relative occupancy")
    for i, nid in enumerate(node_ids):
        ax.text(node_xy[i, 0], node_xy[i, 1], str(nid),
                ha="center", va="center", fontsize=9, zorder=4,
                fontweight="bold" if i in cycle_idx else "normal", color="white")
    for k in range(len(cycle_idx)):
        a, b = cycle_idx[k], cycle_idx[(k + 1) % len(cycle_idx)]
        ax.annotate("", xy=node_xy[b], xytext=node_xy[a],
                    arrowprops=dict(arrowstyle="->", color="red", lw=2), zorder=5)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(useOffset=False)

    plt.tight_layout()
    plot_path = out_dir / f"transition_analysis_{session}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")


def build_and_save_transition(
    rankings:   np.ndarray,
    node_ids:   list[int],
    node_xy:    np.ndarray,
    min_dwell:  int,
    min_prob:   float,
    no_plot:    bool,
    out_dir:    Path,
    session:    str,
) -> None:
    print("\n" + "=" * 60)
    print("  TRANSITION MATRIX")
    print("=" * 60)

    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    valid = rankings[rankings > 0]
    if len(valid) == 0:
        print("  [WARN] No valid rankings — cannot build transition matrix.")
        return
    positions = np.array([id_to_idx[r] for r in valid], dtype=np.int32)

    print(f"\n[1/3] Applying min-dwell filter …")
    positions = _tm_apply_min_dwell(positions, min_dwell)

    print(f"\n[2/3] Building transition matrix …")
    T_prob, T_count = _tm_build_matrix(positions, n)

    print(f"\n[3/3] Extracting dominant cycle …")
    cycle_idx = _tm_greedy_cycle(T_prob, n, min_prob)
    coverage  = _tm_coverage(cycle_idx, T_count)
    cycle_nids = [node_ids[i] for i in cycle_idx]
    print(f"       Cycle (node IDs) : {cycle_nids}")
    print(f"       Coverage         : {coverage * 100:.1f}% of observed transitions")

    out_dir.mkdir(parents=True, exist_ok=True)
    tm_path  = out_dir / f"transition_matrix_{session}.npy"
    raw_path = out_dir / f"transition_matrix_raw_{session}.npy"
    cyc_path = out_dir / f"dominant_cycle_{session}.json"
    rpt_path = out_dir / f"transition_report_{session}.txt"

    np.save(tm_path,  T_prob)
    np.save(raw_path, T_count)

    counts = np.bincount(positions, minlength=n)
    with open(cyc_path, "w") as f:
        json.dump({
            "cycle":       cycle_nids,
            "cycle_str":   ",".join(str(c) for c in cycle_nids),
            "cycle_idx":   cycle_idx,
            "coverage":    round(float(coverage), 4),
            "min_dwell":   min_dwell,
            "min_prob":    min_prob,
            "n_timesteps": int(len(positions)),
            "node_ids":    node_ids,
            "node_counts": counts.tolist(),
        }, f, indent=2)

    lines = [
        "=" * 60,
        "  TRANSITION MATRIX REPORT",
        "=" * 60,
        f"  Timesteps analysed : {len(positions)}",
        f"  Min-dwell filter   : {min_dwell} steps",
        f"  Min-prob threshold : {min_prob}",
        "",
        "  Node occupancy:",
    ]
    for i, nid in enumerate(node_ids):
        bar = "█" * int(counts[i] / max(counts.max(), 1) * 20)
        lines.append(f"    node {nid}: {counts[i]:6d} steps  {bar}")
    lines += [
        "",
        "  Transition probability matrix P[i→j]  (self-loops included):",
        "         " + "  ".join(f"n={nid:3d}" for nid in node_ids),
    ]
    for i, nid_i in enumerate(node_ids):
        row_str = "  ".join(f"{T_prob[i, j]:6.3f}" for j in range(n))
        lines.append(f"  n={nid_i:3d}  {row_str}")
    circle_arg = ",".join(str(c) for c in cycle_nids)
    lines += [
        "",
        f"  Dominant cycle (node IDs) : {cycle_nids}",
        f"  Cycle coverage            : {coverage * 100:.1f}%",
        "",
        "  Next step — finetune_deterministic.py:",
        f"    python examples/finetune_deterministic.py \\",
        f"        --run 201 --circle {circle_arg}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    with open(rpt_path, "w") as f:
        f.write(report)
    print("\n" + report)

    print(f"\n  transition_matrix .npy     → {tm_path}")
    print(f"  transition_matrix_raw .npy → {raw_path}")
    print(f"  dominant_cycle .json       → {cyc_path}")
    print(f"  transition_report .txt     → {rpt_path}")

    if not no_plot:
        _tm_plot(T_prob, T_count, positions, cycle_idx, node_ids, node_xy, out_dir, session)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert per-node ReSpeaker FLAC files to a Track-MDP node ranking array, "
                    "or train Bayesian path-loss parameters from GPS ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Inference mode (default)
------------------------
  python flac_to_trackmdp.py \\
      --flac-dir ./data/audio \\
      --nodes-csv ./data/nodes_gps.csv \\
      --path-loss ./data/TRUCK_path_loss_params.json

  python flac_to_trackmdp.py \\
      --flac-dir ./data/audio \\
      --nodes-csv ./data/nodes_gps.csv \\
      --path-loss ./data/ATV_path_loss_params.json \\
      --step 0.2 --rssi-window 3 --out-dir ./results

Training mode (--train)
-----------------------
  python flac_to_trackmdp.py --train \\
      --flac-dir  iobt_data/ \\
      --nodes-txt iobt_data/node_positions.txt \\
      --gps-csv   iobt_data/20260416_154037_gps2_gps.csv \\
      --session   20260416_154037 \\
      --step      0.5 \\
      --out-params iobt_data/path_loss_params_20260416.json
        """
    )

    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--flac-dir", required=True, type=Path,
                        help="Folder containing per-node *.flac files")
    parser.add_argument("--step",     type=float, default=DEFAULT_STEP_S,
                        help=f"Timestep window in seconds (default: {DEFAULT_STEP_S})")

    # ── Training mode ────────────────────────────────────────────────────────
    parser.add_argument("--train",      action="store_true",
                        help="Fit path-loss parameters from GPS ground truth")
    parser.add_argument("--nodes-txt",  type=Path,
                        help="[--train] Tab-delimited node_positions.txt (Latitude, Longitude)")
    parser.add_argument("--gps-csv",    type=Path,
                        help="[--train] GPS ground truth CSV")
    parser.add_argument("--session",    type=str,
                        help="[--train] Session prefix to filter FLAC files "
                             "(e.g. 20260416_154037)")
    parser.add_argument("--out-params", type=Path,
                        default=Path("path_loss_params.json"),
                        help="[--train] Output JSON for fitted params "
                             "(default: path_loss_params.json)")

    # ── Inference mode ───────────────────────────────────────────────────────
    parser.add_argument("--nodes-csv",   type=Path,
                        help="[inference] Path to nodes_gps.csv (Lat, Lon, Node # columns)")
    parser.add_argument("--path-loss",   type=Path,
                        help="[inference] Path-loss params JSON")
    parser.add_argument("--rssi-window", type=int, default=DEFAULT_RSSI_WINDOW,
                        help=f"[inference] Rolling average window (default: {DEFAULT_RSSI_WINDOW})")
    parser.add_argument("--out-dir",     type=Path, default=Path("./results"),
                        help="[inference] Output directory (default: ./results)")
    parser.add_argument("--infer-session", type=str, default=None,
                        help="[inference] Only use FLAC files whose name starts with this "
                             "prefix (e.g. 20260416_154037). Prevents mixing sessions when "
                             "multiple recordings per node exist in --flac-dir.")

    # ── Transition matrix ────────────────────────────────────────────────────
    parser.add_argument("--transition",  action="store_true",
                        help="Also build a transition matrix from the rankings output")
    parser.add_argument("--min-dwell",   type=int,   default=2,
                        help="[--transition] Min consecutive steps to count as a real visit "
                             "(default: 2; set to 1 to disable)")
    parser.add_argument("--min-prob",    type=float, default=0.05,
                        help="[--transition] Min transition probability for cycle extraction "
                             "(default: 0.05)")
    parser.add_argument("--no-plot",     action="store_true",
                        help="[--transition] Skip matplotlib visualisation")

    args = parser.parse_args()

    if not args.flac_dir.exists():
        print(f"[ERROR] --flac-dir not found: {args.flac_dir}")
        sys.exit(1)

    # ── Branch: train or infer ───────────────────────────────────────────────
    if args.train:
        missing = [n for n, v in [("--nodes-txt", args.nodes_txt),
                                   ("--gps-csv",   args.gps_csv)] if not v]
        if missing:
            print(f"[ERROR] --train requires: {', '.join(missing)}")
            sys.exit(1)
        for p, name in [(args.nodes_txt, "--nodes-txt"), (args.gps_csv, "--gps-csv")]:
            if not p.exists():
                print(f"[ERROR] {name} not found: {p}")
                sys.exit(1)
        run_train(
            flac_dir   = args.flac_dir,
            nodes_txt  = args.nodes_txt,
            gps_csv    = args.gps_csv,
            step_s     = args.step,
            session    = args.session,
            out_params = args.out_params,
        )
    else:
        missing = [n for n, v in [("--nodes-csv", args.nodes_csv),
                                   ("--path-loss", args.path_loss)] if not v]
        if missing:
            print(f"[ERROR] inference mode requires: {', '.join(missing)}")
            sys.exit(1)
        for p, name in [(args.nodes_csv, "--nodes-csv"), (args.path_loss, "--path-loss")]:
            if not p.exists():
                print(f"[ERROR] {name} not found: {p}")
                sys.exit(1)

        print("=" * 60)
        print("  FLAC → TRACK-MDP RANKING")
        print("=" * 60)
        print(f"  FLAC dir     : {args.flac_dir}")
        print(f"  Nodes CSV    : {args.nodes_csv}")
        print(f"  Path-loss    : {args.path_loss}")
        print(f"  Step size    : {args.step} s")
        print(f"  RSSI window  : {args.rssi_window} steps")
        print(f"  Output dir   : {args.out_dir}")

        rankings, node_ids, T, node_xy = process(
            flac_dir    = args.flac_dir,
            nodes_csv   = args.nodes_csv,
            pl_json     = args.path_loss,
            step_s      = args.step,
            rssi_window = args.rssi_window,
            out_dir     = args.out_dir,
            session     = args.infer_session,
        )

        print("\n" + "=" * 60)
        print("  SAVING OUTPUTS")
        print("=" * 60)
        session = save_outputs(rankings, node_ids, T, args.step, args.path_loss, args.out_dir)

        if args.transition:
            build_and_save_transition(
                rankings  = rankings,
                node_ids  = node_ids,
                node_xy   = node_xy,
                min_dwell = args.min_dwell,
                min_prob  = args.min_prob,
                no_plot   = args.no_plot,
                out_dir   = args.out_dir,
                session   = session,
            )

        print("\n" + "=" * 60)
        print("  DONE")
        print("=" * 60)


if __name__ == "__main__":
    main()