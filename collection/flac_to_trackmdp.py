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

def discover_flac_files(flac_dir: Path) -> dict[int, Path]:
    """
    Scan flac_dir for *.flac files and map each to a node ID.

    Node ID is parsed from the filename by looking for:
      - "node<N>"          e.g. node3_respeaker_20250812.flac  -> 3
      - "orin_<N>"         e.g. dvpg_gq_orin_11_respeaker.flac -> 11
    If multiple files map to the same node, the lexicographically last
    (most recent) is kept.
    """
    pattern = re.compile(r'(?:node|orin_)(\d+)', re.IGNORECASE)
    found: dict[int, Path] = {}
    for fpath in sorted(flac_dir.glob("*.flac")):
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
) -> np.ndarray:
    """
    Full pipeline: FLAC → power → dB → rolling avg → posterior → rankings.

    Returns int32 (T,) array of 1-indexed node IDs (top-1 each timestep).
    """
    # ── Discover files ───────────────────────────────────────────────────
    print(f"\n[1/5] Scanning {flac_dir} for FLAC files …")
    file_map = discover_flac_files(flac_dir)
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

    return rankings, node_ids, T


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert per-node ReSpeaker FLAC files to a Track-MDP node ranking array.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output
------
  rankings_SESSION.npy       int32 (T,) — top-1 node ID at each timestep (1-indexed, 0=unknown)
  rankings_SESSION_meta.json — step_s, node list, timestamps, etc.

Loading the output
------------------
  import numpy as np, json
  r    = np.load("results/rankings_20250812_165716.npy")
  meta = json.load(open("results/rankings_20250812_165716_meta.json"))
  # r[t] = node ID of dominant audio source at timestep t

Examples
--------
  python flac_to_trackmdp.py \\
      --flac-dir ./data/audio \\
      --nodes-csv ./data/nodes_gps.csv \\
      --path-loss ./data/TRUCK_path_loss_params.json

  python flac_to_trackmdp.py \\
      --flac-dir ./data/audio \\
      --nodes-csv ./data/nodes_gps.csv \\
      --path-loss ./data/ATV_path_loss_params.json \\
      --step 0.2 --rssi-window 3 --out-dir ./results
        """
    )
    parser.add_argument("--flac-dir",    required=True,  type=Path,
                        help="Folder containing per-node *.flac files")
    parser.add_argument("--nodes-csv",   required=True,  type=Path,
                        help="Path to nodes_gps.csv (Lat, Lon, Node # columns)")
    parser.add_argument("--path-loss",   required=True,  type=Path,
                        help="Path-loss params JSON (vehicle-specific)")
    parser.add_argument("--step",        type=float, default=DEFAULT_STEP_S,
                        help=f"Timestep window in seconds (default: {DEFAULT_STEP_S})")
    parser.add_argument("--rssi-window", type=int,   default=DEFAULT_RSSI_WINDOW,
                        help=f"Rolling average window (default: {DEFAULT_RSSI_WINDOW})")
    parser.add_argument("--out-dir",     type=Path,  default=Path("./results"),
                        help="Output directory (default: ./results)")
    args = parser.parse_args()

    # Validate inputs
    for p, name in [(args.flac_dir,  "--flac-dir"),
                    (args.nodes_csv, "--nodes-csv"),
                    (args.path_loss, "--path-loss")]:
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

    rankings, node_ids, T = process(
        flac_dir    = args.flac_dir,
        nodes_csv   = args.nodes_csv,
        pl_json     = args.path_loss,
        step_s      = args.step,
        rssi_window = args.rssi_window,
        out_dir     = args.out_dir,
    )

    print("\n" + "=" * 60)
    print("  SAVING OUTPUTS")
    print("=" * 60)
    save_outputs(rankings, node_ids, T, args.step, args.path_loss, args.out_dir)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()