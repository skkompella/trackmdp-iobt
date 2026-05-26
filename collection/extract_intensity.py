#!/usr/bin/env python3
"""
extract_intensity.py — Extract per-node acoustic intensity from FLAC files.

Reads the 20260416_154037 session FLAC files for nodes 11-16, computes
mean-squared power per 0.5s window using compute_power_per_step, converts
to dB (matching flac_to_trackmdp.process()), and saves a (T, 6) float32
array to data/20260416_154037_intensity.npy.

Columns: [node_11, node_12, node_13, node_14, node_15, node_16]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from flac_to_trackmdp import compute_power_per_step, DEFAULT_STEP_S

REPO_ROOT = Path(__file__).parent.parent
IOBT_DATA = REPO_ROOT / "iobt_data"
OUT_DIR   = REPO_ROOT / "data"
SESSION   = "20260416_154037"
NODE_IDS  = [11, 12, 13, 14, 15, 16]
STEP_S    = DEFAULT_STEP_S  # 0.5 s

flac_files = {
    nid: IOBT_DATA / f"{SESSION}_dvpg_gq_orin_{nid}_respeaker.flac"
    for nid in NODE_IDS
}

for nid, fpath in flac_files.items():
    if not fpath.exists():
        print(f"[ERROR] Missing: {fpath}")
        sys.exit(1)

power_arrays = {}
for nid, fpath in flac_files.items():
    print(f"  node {nid}: {fpath.name}")
    pwr = compute_power_per_step(fpath, STEP_S)
    # Convert to dB (matches flac_to_trackmdp.process() and the live GUI).
    db = 10.0 * np.log10(np.maximum(pwr, 1e-12))
    power_arrays[nid] = db
    print(f"    → {len(db)} steps  (dB range: {db.min():.1f} … {db.max():.1f})")

T = min(len(a) for a in power_arrays.values())
print(f"\nTruncating to {T} steps (shortest file).")

intensity = np.stack(
    [power_arrays[nid][:T] for nid in NODE_IDS], axis=1
).astype(np.float64)   # (T, 6)  dB, float64

# Per-node z-score: each column gets zero mean and unit variance.
# Without this, fixed hardware gain differences (node 12 is ~8 dB louder on
# average) make argmax permanently lock onto one node and starve others.
intensity = (intensity - intensity.mean(axis=0)) / intensity.std(axis=0)
intensity = intensity.astype(np.float32)

out_path = OUT_DIR / f"{SESSION}_intensity.npy"
np.save(out_path, intensity)
print(f"\nSaved {intensity.shape} {intensity.dtype} → {out_path}")
