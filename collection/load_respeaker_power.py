#!/usr/bin/env python3
"""
load_respeaker_power.py — Load and print a respeaker_power numpy array
saved by mqtt_collector.py.

Usage
-----
    # Load the most recent file in ./data
    python load_respeaker_power.py

    # Load a specific file
    python load_respeaker_power.py data/respeaker_power_20260416_120000.npy

    # Load from a different directory
    python load_respeaker_power.py --data-dir /path/to/data
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

KNOWN_NODES = list(range(11, 17))   # must match mqtt_collector.py
NODE_NAMES  = [f"dvpg_gq_orin_{n}" for n in KNOWN_NODES]


def find_latest(data_dir: Path) -> Path | None:
    files = sorted(data_dir.glob("respeaker_power_[0-9]*.npy"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    # Exclude timestamp files (*_ts_*.npy)
    files = [f for f in files if "_ts_" not in f.name]
    return files[0] if files else None


def load_session(arr_path: Path):
    arr_path = arr_path.resolve()
    data_dir = arr_path.parent
    stem     = arr_path.stem                          # respeaker_power_YYYYMMDD_HHMMSS

    ts_path   = data_dir / f"respeaker_power_ts_{stem.removeprefix('respeaker_power_')}.npy"
    meta_path = data_dir / f"{stem}_meta.json"

    array = np.load(arr_path)

    timestamps = np.load(ts_path) if ts_path.exists() else None
    meta       = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return array, timestamps, meta


def print_report(array: np.ndarray, timestamps, meta: dict, arr_path: Path) -> None:
    steps, n_nodes = array.shape
    node_ids   = meta.get("node_ids",   KNOWN_NODES)
    node_names = meta.get("node_names", NODE_NAMES)

    print("=" * 60)
    print("  RESPEAKER POWER — LOADED ARRAY")
    print("=" * 60)
    print(f"  File       : {arr_path}")
    print(f"  Shape      : {array.shape}  (steps × nodes)")
    print(f"  dtype      : {array.dtype}")
    if meta.get("broker"):
        print(f"  Broker     : {meta['broker']}")
    if meta.get("created_at_utc"):
        print(f"  Recorded   : {meta['created_at_utc']}")
    print()

    # Column header
    col_w  = 12
    header = f"{'step':>6}  {'timestamp':<22}" + "".join(
        f"  {f'orin_{n}':>{col_w}}" for n in node_ids
    )
    print(header)
    print("-" * len(header))

    # Print every row (cap at 200 rows to avoid flooding the terminal)
    max_rows = 200
    show     = min(steps, max_rows)
    for t in range(show):
        if timestamps is not None:
            ts_str = datetime.fromtimestamp(float(timestamps[t])).strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = "n/a"
        vals = "".join(f"  {array[t, i]:>{col_w}.2f}" for i in range(n_nodes))
        print(f"  {t+1:>4}  {ts_str:<22}{vals}")

    if steps > max_rows:
        print(f"  ... ({steps - max_rows} more rows not shown)")

    # Summary statistics
    print()
    print("  Summary statistics (per node)")
    print("-" * len(header))
    stat_header = f"{'stat':>6}  {'':22}" + "".join(
        f"  {f'orin_{n}':>{col_w}}" for n in node_ids
    )
    print(stat_header)
    for label, vals in [
        ("min",    array.min(axis=0)),
        ("max",    array.max(axis=0)),
        ("mean",   array.mean(axis=0)),
        ("std",    array.std(axis=0)),
        ("last",   array[-1]),
    ]:
        row = "".join(f"  {v:>{col_w}.2f}" for v in vals)
        print(f"  {label:>4}  {'':22}{row}")

    print()
    print("  Full numpy array (array variable):")
    print(array)


def main():
    parser = argparse.ArgumentParser(
        description="Load and print a respeaker_power .npy file from mqtt_collector.py.",
    )
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to respeaker_power_*.npy file (default: latest in --data-dir)")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory to search for the latest file (default: ./data)")
    args = parser.parse_args()

    if args.file:
        arr_path = Path(args.file)
        if not arr_path.exists():
            print(f"[ERROR] File not found: {arr_path}")
            raise SystemExit(1)
    else:
        data_dir = Path(args.data_dir)
        arr_path = find_latest(data_dir)
        if arr_path is None:
            print(f"[ERROR] No respeaker_power_*.npy files found in '{data_dir}'")
            raise SystemExit(1)
        print(f"[INFO] Loading latest file: {arr_path}\n")

    array, timestamps, meta = load_session(arr_path)
    print_report(array, timestamps, meta, arr_path)


if __name__ == "__main__":
    main()
