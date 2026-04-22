#!/usr/bin/env python3
"""
iobt_step_builder.py — Convert per-node YOLO .jsonl files into aligned step
arrays saved as numpy .npy files.

Input
-----
Per-node .jsonl files produced by iobt_collector.py, e.g.:
    data/node1_20250812_165716.jsonl
    data/node10_20250812_165716.jsonl

Each line is a collector record:
    {"rx_ts":..., "node":"node10", "topic":...,
     "detections":[{"class":"car","conf":0.91,"t":"2025-08-12 16:57:16.428494",...}]}

Bucketing rule
--------------
Every detection is floored to the nearest step_s boundary:
    bucket = floor(capture_time / step_s) * step_s  (default step_s = 0.5 s)

Output — three files per session
---------------------------------
  detections_YYYYMMDD_HHMMSS.npy
      uint8 array, shape (T, 10)
      arr[t, i] = 1 if node_{i+1} had a qualifying detection in timestep t
      node1 -> col 0, node2 -> col 1, ..., node10 -> col 9
      "qualifying" = max_conf in bucket >= conf_threshold AND class matches filter

  timestamps_YYYYMMDD_HHMMSS.npy
      float64 array, shape (T,)
      Unix timestamp of each bucket floor (seconds)
      timestamps[t] corresponds to row t of the detection array

  metadata_YYYYMMDD_HHMMSS.json
      step_s, conf_threshold, class_filter, node ordering,
      human-readable timestamp strings, raw stats

Usage
-----
    python iobt_step_builder.py                          # all sessions in ./data/
    python iobt_step_builder.py --broker 192.168.1.50    # (collector runs separately)
    python iobt_step_builder.py --step 0.5               # bucket size in seconds
    python iobt_step_builder.py --conf 0.6               # min confidence to count
    python iobt_step_builder.py --class-filter car person # only these classes
    python iobt_step_builder.py --session 20250812_165716
    python iobt_step_builder.py --files data/node1.jsonl data/node5.jsonl
    python iobt_step_builder.py --input-dir ./raw --output-dir ./arrays
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_NODES      = [f"node{i}" for i in range(1, 11)]   # node1 … node10
N_NODES        = len(ALL_NODES)                         # 10
NODE_TO_COL    = {n: i for i, n in enumerate(ALL_NODES)}  # node1->0 … node10->9

DEFAULT_INPUT_DIR  = "./data"
DEFAULT_OUTPUT_DIR = "./data"
DEFAULT_STEP_S     = 0.5    # seconds
DEFAULT_CONF       = 0.5    # minimum confidence to count as a detection

NODE_TS_FMT   = "%Y-%m-%d %H:%M:%S.%f"
SESSION_RE    = re.compile(r"node\d+_(\d{8}_\d{6})\.jsonl$")


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def snap_to_bucket(t_str: str, step_s: float) -> float:
    """Parse a node timestamp string and floor to the nearest step_s boundary."""
    dt = datetime.strptime(t_str, NODE_TS_FMT)
    ts = dt.timestamp()
    return math.floor(ts / step_s) * step_s


def bucket_to_str(bucket_ts: float) -> str:
    """Convert bucket Unix timestamp to a readable local string."""
    dt = datetime.fromtimestamp(bucket_ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def build_buckets(
    jsonl_files:   List[Path],
    step_s:        float,
    conf_threshold: float,
    class_filter:  Optional[Set[str]],
) -> Tuple[Dict, dict]:
    """
    Read all .jsonl files and accumulate per-(bucket_ts, node) max confidence.

    Returns
    -------
    detections : dict[(bucket_ts, node)] -> float  (max conf seen in bucket)
    meta       : raw stats dict
    """
    # (bucket_ts, node) -> max confidence of any qualifying detection
    detections:  Dict[Tuple[float, str], float] = defaultdict(float)
    total_lines  = 0
    total_dets   = 0
    parse_errors = 0
    nodes_seen:  Set[str] = set()

    for fpath in sorted(jsonl_files):
        node_from_file = _node_from_filename(fpath)
        print(f"  {fpath.name} … ", end="", flush=True)
        file_lines = 0
        file_dets  = 0

        with open(fpath) as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                det_list = record.get("detections", [])
                if not isinstance(det_list, list):
                    parse_errors += 1
                    continue

                file_lines += 1

                for det in det_list:
                    if not isinstance(det, dict):
                        continue

                    node = det.get("node") or node_from_file
                    if node not in NODE_TO_COL:
                        continue   # unknown node name — skip

                    t_str = det.get("t")
                    if not t_str:
                        parse_errors += 1
                        continue

                    cls  = det.get("class", "")
                    conf = float(det.get("conf", 0.0))

                    # Apply class filter if requested
                    if class_filter and cls not in class_filter:
                        continue

                    try:
                        bucket_ts = snap_to_bucket(t_str, step_s)
                    except ValueError:
                        parse_errors += 1
                        continue

                    # Keep the max confidence seen for this (bucket, node) pair
                    key = (bucket_ts, node)
                    if conf > detections[key]:
                        detections[key] = conf

                    nodes_seen.add(node)
                    file_dets += 1

        total_lines += file_lines
        total_dets  += file_dets
        print(f"{file_lines} msgs, {file_dets} dets")

    meta = {
        "total_lines":  total_lines,
        "total_dets":   total_dets,
        "parse_errors": parse_errors,
        "nodes_seen":   sorted(nodes_seen),
    }
    return dict(detections), meta


def _node_from_filename(fpath: Path) -> Optional[str]:
    m = re.match(r"(node\d+)_", fpath.name)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Array construction
# ---------------------------------------------------------------------------

def build_arrays(
    detections:    Dict[Tuple[float, str], float],
    conf_threshold: float,
    step_s:        float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the (bucket_ts, node) -> max_conf dict into numpy arrays.

    Returns
    -------
    det_array   : uint8 (T, 10)  — 1 where max_conf >= conf_threshold
    ts_array    : float64 (T,)   — Unix timestamp of each bucket
    """
    if not detections:
        return np.zeros((0, N_NODES), dtype=np.uint8), np.array([], dtype=np.float64)

    # Sorted unique bucket timestamps → row indices
    all_ts    = sorted(set(bt for bt, _ in detections.keys()))
    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}
    T         = len(all_ts)

    det_array = np.zeros((T, N_NODES), dtype=np.uint8)

    for (bucket_ts, node), max_conf in detections.items():
        if max_conf >= conf_threshold:
            row = ts_to_idx[bucket_ts]
            col = NODE_TO_COL[node]
            det_array[row, col] = 1

    ts_array = np.array(all_ts, dtype=np.float64)
    return det_array, ts_array


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def discover_sessions(input_dir: Path) -> Dict[str, List[Path]]:
    sessions: Dict[str, List[Path]] = defaultdict(list)
    for fpath in sorted(input_dir.glob("node*.jsonl")):
        m = SESSION_RE.search(fpath.name)
        if m:
            sessions[m.group(1)].append(fpath)
    return dict(sessions)


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def save_session(
    det_array:     np.ndarray,
    ts_array:      np.ndarray,
    session_ts:    str,
    out_dir:       Path,
    input_files:   List[Path],
    step_s:        float,
    conf_threshold: float,
    class_filter:  Optional[Set[str]],
    meta:          dict,
) -> None:
    stem = f"detections_{session_ts}"

    det_path  = out_dir / f"{stem}.npy"
    ts_path   = out_dir / f"timestamps_{session_ts}.npy"
    meta_path = out_dir / f"metadata_{session_ts}.json"

    np.save(det_path, det_array)
    np.save(ts_path,  ts_array)

    T = len(ts_array)
    ts_strings = [bucket_to_str(float(ts)) for ts in ts_array]

    metadata = {
        "schema_version":   1,
        "created_at_utc":   datetime.now(timezone.utc).isoformat(),
        "session_ts":       session_ts,
        "step_s":           step_s,
        "conf_threshold":   conf_threshold,
        "class_filter":     sorted(class_filter) if class_filter else None,
        "node_ordering":    ALL_NODES,          # col i = ALL_NODES[i]
        "array_shape":      list(det_array.shape),
        "num_timesteps":    T,
        "num_nodes":        N_NODES,
        "ts_strings":       ts_strings,         # human-readable per timestep
        "nodes_seen":       meta["nodes_seen"],
        "raw_messages":     meta["total_lines"],
        "raw_detections":   meta["total_dets"],
        "parse_errors":     meta["parse_errors"],
        "input_files":      [str(p) for p in input_files],
        "detection_array_file": str(det_path),
        "timestamp_array_file": str(ts_path),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Array shape        : {det_array.shape}  (timesteps × nodes)")
    print(f"  Active cells       : {int(det_array.sum())}  "
          f"({int(det_array.sum()) / max(det_array.size, 1) * 100:.1f}% of array)")
    print(f"  Timesteps          : {T}")
    if T:
        print(f"  Time range         : {ts_strings[0]}  →  {ts_strings[-1]}")
        dur = float(ts_array[-1]) - float(ts_array[0]) + step_s
        print(f"  Coverage           : {dur:.1f} s")
    print(f"  detections .npy    : {det_path}")
    print(f"  timestamps  .npy   : {ts_path}")
    print(f"  metadata    .json  : {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLO .jsonl collector files to numpy detection arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output arrays
-------------
  detections_SESSION.npy  — uint8 (T, 10): arr[t, i] = 1 if node_{i+1} detected
                             a qualifying object in timestep t
  timestamps_SESSION.npy  — float64 (T,): Unix timestamp of each bucket floor
  metadata_SESSION.json   — step_s, conf_threshold, node ordering, ts_strings, ...

Node-to-column mapping (fixed):
  node1=col0  node2=col1  ...  node10=col9

Examples
--------
  python iobt_step_builder.py
  python iobt_step_builder.py --step 0.5 --conf 0.6 --class-filter car person
  python iobt_step_builder.py --session 20250812_165716 --output-dir ./arrays
  python iobt_step_builder.py --files data/node1.jsonl data/node10.jsonl
        """
    )
    parser.add_argument("--input-dir",    default=DEFAULT_INPUT_DIR,  metavar="DIR")
    parser.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR, metavar="DIR")
    parser.add_argument("--step",  type=float, default=DEFAULT_STEP_S, metavar="S",
                        help=f"Bucket size in seconds (default: {DEFAULT_STEP_S})")
    parser.add_argument("--conf",  type=float, default=DEFAULT_CONF,  metavar="F",
                        help=f"Min confidence to count as detection (default: {DEFAULT_CONF})")
    parser.add_argument("--class-filter", nargs="+", metavar="CLASS",
                        help="Only count these YOLO class names (default: all classes)")
    parser.add_argument("--session", type=str, default=None, metavar="YYYYMMDD_HHMMSS")
    parser.add_argument("--files",   nargs="+", metavar="FILE",
                        help="Explicit .jsonl files (overrides --input-dir)")
    args = parser.parse_args()

    class_filter: Optional[Set[str]] = set(args.class_filter) if args.class_filter else None

    print("=" * 60)
    print("  IoBT STEP BUILDER")
    print("=" * 60)
    print(f"  Step size     : {args.step} s")
    print(f"  Min conf      : {args.conf}")
    print(f"  Class filter  : {sorted(class_filter) if class_filter else 'all classes'}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve input files ───────────────────────────────────────────────
    if args.files:
        input_files = [Path(f) for f in args.files]
        missing = [f for f in input_files if not f.exists()]
        if missing:
            print(f"[ERROR] Files not found: {missing}")
            sys.exit(1)
        sessions = {"manual": input_files}
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] Input directory not found: {input_dir}")
            sys.exit(1)
        sessions = discover_sessions(input_dir)
        if args.session:
            if args.session not in sessions:
                print(f"[ERROR] Session {args.session} not found.")
                print(f"  Available: {sorted(sessions.keys())}")
                sys.exit(1)
            sessions = {args.session: sessions[args.session]}

    if not sessions:
        print(f"[ERROR] No node*.jsonl files found in {args.input_dir}")
        sys.exit(1)

    # ── Process each session ──────────────────────────────────────────────
    for session_ts, files in sorted(sessions.items()):
        print(f"\n{'─'*60}")
        print(f"  Session: {session_ts}  ({len(files)} node file(s))")
        print(f"{'─'*60}")

        detections, meta = build_buckets(
            files, args.step, args.conf, class_filter
        )
        det_array, ts_array = build_arrays(detections, args.conf, args.step)

        save_session(
            det_array      = det_array,
            ts_array       = ts_array,
            session_ts     = session_ts,
            out_dir        = out_dir,
            input_files    = files,
            step_s         = args.step,
            conf_threshold = args.conf,
            class_filter   = class_filter,
            meta           = meta,
        )

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()