#!/usr/bin/env python3
"""
mqtt_collector.py — Subscribe to /<node>/respeaker/power for nodes
dvpg_gq_orin_11 through dvpg_gq_orin_16, downsample to one sample per
speaker per second, collect for N steps, and save a numpy array of shape
(steps, N_NODES).

Each row arr[t] contains the most recent power reading for each speaker
received during step t.  arr[t, i] = power of KNOWN_NODES[i] at step t.

Output files
------------
    data/respeaker_power_YYYYMMDD_HHMMSS.npy       float32 (steps, N_NODES)
    data/respeaker_power_ts_YYYYMMDD_HHMMSS.npy    float64 (steps,) Unix timestamps
    data/respeaker_power_YYYYMMDD_HHMMSS_meta.json  metadata
    data/respeaker_power_YYYYMMDD_HHMMSS_steps.jsonl  per-step records

Usage
-----
    python mqtt_collector.py --broker 192.168.68.51
    python mqtt_collector.py --broker 192.168.68.51 --steps 1000
"""

import argparse
import json
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_NODES  = list(range(11, 17))          # 11, 12, 13, 14, 15, 16
N_NODES      = len(KNOWN_NODES)             # 6
NODE_TO_COL  = {n: i for i, n in enumerate(KNOWN_NODES)}

NODE_NAMES   = [f"dvpg_gq_orin_{n}" for n in KNOWN_NODES]
TOPICS       = [f"/{name}/respeaker/power" for name in NODE_NAMES]

DEFAULT_BROKER  = "192.168.68.51"
DEFAULT_PORT    = 1883
DEFAULT_STEPS   = 100000
DEFAULT_STEP_S  = 1.0          # 1 sample per speaker per second
DEFAULT_OUT_DIR = "./data"


# ---------------------------------------------------------------------------
# Payload parser
# ---------------------------------------------------------------------------

def parse_power(payload_bytes: bytes) -> float | None:
    """
    Parse a scalar power value from the raw MQTT payload.

    Handles:
      42.7                    (plain float/int)
      {"power": 42.7}         (JSON dict with "power" key)
      {"value": 42.7}         (JSON dict with "value" key)
      [42.7]                  (single-element list)
    Returns None on parse failure.
    """
    try:
        text = payload_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None

    # Try plain numeric first (fastest path)
    try:
        return float(text)
    except ValueError:
        pass

    # Try JSON
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(raw, (int, float)):
        return float(raw)

    if isinstance(raw, list) and len(raw) >= 1 and isinstance(raw[0], (int, float)):
        return float(raw[0])

    if isinstance(raw, dict):
        for key in ("power", "value", "level", "db", "amplitude"):
            if key in raw and isinstance(raw[key], (int, float)):
                return float(raw[key])
        # Fall back to first numeric value
        for v in raw.values():
            if isinstance(v, (int, float)):
                return float(v)

    return None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class RespeakerPowerCollector:
    def __init__(self, broker, port, steps, step_s, out_dir):
        self._broker  = broker
        self._port    = port
        self._steps   = steps
        self._step_s  = step_s
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # Latest power per node — updated by on_message, read by sample loop
        self._latest_power: np.ndarray = np.zeros(N_NODES, dtype=np.float32)
        self._latest_rx_ts: list[str]  = [""] * N_NODES
        self._latest_lock  = threading.Lock()
        self._connected    = threading.Event()
        self._stop         = threading.Event()

        # Preview: print first message per topic before collection starts
        self._preview_seen: set[str] = set()
        self._preview_lock = threading.Lock()

        self._client = mqtt.Client(client_id="respeaker_power_collector",
                                   clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"[ERROR] Connection failed (rc={rc})")
            self._stop.set()
            return
        print(f"[OK] Connected to {self._broker}:{self._port}")
        for topic in TOPICS:
            client.subscribe(topic, qos=0)
            print(f"[OK] Subscribed to '{topic}'")
        print()
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        topic     = msg.topic
        raw_bytes = msg.payload
        rx_ts     = datetime.now(timezone.utc).isoformat()

        # Identify which node sent this message
        col = None
        for i, t in enumerate(TOPICS):
            if t == topic:
                col = i
                break
        if col is None:
            return

        # Print first message per topic for format confirmation
        with self._preview_lock:
            if topic not in self._preview_seen:
                try:
                    decoded = raw_bytes.decode("utf-8", errors="replace")
                except Exception:
                    decoded = "<binary>"
                print(f"  [RAW] {topic}: {decoded}")
                self._preview_seen.add(topic)

        power = parse_power(raw_bytes)
        if power is None:
            return

        with self._latest_lock:
            self._latest_power[col] = power
            self._latest_rx_ts[col] = rx_ts

    def run(self) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Connect, wait for at least one message per topic (or 10 s), then
        collect self._steps samples at self._step_s intervals.

        Returns
        -------
        array      : float32 (steps, N_NODES)  — power values
        timestamps : float64 (steps,)           — Unix seconds of each sample
        step_records : list[dict]               — one record per step
        """
        print(f"Connecting to {self._broker}:{self._port} ...")
        self._client.connect(self._broker, self._port, keepalive=60)
        self._client.loop_start()

        if not self._connected.wait(timeout=10):
            print("[ERROR] Could not connect within 10 s.")
            self._client.loop_stop()
            return (np.zeros((0, N_NODES), dtype=np.float32),
                    np.array([]), [])

        # Preview phase — wait until all topics seen or timeout
        print("Waiting for first message from each speaker (up to 10 s) ...")
        deadline = time.time() + 10
        while time.time() < deadline:
            with self._preview_lock:
                seen = len(self._preview_seen)
            if seen >= N_NODES:
                break
            time.sleep(0.1)

        with self._preview_lock:
            seen = len(self._preview_seen)
        if seen == 0:
            print("[WARN] No messages received — check topics and broker.")
        else:
            print(f"\nReceived messages from {seen}/{N_NODES} speakers.")
            print(f"Node mapping: {list(zip(NODE_NAMES, range(N_NODES)))}")
            print(f"\nStarting collection: {self._steps} steps × {self._step_s} s "
                  f"= {self._steps * self._step_s:.0f} s total\n")

        # Collection phase
        self._partial_array        = np.zeros((self._steps, N_NODES), dtype=np.float32)
        self._partial_timestamps   = np.zeros(self._steps,            dtype=np.float64)
        self._partial_step_records = []
        self._partial_steps_done   = 0

        array        = self._partial_array
        timestamps   = self._partial_timestamps
        step_records = self._partial_step_records

        for t in range(self._steps):
            step_start = time.time()

            with self._latest_lock:
                array[t]  = self._latest_power.copy()
                rx_ts_snap = list(self._latest_rx_ts)
            timestamps[t] = step_start

            step_records.append({
                "step":        t,
                "sample_ts":   datetime.fromtimestamp(step_start,
                                                       tz=timezone.utc).isoformat(),
                "rx_ts":       rx_ts_snap,
                "topics":      TOPICS,
                "power_values": array[t].tolist(),
            })

            # Status every 10 steps
            if t % 10 == 0 or t == self._steps - 1:
                vals = ", ".join(f"{n}={array[t, i]:.1f}"
                                 for i, n in enumerate(KNOWN_NODES))
                print(f"  step {t+1:5d}/{self._steps}  [{vals}]")

            self._partial_steps_done = t + 1

            # Sleep remainder of step window
            elapsed   = time.time() - step_start
            remaining = self._step_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._client.loop_stop()
        self._client.disconnect()
        return array, timestamps, step_records

    def save(self, array: np.ndarray, timestamps: np.ndarray,
             step_records: list | None = None) -> None:
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arr_path   = self._out_dir / f"respeaker_power_{session_ts}.npy"
        ts_path    = self._out_dir / f"respeaker_power_ts_{session_ts}.npy"
        meta_path  = self._out_dir / f"respeaker_power_{session_ts}_meta.json"
        steps_path = self._out_dir / f"respeaker_power_{session_ts}_steps.jsonl"

        np.save(arr_path, array)
        np.save(ts_path,  timestamps)

        ts_strings = [
            datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
            for ts in timestamps
        ]
        meta = {
            "created_at_utc":  datetime.now(timezone.utc).isoformat(),
            "topics":          TOPICS,
            "broker":          self._broker,
            "steps":           int(array.shape[0]),
            "step_s":          self._step_s,
            "n_nodes":         N_NODES,
            "node_ids":        KNOWN_NODES,
            "node_names":      NODE_NAMES,
            "node_to_col":     {str(k): v for k, v in NODE_TO_COL.items()},
            "array_shape":     list(array.shape),
            "array_dtype":     str(array.dtype),
            "ts_strings":      ts_strings,
            "array_file":      str(arr_path),
            "timestamps_file": str(ts_path),
            "steps_file":      str(steps_path),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        records = step_records or []
        with open(steps_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        print(f"\n  Array shape  : {array.shape}  dtype={array.dtype}")
        print(f"  Saved array  → {arr_path}")
        print(f"  Saved ts     → {ts_path}")
        print(f"  Saved meta   → {meta_path}")
        print(f"  Saved steps  → {steps_path}  ({len(records)} records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect respeaker/power from 6 IOBT nodes into a numpy array.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output
------
  respeaker_power_YYYYMMDD_HHMMSS.npy       float32 (steps, 6)
  respeaker_power_ts_YYYYMMDD_HHMMSS.npy   float64 (steps,) Unix timestamps
  respeaker_power_YYYYMMDD_HHMMSS_meta.json
  respeaker_power_YYYYMMDD_HHMMSS_steps.jsonl

Load example
------------
  import numpy as np, json
  arr  = np.load("data/respeaker_power_20260416_120000.npy")
  ts   = np.load("data/respeaker_power_ts_20260416_120000.npy")
  meta = json.load(open("data/respeaker_power_20260416_120000_meta.json"))
  # arr[t, i] = power of node KNOWN_NODES[i] at step t
        """
    )
    parser.add_argument("--broker",   default=DEFAULT_BROKER)
    parser.add_argument("--port",     type=int,   default=DEFAULT_PORT)
    parser.add_argument("--steps",    type=int,   default=DEFAULT_STEPS,
                        help=f"Number of 1-second samples to collect (default: {DEFAULT_STEPS})")
    parser.add_argument("--step-s",   type=float, default=DEFAULT_STEP_S,
                        help=f"Seconds per sample (default: {DEFAULT_STEP_S})")
    parser.add_argument("--out-dir",  default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("  RESPEAKER POWER COLLECTOR")
    print("=" * 60)
    print(f"  Broker     : {args.broker}:{args.port}")
    print(f"  Topics     : {TOPICS[0]}")
    for t in TOPICS[1:]:
        print(f"               {t}")
    print(f"  Steps      : {args.steps}")
    print(f"  Step size  : {args.step_s} s  (1 sample/speaker/s)")
    print(f"  Duration   : {args.steps * args.step_s:.0f} s")
    print(f"  Nodes      : {KNOWN_NODES}")
    print(f"  Output dir : {args.out_dir}")
    print()

    collector = RespeakerPowerCollector(
        broker  = args.broker,
        port    = args.port,
        steps   = args.steps,
        step_s  = args.step_s,
        out_dir = args.out_dir,
    )

    try:
        array, timestamps, step_records = collector.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — saving partial results.")
        n = getattr(collector, "_partial_steps_done", 0)
        array        = collector._partial_array[:n]        if n > 0 else np.zeros((0, N_NODES), dtype=np.float32)
        timestamps   = collector._partial_timestamps[:n]   if n > 0 else np.array([])
        step_records = collector._partial_step_records[:n] if n > 0 else []
        print(f"[INFO] Captured {n} step(s) before interrupt.")

    print("\n" + "=" * 60)
    print("  SAVING")
    print("=" * 60)
    collector.save(array, timestamps, step_records)
    print("\nDone.")


if __name__ == "__main__":
    main()
