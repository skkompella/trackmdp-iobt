#!/usr/bin/env python3
"""
iobt_collector.py — Subscribe to YOLO detection topics from all IoBT-MAX
sensor nodes and save the raw detections to per-node JSON Lines files.

Each node publishes its YOLO detections on:
    /{hostname}/analytics/yolo/bbox

Payload: JSON array of detection dicts
    [{"node": "node10", "model": "yolov8", "class": "car",
      "conf": 0.91, "box": [cx, cy, w, h],
      "depth": 15.3, "world": [x, y, z], "t": "2025-08-12 16:57:16.428494"},
     ...]

Output: one .jsonl file per node, written to --output-dir (default: ./data/).
    data/node1_YYYYMMDD_HHMMSS.jsonl
    data/node2_YYYYMMDD_HHMMSS.jsonl
    ...

Each line in a .jsonl file is one raw detection message (the full JSON array)
with a receiver-side timestamp prepended:
    {"rx_ts": "2025-08-12T16:57:16.428494Z", "node": "node1",
     "topic": "/node1/analytics/yolo/bbox", "detections": [...]}

Usage
-----
    python iobt_collector.py                        # all 10 nodes, localhost broker
    python iobt_collector.py --broker 192.168.1.50  # explicit broker IP
    python iobt_collector.py --nodes node1 node3    # subset of nodes
    python iobt_collector.py --duration 60          # run for 60 seconds then exit
    python iobt_collector.py --output-dir ./raw     # save to ./raw/
    python iobt_collector.py --port 1883            # non-default MQTT port

Press Ctrl-C to stop and flush all files.
"""

import argparse
import json
import os
import queue
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

# All 10 IoBT-MAX nodes by their short hostname.
# If your network uses .local mDNS names, set --broker to the MQTT broker IP
# (usually one of the nodes or a central hub on the same subnet).
ALL_NODES = [f"node{i}" for i in range(1, 11)]

DEFAULT_BROKER  = "localhost"   # override with --broker
DEFAULT_PORT    = 1883
DEFAULT_OUT_DIR = "./data"
DEFAULT_TOPIC_PATTERN = "/{node}/analytics/yolo/bbox"

# How often (seconds) to flush file buffers to disk even if no data arrived.
FLUSH_INTERVAL_S = 5.0

# ---------------------------------------------------------------------------
# File writer — one per node, thread-safe
# ---------------------------------------------------------------------------

class NodeWriter:
    """
    Writes incoming YOLO detection messages for one node to a .jsonl file.

    Each line is a JSON object:
        {
          "rx_ts":     "<ISO-8601 UTC>",   # when this process received it
          "node":      "node1",
          "topic":     "/node1/analytics/yolo/bbox",
          "detections": [...]              # raw payload array from the node
        }
    """

    def __init__(self, node_name: str, out_dir: Path, session_ts: str) -> None:
        self.node_name  = node_name
        self._lock      = threading.Lock()
        self._count     = 0

        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{node_name}_{session_ts}.jsonl"
        self._file = open(fname, "w", buffering=1)   # line-buffered
        self._path = fname
        print(f"  [{node_name}] writing to {fname}")

    def write(self, topic: str, detections: list) -> None:
        record = {
            "rx_ts":      datetime.now(timezone.utc).isoformat(),
            "node":       self.node_name,
            "topic":      topic,
            "detections": detections,
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._count += 1

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def close(self) -> int:
        with self._lock:
            self._file.flush()
            self._file.close()
        return self._count

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


# ---------------------------------------------------------------------------
# MQTT collector
# ---------------------------------------------------------------------------

class IoBTCollector:
    """
    Subscribes to YOLO detection topics from all requested IoBT nodes and
    writes each received message to the corresponding NodeWriter.
    """

    def __init__(
        self,
        broker:   str,
        port:     int,
        nodes:    list,
        out_dir:  Path,
        duration: float | None,
    ) -> None:
        self._broker   = broker
        self._port     = port
        self._nodes    = nodes
        self._duration = duration
        self._stop     = threading.Event()

        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._writers: dict[str, NodeWriter] = {
            n: NodeWriter(n, out_dir, session_ts) for n in nodes
        }

        # Map topic string -> node name for fast O(1) lookup in on_message
        self._topic_to_node: dict[str, str] = {
            DEFAULT_TOPIC_PATTERN.format(node=n): n for n in nodes
        }

        # Running stats
        self._rx_counts: dict[str, int] = {n: 0 for n in nodes}
        self._err_count  = 0
        self._start_time = 0.0

        # MQTT client
        self._client = mqtt.Client(
            client_id=f"iobt_collector_{session_ts}",
            clean_session=True,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc != 0:
            print(f"[ERROR] MQTT connection failed (rc={rc}). "
                  f"Check broker address and port.")
            self._stop.set()
            return

        print(f"[OK] Connected to MQTT broker {self._broker}:{self._port}")
        for topic in self._topic_to_node:
            client.subscribe(topic, qos=0)
            print(f"     subscribed → {topic}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        if not self._stop.is_set():
            print(f"[WARN] Unexpected disconnect (rc={rc}). "
                  f"Client will attempt to reconnect.")

    def _on_message(self, client, userdata, msg) -> None:
        topic   = msg.topic
        node    = self._topic_to_node.get(topic)

        if node is None:
            # Message on an unexpected topic — ignore
            return

        try:
            detections = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._err_count += 1
            print(f"[WARN] Bad payload from {topic}: {exc}")
            return

        if not isinstance(detections, list):
            # Some nodes wrap in a dict — handle gracefully
            self._err_count += 1
            print(f"[WARN] Unexpected payload type from {topic}: "
                  f"{type(detections).__name__}")
            return

        self._writers[node].write(topic, detections)
        self._rx_counts[node] += 1

    # ------------------------------------------------------------------
    # Periodic flush + status printer
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        next_flush = time.time() + FLUSH_INTERVAL_S
        while not self._stop.is_set():
            now = time.time()
            if now >= next_flush:
                for w in self._writers.values():
                    w.flush()
                next_flush = now + FLUSH_INTERVAL_S
                self._print_status()
            time.sleep(0.5)

    def _print_status(self) -> None:
        elapsed = time.time() - self._start_time
        total   = sum(self._rx_counts.values())
        rate    = total / max(elapsed, 1)
        active  = sum(1 for c in self._rx_counts.values() if c > 0)
        counts  = "  ".join(
            f"{n}:{self._rx_counts[n]}" for n in self._nodes
        )
        print(f"[{elapsed:6.0f}s]  {total:6d} msgs  {rate:.1f}/s  "
              f"active={active}/{len(self._nodes)}   {counts}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"\nConnecting to broker {self._broker}:{self._port} …")
        try:
            self._client.connect(self._broker, self._port, keepalive=60)
        except OSError as exc:
            print(f"[ERROR] Cannot reach broker: {exc}")
            sys.exit(1)

        self._start_time = time.time()
        self._client.loop_start()

        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()

        try:
            if self._duration:
                print(f"Collecting for {self._duration:.0f} seconds …")
                self._stop.wait(timeout=self._duration)
            else:
                print("Collecting — press Ctrl-C to stop …\n")
                self._stop.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self._client.loop_stop()
            self._client.disconnect()
            self._summarise()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Shutdown summary
    # ------------------------------------------------------------------

    def _summarise(self) -> None:
        elapsed = time.time() - self._start_time
        print("\n" + "=" * 60)
        print("  COLLECTION SUMMARY")
        print("=" * 60)
        print(f"  Duration         : {elapsed:.1f} s")
        print(f"  Parse errors     : {self._err_count}")
        print()

        total = 0
        for node in self._nodes:
            n = self._writers[node].close()
            path = self._writers[node].path
            total += n
            status = "✓" if n > 0 else "○"
            print(f"  {status}  {node:8s}  {n:6d} messages   → {path.name}")

        print()
        print(f"  Total messages   : {total}")
        print(f"  Output directory : {self._writers[self._nodes[0]].path.parent}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect YOLO detections from IoBT-MAX sensor nodes via MQTT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Collect from all 10 nodes using a central broker at 192.168.1.50
  python iobt_collector.py --broker 192.168.1.50

  # Collect only from node1 and node5 for 2 minutes
  python iobt_collector.py --broker 192.168.1.50 --nodes node1 node5 --duration 120

  # Save to a custom directory
  python iobt_collector.py --broker 192.168.1.50 --output-dir /data/field_run_01
        """
    )
    parser.add_argument(
        "--broker", default=DEFAULT_BROKER,
        help=f"MQTT broker hostname or IP (default: {DEFAULT_BROKER})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"MQTT broker port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--nodes", nargs="+", default=ALL_NODES,
        metavar="NODE",
        help="Node names to subscribe to (default: node1 … node10)"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        metavar="SECONDS",
        help="Stop after this many seconds (default: run until Ctrl-C)"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"Directory for output .jsonl files (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete all files in the output directory before collecting"
    )
    args = parser.parse_args()

    if args.clear:
        out_dir = Path(args.output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
            print(f"[--clear] Removed {out_dir}")

    # Validate node names
    bad = [n for n in args.nodes if n not in ALL_NODES]
    if bad:
        parser.error(f"Unknown node names: {bad}. Valid: {ALL_NODES}")

    print("=" * 60)
    print("  IoBT-MAX YOLO COLLECTOR")
    print("=" * 60)
    print(f"  Broker     : {args.broker}:{args.port}")
    print(f"  Nodes      : {', '.join(args.nodes)}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Duration   : {args.duration or 'until Ctrl-C'}")
    print()
    print("  Output files (one per node):")

    collector = IoBTCollector(
        broker   = args.broker,
        port     = args.port,
        nodes    = args.nodes,
        out_dir  = Path(args.output_dir),
        duration = args.duration,
    )

    # Allow SIGTERM to trigger a clean shutdown
    signal.signal(signal.SIGTERM, lambda *_: collector.stop())

    collector.run()


if __name__ == "__main__":
    main()