#!/usr/bin/env python
"""
summarize_dal_validation.py — detection and accuracy table for transition_dal runs.

Each run directory holds online_summary.json and alarms.json (written by
src/core/online_dal.py).  An alarm counts as catching a switch when it fires
0..tolerance episodes after it; every other alarm is a false alarm.

Usage:
    ./track_mdp_env/bin/python examples/summarize_dal_validation.py \\
        --runs 'experiments/transition_dal/null/*' --tolerance-episodes 300 \\
        --out experiments/transition_dal/null_summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def detection_metrics(alarms: list[dict], switches: list[dict],
                      tolerance_episodes: int) -> dict:
    sw = sorted(s["episode"] for s in switches)
    matched = {}
    false_alarms = 0
    for a in sorted(x["episode"] for x in alarms):
        hit = next((k for k, s in enumerate(sw)
                    if k not in matched and 0 <= a - s <= tolerance_episodes), None)
        if hit is None:
            false_alarms += 1
        else:
            matched[hit] = a - sw[hit]
    return {"caught": len(matched), "n_switches": len(sw),
            "false_alarms": false_alarms,
            "delays": [matched[k] for k in sorted(matched)]}


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.2f}" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def summarize_runs(run_dirs: list[str], tolerance_episodes: int):
    rows = []
    for d in sorted(run_dirs):
        sp, ap = os.path.join(d, "online_summary.json"), os.path.join(d, "alarms.json")
        if not os.path.exists(sp):
            continue
        if not os.path.exists(ap):
            print(f"warning: {d} has online_summary.json but is missing alarms.json; skipped",
                  file=sys.stderr)
            continue
        with open(sp) as fh:
            s = json.load(fh)["summary"]
        with open(ap) as fh:
            al = json.load(fh)
        m = detection_metrics(al["alarms"], al["switches"], tolerance_episodes)
        rows.append({
            "run": os.path.basename(os.path.normpath(d)), "env": s.get("env"),
            "detector": s.get("detector"), "mode": s.get("detector_mode"),
            "rho": s.get("probe_rate"), "seed": s.get("train_seed"),
            "caught": m["caught"], "n_switches": m["n_switches"],
            "false_alarms": m["false_alarms"],
            "median_delay": float(np.median(m["delays"])) if m["delays"] else None,
            "greedy_mean": s.get("greedy_mean_accuracy"),
            "greedy_end": s.get("greedy_final_accuracy"),
            "sensors": s.get("greedy_mean_sensors"),
            # training sensors/step: the only figure that includes probe steps
            # (greedy evaluation never probes)
            "train_sensors": s.get("mean_sensors"),
            "probe_fraction": s.get("probe_fraction"),
            "restarts": s.get("n_restarts")})
    head = ("| run | env | detector | mode | rho | seed | caught | false alarms | "
            "median delay (ep) | greedy mean % | greedy end % | greedy sensors/step | "
            "train sensors/step | probe frac % | restarts |")
    lines = [head, "|" + "---|" * 15]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['env']} | {r['detector']} | {r['mode']} | "
            f"{_fmt(r['rho'])} | {r['seed']} | {r['caught']}/{r['n_switches']} | "
            f"{r['false_alarms']} | {_fmt(r['median_delay'])} | "
            f"{_fmt(r['greedy_mean'], True)} | {_fmt(r['greedy_end'], True)} | "
            f"{_fmt(r['sensors'])} | {_fmt(r['train_sensors'])} | "
            f"{_fmt(r['probe_fraction'], True)} | {r['restarts']} |")
    return rows, "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   help="Glob of run directories; may be given more than once")
    p.add_argument("--tolerance-episodes", type=int, default=300)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dirs = [d for g in args.runs for d in glob.glob(g) if os.path.isdir(d)]
    _, md = summarize_runs(dirs, args.tolerance_episodes)
    with open(args.out, "w") as fh:
        fh.write(f"Tolerance: {args.tolerance_episodes} episodes\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()
