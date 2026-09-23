#!/usr/bin/env python
"""
calibrate_detector.py — offline sweep over recorded detection streams.

Replays recorded streams (from record_detection_streams.py) through a grid of
detector configurations and reports each one on BOTH axes:

  ARL0  mean environment steps between alarms on a stream containing no change
  ARL1  environment steps from a true switch to the first alarm after it

Configurations are only comparable at matched ARL0. Comparing at matched delta
is what produced the original miscalibration: the same delta=0.01 gave ARL0
between 4,166 and 13,888 steps across arms, purely because they fed the
detector at different rates.

The headline output is the operating characteristic — detection delay against
false-alarm rate, one curve per candidate signal. If one signal's curve sits
above another's at every false-alarm rate, no threshold tuning closes the gap
and the answer is to monitor something else.

Example
-------
    ./track_mdp_env/bin/python examples/calibrate_detector.py \\
        --switching experiments/detector_calibration/grid_learn_switch.npz \\
        --stationary experiments/detector_calibration/grid_learn_stat.npz \\
        --out experiments/detector_calibration/grid_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.detection_signals import SignalRecorder      # noqa: E402
from src.core.detector_eval import (                       # noqa: E402
    at_target_arl0, operating_curve,
)

# Bernoulli only suits the 0/1 hit stream; everything else is continuous.
SIGNAL_DIVERGENCE = {
    "hit": "bernoulli",
    "miss_run": "gaussian",
    "staleness": "gaussian",
    "reward": "gaussian",
    "sensors": "gaussian",
    "transition_surprise": "gaussian",
}


def truncate(stream, limit):
    """Keep the prefix of a stream up to ``limit`` environment steps."""
    if limit is None:
        return stream
    steps, values = stream["steps"], stream["values"]
    keep = [i for i, s in enumerate(steps) if s < limit]
    return {"steps": [steps[i] for i in keep],
            "values": [values[i] for i in keep]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--switching", required=True)
    p.add_argument("--stationary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--signals", type=str, default=",".join(SIGNAL_DIVERGENCE))
    p.add_argument("--n-deltas", type=int, default=8)
    p.add_argument("--delta-lo", type=float, default=1e-12)
    p.add_argument("--delta-hi", type=float, default=0.4)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--max-window", type=int, default=200,
                   help="Replay cost is O(window) per step; 200 keeps a full "
                        "sweep to minutes instead of an hour")
    p.add_argument("--stationary-limit", type=int, default=200000,
                   help="Prefix of the stationary stream used for ARL0. It is "
                        "stationary, so a prefix is representative")
    p.add_argument("--target-arl0", type=float, default=50000.0)
    args = p.parse_args()

    sw = SignalRecorder.load(args.switching)
    st = SignalRecorder.load(args.stationary)
    switches = sw["switch_steps"]
    horizon = sw["total_steps"]
    deltas = np.logspace(np.log10(args.delta_lo), np.log10(args.delta_hi),
                         args.n_deltas)

    names = [s.strip() for s in args.signals.split(",") if s.strip()]

    print("=" * 78)
    print("  DETECTOR CALIBRATION SWEEP")
    print("=" * 78)
    print(f"  switching : {args.switching}")
    print(f"              {horizon:,} steps, {len(switches)} switches at {switches}")
    print(f"  stationary: {args.stationary} (first {args.stationary_limit:,} steps)")
    print(f"  deltas    : {args.n_deltas} log-spaced over "
          f"[{args.delta_lo:.0e}, {args.delta_hi:.2f}]")
    print(f"  window    : {args.max_window}, min_samples {args.min_samples}")
    print(f"  target    : ARL0 >= {args.target_arl0:,.0f} steps\n")

    results, summary = {}, []
    print(f"  {'signal':>20} {'samples':>9} {'ARL0':>10} {'delay':>9} "
          f"{'caught':>8} {'delta':>10} {'secs':>6}")
    print(f"  {'-'*20} {'-'*9} {'-'*10} {'-'*9} {'-'*8} {'-'*10} {'-'*6}")

    for name in names:
        if name not in sw["signals"]:
            continue
        div = SIGNAL_DIVERGENCE.get(name, "gaussian")
        a = sw["signals"][name]
        b = truncate(st["signals"][name], args.stationary_limit)
        t0 = time.time()
        curve = operating_curve(a, b, switches, deltas,
                                min_samples=args.min_samples,
                                max_window=args.max_window,
                                divergence=div, horizon=horizon)
        dt = time.time() - t0
        results[name] = curve

        best = at_target_arl0(curve, args.target_arl0)
        met_target = best is not None
        if best is None:
            # Do NOT silently substitute a row at a different false-alarm rate:
            # that is precisely the matched-delta mistake this sweep exists to
            # avoid. Report the best achievable ARL0 and flag it.
            cand = [r for r in curve if r["mean_delay"] is not None]
            best = max(cand, key=lambda r: r["arl0"]) if cand else None

        if best is None:
            print(f"  {name:>20} {len(a['values']):>9,} {'-':>10} "
                  f"{'never':>9} {'0/'+str(len(switches)):>8} {'-':>10} {dt:>6.0f}")
            summary.append({"signal": name, "detectable": False})
        else:
            flag = "" if met_target else "  << BELOW TARGET ARL0"
            print(f"  {name:>20} {len(a['values']):>9,} {best['arl0']:>10.0f} "
                  f"{best['mean_delay']:>9.0f} "
                  f"{str(best['n_caught'])+'/'+str(best['n_switches']):>8} "
                  f"{best['delta']:>10.2e} {dt:>6.0f}{flag}")
            summary.append({"signal": name, "detectable": True,
                            "met_target_arl0": met_target,
                            "arl0": best["arl0"],
                            "mean_delay": best["mean_delay"],
                            "n_caught": best["n_caught"],
                            "n_switches": best["n_switches"],
                            "delta": best["delta"],
                            "per_switch_delays": best["delays"],
                            "samples": len(a["values"])})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"meta": {"switching": args.switching,
                            "stationary": args.stationary,
                            "switches": switches, "horizon": horizon,
                            "target_arl0": args.target_arl0,
                            "min_samples": args.min_samples,
                            "max_window": args.max_window},
                   "summary": summary,
                   "curves": {k: [{kk: vv for kk, vv in r.items()
                                   if kk != "delays"} for r in v]
                              for k, v in results.items()}}, fh, indent=1)

    print(f"\n  Wrote {args.out}")
    ranked = [s for s in summary
              if s.get("detectable") and s.get("met_target_arl0")]
    rejected = [s for s in summary
                if s.get("detectable") and not s.get("met_target_arl0")]
    if rejected:
        print(f"\n  Could NOT reach ARL0 >= {args.target_arl0:,.0f} at any "
              f"delta in the grid (their own streams are non-stationary):")
        for s in rejected:
            print(f"    {s['signal']:>20}: best ARL0 {s['arl0']:,.0f}")
    if ranked:
        ranked.sort(key=lambda s: (s["n_switches"] - s["n_caught"],
                                   s["mean_delay"]))
        print(f"\n  Ranked at ARL0 >= {args.target_arl0:,.0f}:")
        for s in ranked:
            print(f"    {s['signal']:>20}: {s['n_caught']}/{s['n_switches']} "
                  f"caught, mean delay {s['mean_delay']:,.0f} steps")


if __name__ == "__main__":
    main()
