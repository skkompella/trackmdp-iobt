"""
detector_eval.py — step-resolution scoring of a change detector.

Replaces `detector_scores` (src/core/online_schedule.py) for calibration work.
That function matches alarms to switches at EPISODE resolution with a tolerance
of `episodes_per_segment // 5`, which in run 821 meant 204 episodes; at that
tolerance "precision 0.07, recall 1.0" carries almost no information. What
calibration needs is detection delay measured in environment steps, against a
separately measured false-alarm rate.

The two axes, which must be reported together:

  ARL0  mean environment steps between alarms on a stream containing NO change.
        Every alarm there is false, so this is the false-alarm axis.
  ARL1  environment steps from a true switch to the first alarm after it.
        The responsiveness axis.

Any detector can be made to look good on one axis alone. A configuration is
only comparable to another at MATCHED ARL0 — which is precisely what was not
done before, and is why the tabular arm ended up firing at a 1.0pp shift while
PPO fired at 2.1pp from the same delta.
"""
from __future__ import annotations

import numpy as np

from .change_detection import GLRChangeDetector


def replay(stream, delta: float, min_samples: int = 30, exponent: float = 1.5,
           max_window: int | None = 500, divergence: str = "bernoulli",
           variance: float | None = None) -> list[int]:
    """
    Push a recorded stream through a detector; return alarm ENVIRONMENT STEPS.

    ``stream`` is {"steps": [...], "values": [...]}. Sparse signals update only
    on some steps, so the returned indices come from ``steps``, never from the
    sample position — mixing those up would make signals of different rates
    incomparable.
    """
    values = stream["values"]
    steps = stream["steps"]
    if len(values) == 0:
        return []

    det = GLRChangeDetector(delta=delta, min_samples=min_samples,
                            exponent=exponent, max_window=max_window,
                            divergence=divergence, variance=variance)
    return [int(steps[i]) for i, v in enumerate(values) if det.update(v)]


def detection_delays(fire_steps, switch_steps, horizon: int):
    """
    Delay from each switch to the first alarm after it, in environment steps.

    Returns one entry per switch: an int delay, or None when the switch was
    missed. A switch is missed when no unused alarm falls between it and the
    next switch — an alarm arriving after the next switch is too late to be
    evidence for this one, and an alarm before a switch is a false alarm rather
    than early insight.
    """
    fires = sorted(int(f) for f in fire_steps)
    switches = sorted(int(s) for s in switch_steps)
    used: set[int] = set()
    out: list[int | None] = []

    for i, sw in enumerate(switches):
        end = switches[i + 1] if i + 1 < len(switches) else horizon
        hit = None
        for j, f in enumerate(fires):
            if j in used:
                continue
            if sw <= f < end:
                hit = f - sw
                used.add(j)
                break
        out.append(hit)
    return out


def evaluate_config(switching_stream, stationary_stream, switch_steps,
                    delta: float, min_samples: int = 30, exponent: float = 1.5,
                    max_window: int | None = 500,
                    divergence: str = "bernoulli",
                    variance: float | None = None,
                    horizon: int | None = None) -> dict:
    """Score one detector configuration on both axes at once."""
    kw = dict(min_samples=min_samples, exponent=exponent,
              max_window=max_window, divergence=divergence, variance=variance)

    st_steps = stationary_stream["steps"]
    st_span = (int(st_steps[-1]) + 1) if len(st_steps) else 0
    st_fires = replay(stationary_stream, delta, **kw)
    arl = float("inf") if not st_fires else st_span / len(st_fires)

    sw_steps = switching_stream["steps"]
    span = horizon if horizon is not None else (
        (int(sw_steps[-1]) + 1) if len(sw_steps) else 0)
    sw_fires = replay(switching_stream, delta, **kw)
    delays = detection_delays(sw_fires, switch_steps, span)
    caught = [d for d in delays if d is not None]

    return {
        "delta": float(delta),
        "min_samples": int(min_samples),
        "exponent": float(exponent),
        "max_window": max_window,
        "divergence": divergence,
        "arl0": arl,
        "n_false_alarms": len(st_fires),
        "n_alarms_switching": len(sw_fires),
        "n_switches": len(switch_steps),
        "n_caught": len(caught),
        "missed": len(switch_steps) - len(caught),
        "mean_delay": float(np.mean(caught)) if caught else None,
        "median_delay": float(np.median(caught)) if caught else None,
        "delays": delays,
    }


def operating_curve(switching_stream, stationary_stream, switch_steps,
                    deltas, **kw) -> list[dict]:
    """
    Sweep delta and return the delay-vs-false-alarm operating characteristic.

    This is the deliverable that answers whether the SIGNAL or the THRESHOLD is
    the binding constraint. If one signal's curve sits above another's at every
    false-alarm rate, no amount of threshold tuning closes the gap and the
    answer is to monitor something else.
    """
    return [evaluate_config(switching_stream, stationary_stream, switch_steps,
                            delta=float(d), **kw)
            for d in deltas]


def at_target_arl0(curve, target_arl0: float) -> dict | None:
    """
    The most responsive configuration on a curve whose false-alarm rate is no
    worse than the target. Comparing signals means comparing these, not
    comparing raw delta.
    """
    ok = [r for r in curve
          if r["arl0"] >= target_arl0 and r["mean_delay"] is not None]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["missed"], r["mean_delay"]))


__all__ = ["replay", "detection_delays", "evaluate_config",
           "operating_curve", "at_target_arl0"]
