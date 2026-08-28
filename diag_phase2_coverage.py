"""Phase-2 analytic pre-checks (read-only, no training).

Computes GT-node lock coverage on the 130-step eval sequence for session
20250812_165739 under:
  (a) or_max fusion at the lock bars implied by soft-threshold {0.2, 0.25, 0.3}
      with soft-scale 0.8 (bar = threshold / scale), per phase-2 experiment #2;
  (b) temporal max-pool smoothing of P_cam over +/-1 and +/-2 bins before
      fusion, per phase-2 experiment #3 (both camera-only and or_max variants).

Lock coverage = fraction of eval steps where P_fused[t, gt_node] * scale >=
threshold, i.e. the tracker's confirmation test at the true node. Run from the
worktree root with track_mdp_env/bin/python.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from pathlib import Path

from collection.train_node_classifiers_10 import build_p_cam_10, _parse_session_start_10
from examples.finetune_deterministic import build_iobt_gt_sequence, build_p_audio_iobt_10
import joblib

session = "20250812_165739"
session_dir = os.path.join(os.getcwd(), "iobt_data_10", session)

gt_seq, gps_mask = build_iobt_gt_sequence(session_dir, step_s=0.5)
T_full = len(gps_mask)

calib = joblib.load("results/pooled/pooled_calibrators_20260521_164149.pkl")
P_audio = build_p_audio_iobt_10("results/pooled/pooled_clf_20260521_164149.pkl",
                                session_dir, list(range(1, 11)),
                                calibrators=calib, step_s=0.5)
start_unix = _parse_session_start_10(session).timestamp()
P_cam = build_p_cam_10(Path(session_dir), list(range(1, 11)), start_unix, T_full, bin_size_s=0.5)


def maxpool_time(P, w):
    """Max-pool each node's time series over a +/-w bin window."""
    if w == 0:
        return P
    out = P.copy()
    for s in range(1, w + 1):
        out[s:] = np.maximum(out[s:], P[:-s])   # carry forward
        out[:-s] = np.maximum(out[:-s], P[s:])  # carry backward
    return out


t = np.arange(len(gt_seq))
scale = 0.8

print(f"eval steps={len(gt_seq)}  scale={scale}")
print("\n(a) or_max lock coverage vs soft-threshold (no smoothing):")
pa = P_audio[gps_mask]
pc = P_cam[gps_mask]
or_gt = np.maximum(pc, pa)[t, gt_seq]
for thr in (0.4, 0.3, 0.25, 0.2):
    bar = thr / scale
    print(f"  threshold={thr:.2f}  bar={bar:.4f}  coverage={(or_gt >= bar).mean():.4f} "
          f"({(or_gt >= bar).sum()}/{len(gt_seq)})")

print("\n(b) P_cam max-pool smoothing (bar shown = threshold/scale):")
for w in (0, 1, 2, 3):
    pc_s = maxpool_time(P_cam, w)[gps_mask]
    cam_gt = pc_s[t, gt_seq]
    or_s_gt = np.maximum(pc_s, pa)[t, gt_seq]
    # camera-only at threshold 0.4 (bar 0.5), or_max at thresholds 0.4 and 0.2
    dead = (cam_gt < 0.5)
    gaps, run = [], 0
    for d in dead:
        if d: run += 1
        elif run: gaps.append(run); run = 0
    if run: gaps.append(run)
    print(f"  +/-{w} bins: cam-only thr0.4 cov={(cam_gt >= 0.5).mean():.4f} "
          f"({(cam_gt >= 0.5).sum()}/{len(gt_seq)})  "
          f"or_max thr0.4 cov={(or_s_gt >= 0.5).mean():.4f}  "
          f"or_max thr0.2 cov={(or_s_gt >= 0.25).mean():.4f} "
          f"({(or_s_gt >= 0.25).sum()}/{len(gt_seq)})  "
          f"remaining cam gaps: {sorted(gaps, reverse=True)}")


# ── (c) Oracle-policy accuracy upper bound ──────────────────────────────────
# Exact simulation of the eval-loop state machine (evaluate_policy +
# RealIoBTEnv.get_reward_next_state semantics, verified in source):
#   - tracked step: counts as a HIT iff the policy activated the GT node
#     (regardless of soft confirm); confirm success (covered[t]) resets
#     time_delay, failure increments it; time_delay > time_limit -> missing.
#   - missing step: always a MISS; full rescan always succeeds -> tracked next
#     step with time_delay 0.
#   - the 130-step sequence loops (object_move uses modulo), eval is 2000 steps.
# An oracle policy always activates the GT node, so its accuracy is
# 1 - steady-state missing-step rate — an upper bound no trained policy can
# exceed for a given coverage pattern and time_limit.

def oracle_accuracy(covered, time_limit, n_steps=130000):
    misses, tracked, td, t = 0, False, 0, 0
    T = len(covered)
    for _ in range(n_steps):
        if not tracked:
            misses += 1
            tracked, td = True, 0
        elif covered[t]:
            td = 0
        else:
            td += 1
            if td > time_limit:
                tracked = False
        t = (t + 1) % T
    return 1.0 - misses / n_steps


print("\n(c) Oracle-policy accuracy upper bound vs time_limit "
      "(exact eval-loop state machine, looped sequence):")
configs = [
    ("no-smooth  cam-only thr0.4", 0, False, 0.5),
    ("no-smooth  or_max   thr0.2", 0, True,  0.25),
    ("smooth+/-2 cam-only thr0.4", 2, False, 0.5),
    ("smooth+/-2 or_max   thr0.2", 2, True,  0.25),
    ("smooth+/-3 or_max   thr0.2", 3, True,  0.25),
]
for label, w, use_or, bar in configs:
    pc_s = maxpool_time(P_cam, w)[gps_mask]
    p = np.maximum(pc_s, pa) if use_or else pc_s
    cov = p[t, gt_seq] >= bar
    accs = "  ".join(f"tl={tl}: {oracle_accuracy(cov, tl):.4f}"
                     for tl in (1, 3, 5))
    print(f"  {label}  cov={cov.mean():.4f}  ->  {accs}")
