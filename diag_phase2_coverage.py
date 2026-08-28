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
