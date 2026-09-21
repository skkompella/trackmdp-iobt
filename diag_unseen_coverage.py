"""T0a: analytic coverage decomposition for UNSEEN sessions (read-only, no training).

Generalizes diag_phase2_coverage.py to any session. For the given session it reports:
  (0) per-node data availability (FLAC / YOLO present, per-node signal stats) and the
      fraction of GT steps spent at nodes with no audio / no camera / neither —
      the irreducible dead-signal floor no classifier work can fix;
  (a) or_max lock coverage vs soft-threshold (bar = threshold / 0.8);
  (b) P_cam max-pool smoothing sweep (cam-only and or_max variants, gap structure);
  (c) oracle-policy accuracy upper bounds vs time_limit (exact eval state machine).

Usage: track_mdp_env/bin/python diag_unseen_coverage.py <session> [<session> ...]
Run from the worktree/checkout root.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from pathlib import Path

from collection.train_node_classifiers_10 import build_p_cam_10, _parse_session_start_10
from examples.finetune_deterministic import build_iobt_gt_sequence, build_p_audio_iobt_10
import joblib

SCALE = 0.8
NODE_IDS = list(range(1, 11))
CLF = "results/pooled/pooled_clf_20260521_164149.pkl"
CALIB = "results/pooled/pooled_calibrators_20260521_164149.pkl"


def maxpool_time(P, w):
    if w == 0:
        return P
    out = P.copy()
    for s in range(1, w + 1):
        out[s:] = np.maximum(out[s:], P[:-s])
        out[:-s] = np.maximum(out[:-s], P[s:])
    return out


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


def gap_lengths(dead):
    gaps, run = [], 0
    for d in dead:
        if d:
            run += 1
        elif run:
            gaps.append(run); run = 0
    if run:
        gaps.append(run)
    return sorted(gaps, reverse=True)


def analyze(session):
    # "SESSION=/path/to/dir" overrides the data dir (e.g. a repaired shadow copy)
    session, _, override = session.partition("=")
    session_dir = override or os.path.join(os.getcwd(), "iobt_data_10", session)
    print("=" * 72)
    print(f"SESSION {session}")
    print("=" * 72)

    gt_seq, gps_mask = build_iobt_gt_sequence(session_dir, step_s=0.5)
    T_full = len(gps_mask)
    n = len(gt_seq)
    print(f"eval steps={n}  (full bins={T_full})  scale={SCALE}")

    calib = joblib.load(CALIB)
    P_audio = build_p_audio_iobt_10(CLF, session_dir, NODE_IDS,
                                    calibrators=calib, step_s=0.5)
    start_unix = _parse_session_start_10(session).timestamp()
    # CAM_CLASSES / CAM_MIN_CONF env vars override the camera filter
    classes = set(os.environ.get("CAM_CLASSES", "car").split(","))
    min_conf = float(os.environ.get("CAM_MIN_CONF", "0.5"))
    print(f"[cam-filter] classes={sorted(classes)} min_conf={min_conf}")
    P_cam = build_p_cam_10(Path(session_dir), NODE_IDS, start_unix, T_full,
                           bin_size_s=0.5, target_class=classes,
                           min_conf=min_conf)

    pa = P_audio[gps_mask]
    pc = P_cam[gps_mask]
    t = np.arange(n)

    # ── (0) data availability + irreducible dead-signal floor ──
    print("\n(0) per-node availability and GT-step exposure:")
    sd = Path(session_dir)
    has_flac = {i: (sd / f"node{i}_respeaker.flac").exists() for i in NODE_IDS}
    has_yolo = {i: (sd / f"node{i}_zed_yolo.json").exists() for i in NODE_IDS}
    gt_counts = np.bincount(gt_seq, minlength=10)
    for i in NODE_IDS:
        k = i - 1
        cam_alive = pc[:, k].max() > 0
        print(f"  node{i:>2}: flac={'Y' if has_flac[i] else '-'} "
              f"yolo={'Y' if has_yolo[i] else '-'} cam_signal={'Y' if cam_alive else '-'}  "
              f"gt_steps={gt_counts[k]:>4}  "
              f"P_aud(gt) max={pa[gt_seq == k, k].max() if gt_counts[k] else 0:.3f}  "
              f"P_cam(gt) max={pc[gt_seq == k, k].max() if gt_counts[k] else 0:.3f}")
    no_audio = np.array([not has_flac[i] for i in NODE_IDS])
    cam_dead_node = np.array([pc[:, i - 1].max() == 0 for i in NODE_IDS])
    at_no_audio = no_audio[gt_seq].mean()
    at_no_cam = cam_dead_node[gt_seq].mean()
    at_neither = (no_audio & cam_dead_node)[gt_seq].mean()
    print(f"  GT-step fraction at nodes with: no FLAC={at_no_audio:.3f}  "
          f"no camera signal={at_no_cam:.3f}  NEITHER={at_neither:.3f} "
          f"(irreducible dead floor)")

    # ── (a) or_max coverage vs threshold ──
    print("\n(a) or_max lock coverage vs soft-threshold (no smoothing):")
    or_gt = np.maximum(pc, pa)[t, gt_seq]
    for thr in (0.4, 0.3, 0.25, 0.2):
        bar = thr / SCALE
        cov = (or_gt >= bar)
        print(f"  threshold={thr:.2f}  bar={bar:.4f}  coverage={cov.mean():.4f} "
              f"({cov.sum()}/{n})")

    # ── (b) smoothing sweep ──
    print("\n(b) P_cam max-pool smoothing (bar = threshold/scale):")
    for w in (0, 1, 2, 3):
        pc_s = maxpool_time(P_cam, w)[gps_mask]
        cam_gt = pc_s[t, gt_seq]
        or_s_gt = np.maximum(pc_s, pa)[t, gt_seq]
        print(f"  +/-{w} bins: cam-only thr0.4 cov={(cam_gt >= 0.5).mean():.4f} "
              f"({(cam_gt >= 0.5).sum()}/{n})  "
              f"or_max thr0.4 cov={(or_s_gt >= 0.5).mean():.4f}  "
              f"or_max thr0.2 cov={(or_s_gt >= 0.25).mean():.4f} "
              f"({(or_s_gt >= 0.25).sum()}/{n})  "
              f"remaining cam gaps: {gap_lengths(cam_gt < 0.5)}")

    # ── (c) oracle bounds ──
    print("\n(c) Oracle-policy accuracy upper bound vs time_limit:")
    configs = [
        ("no-smooth  cam-only thr0.4", 0, False, 0.5),
        ("no-smooth  or_max   thr0.2", 0, True,  0.25),
        ("smooth+/-2 cam-only thr0.4", 2, False, 0.5),
        ("smooth+/-2 or_max   thr0.2", 2, True,  0.25),
    ]
    for label, w, use_or, bar in configs:
        pc_s = maxpool_time(P_cam, w)[gps_mask]
        p = np.maximum(pc_s, pa) if use_or else pc_s
        cov = p[t, gt_seq] >= bar
        accs = "  ".join(f"tl={tl}: {oracle_accuracy(cov, tl):.4f}"
                         for tl in (1, 3, 5))
        print(f"  {label}  cov={cov.mean():.4f}  ->  {accs}")
    print()


if __name__ == "__main__":
    for s in sys.argv[1:] or ["20250812_091600"]:
        analyze(s)
