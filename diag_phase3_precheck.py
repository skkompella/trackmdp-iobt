"""Phase-3 analytic pre-checks (read-only, no training) — Tracks C and D.

Session 20250812_165739, 130-step eval sequence. Baseline = the run-610/612
champion stack: or_max fusion, calibrated audio, soft-threshold 0.2 (lock bar
0.25 at scale 0.8), P_cam max-pooled +/-2 bins. Analytic lock coverage 90.00%
(117/130), residual gaps [10, 8, 3].

Decision rule (plan): a config earns a training run only if it projects a
>= 2pp lock-coverage gain over the baseline stack (i.e. >= 120/130).

  (C) camera smoothing re-tune: wider symmetric windows and forward-only
      carry-forward holds, per residual gap.
  (D-thr) lock-bar depth: thresholds 0.15/0.10 — how deep does calibrated
      audio reach on the residual uncovered steps?
  (D1) recalibration potential: per-node AUC of the RAW pooled score at the
      GT node across eval steps. Monotone recalibration preserves ranking, so
      AUC ~0.5 on the uncovered steps == no honest recalibration can help.
  (D2) multiclass audio prior: LGBM 10-class port (diag_audio_multiclass
      features, cached) as an extra or_max channel; LOSO (train excl. this
      session) and in-domain variants.

Run from the worktree root: track_mdp_env/bin/python diag_phase3_precheck.py
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from pathlib import Path
import joblib

from collection.train_node_classifiers_10 import build_p_cam_10, _parse_session_start_10
from examples.finetune_deterministic import build_iobt_gt_sequence, build_p_audio_iobt_10


def maxpool_time(P, w):
    """Max-pool each node's time series over a +/-w bin window
    (same as diag_phase2_coverage.py — not imported: that module runs at import)."""
    if w == 0:
        return P
    out = P.copy()
    for s in range(1, w + 1):
        out[s:] = np.maximum(out[s:], P[:-s])
        out[:-s] = np.maximum(out[:-s], P[s:])
    return out


def oracle_accuracy(covered, time_limit, n_steps=130000):
    """Exact eval-loop state machine (see diag_phase2_coverage.py section (c))."""
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

SESSION = "20250812_165739"
SDIR    = os.path.join(os.getcwd(), "iobt_data_10", SESSION)
SCALE   = 0.8

gt_seq, gps_mask = build_iobt_gt_sequence(SDIR, step_s=0.5)
T_full = len(gps_mask)
t = np.arange(len(gt_seq))

calib   = joblib.load("results/pooled/pooled_calibrators_20260521_164149.pkl")
P_audio = build_p_audio_iobt_10("results/pooled/pooled_clf_20260521_164149.pkl",
                                SDIR, list(range(1, 11)),
                                calibrators=calib, step_s=0.5)
P_audio_raw = build_p_audio_iobt_10("results/pooled/pooled_clf_20260521_164149.pkl",
                                    SDIR, list(range(1, 11)),
                                    calibrators=None, step_s=0.5)
start_unix = _parse_session_start_10(SESSION).timestamp()
P_cam = build_p_cam_10(Path(SDIR), list(range(1, 11)), start_unix, T_full, bin_size_s=0.5)

pa     = P_audio[gps_mask]
pa_raw = P_audio_raw[gps_mask]


def carry_forward(P, w):
    """Forward-only hold: each node's score is the max over the last w bins."""
    if w == 0:
        return P
    out = P.copy()
    for s in range(1, w + 1):
        out[s:] = np.maximum(out[s:], P[:-s])
    return out


def gaps_of(covered):
    gaps, run = [], 0
    for c in covered:
        if not c: run += 1
        elif run: gaps.append(run); run = 0
    if run: gaps.append(run)
    return sorted(gaps, reverse=True)


def report(label, cov):
    n = int(cov.sum())
    acc3 = oracle_accuracy(cov, 3)
    print(f"  {label:42s} cov={cov.mean():.4f} ({n}/130)  "
          f"oracle tl=3={acc3:.4f}  gaps={gaps_of(cov)}")
    return n


# Baseline champion stack
pc2 = maxpool_time(P_cam, 2)[gps_mask]
base_gtv = np.maximum(pc2, pa)[t, gt_seq]
base_cov = base_gtv >= 0.25
print("BASELINE (or_max@0.2 + smooth +/-2):")
base_n = report("champion stack", base_cov)
uncov = np.where(~base_cov)[0]
print(f"  uncovered step indices: {uncov.tolist()}")

print("\n(C) camera smoothing variants (all with or_max@0.2, calibrated audio):")
for w in (3, 4, 5):
    pcs = maxpool_time(P_cam, w)[gps_mask]
    report(f"symmetric +/-{w} bins ({w*0.5:.1f}s)",
           np.maximum(pcs, pa)[t, gt_seq] >= 0.25)
for w in (2, 4, 6, 8):
    pcs = carry_forward(P_cam, w)[gps_mask]
    report(f"forward-only hold {w} bins ({w*0.5:.1f}s)",
           np.maximum(pcs, pa)[t, gt_seq] >= 0.25)
for w in (4, 6, 8):
    # hybrid: symmetric +/-2 plus forward-only extension to w
    pcs = np.maximum(maxpool_time(P_cam, 2), carry_forward(P_cam, w))[gps_mask]
    report(f"+/-2 sym + forward hold {w} bins",
           np.maximum(pcs, pa)[t, gt_seq] >= 0.25)

print("\n(D-thr) lock-bar depth on the champion stack (bar = thr/0.8):")
for thr in (0.20, 0.15, 0.10, 0.05):
    report(f"threshold {thr:.2f} (bar {thr/SCALE:.4f})", base_gtv >= thr / SCALE)
au = pa[t, gt_seq][uncov]
print(f"  calibrated audio at GT node on the {len(uncov)} uncovered steps: "
      f"min={au.min():.3f} median={np.median(au):.3f} max={au.max():.3f}")
print(f"  counts >= bar: 0.25:{(au>=0.25).sum()}  0.1875:{(au>=0.1875).sum()}  "
      f"0.125:{(au>=0.125).sum()}  0.0625:{(au>=0.0625).sum()}")
# Deployment-honesty stat: in a real system any activated node with audio >= bar
# would claim a detection. The eval metric can't see this (state machine only
# consults the GT node), so record the audio false-positive rate per bar.
gt_onehot_mask = np.zeros_like(pa, dtype=bool)
gt_onehot_mask[t, gt_seq] = True
non_gt = pa[~gt_onehot_mask]
print(f"  audio FP rate at NON-GT nodes (deployment honesty, not in eval "
      f"metric): >=0.25: {(non_gt>=0.25).mean():.3f}  "
      f">=0.1875: {(non_gt>=0.1875).mean():.3f}  "
      f">=0.125: {(non_gt>=0.125).mean():.3f}  "
      f">=0.0625: {(non_gt>=0.0625).mean():.3f}")

# Smoothing legitimacy: how long does the object dwell at a node? A carry of w
# bins is mostly legitimate if dwell times are >= w (the carried detection was
# taken while the object really was at that node).
dwells, run = [], 1
for i in range(1, len(gt_seq)):
    if gt_seq[i] == gt_seq[i - 1]:
        run += 1
    else:
        dwells.append(run); run = 1
dwells.append(run)
dw = np.array(dwells)
print(f"  GT dwell times (bins): median={np.median(dw):.0f} "
      f"mean={dw.mean():.1f} min={dw.min()} max={dw.max()}  "
      f"share >=4: {(dw>=4).mean():.2f}  >=8: {(dw>=8).mean():.2f}")

print("\n(D1) recalibration potential — RAW pooled score discrimination:")
gt_onehot = np.zeros_like(pa_raw, dtype=bool)
gt_onehot[t, gt_seq] = True
aucs = []
for k in range(10):
    pos = pa_raw[gt_onehot[:, k], k]
    neg = pa_raw[~gt_onehot[:, k], k]
    if len(pos) == 0:
        continue
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    a = (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    aucs.append((k + 1, len(pos), a))
print("  node  gt_steps  AUC(raw score, GT vs non-GT steps)")
for nid, n, a in aucs:
    print(f"  {nid:>4} {n:>9} {a:>8.3f}")
raw_u = pa_raw[t, gt_seq][uncov]
raw_med = np.median(pa_raw, axis=0)
above_med = sum(raw_u[i] > raw_med[gt_seq[uncov[i]]] for i in range(len(uncov)))
print(f"  uncovered steps whose GT-node RAW score exceeds that node's median: "
      f"{above_med}/{len(uncov)}")
print("  (monotone recalibration can only rescue steps that rank high vs the "
      "node's own distribution)")

print("\n(D2) multiclass audio prior as extra or_max channel:")
from diag_audio_multiclass import build_session_xy, SESSIONS
import lightgbm as lgb
data = {}
for s in SESSIONS:
    X, y = build_session_xy(s)
    if X is not None:
        data[s] = (X, y)
X_here, y_here = data[SESSION]
if len(y_here) == T_full:
    Xe, ye = X_here[gps_mask], y_here[gps_mask]
elif len(y_here) == len(gt_seq):
    # feature rows are already the GPS-covered steps
    Xe, ye = X_here, y_here
else:
    raise SystemExit(f"cannot align feature rows {len(y_here)} to "
                     f"T_full {T_full} / eval {len(gt_seq)}")
lab = ye >= 0
agree = (ye[lab] == gt_seq[lab]).mean() if lab.any() else float("nan")
print(f"  label alignment check: {lab.sum()}/{len(gt_seq)} labeled, "
      f"agreement with gt_seq = {agree:.3f}")
for mode in ("LOSO", "in-domain"):
    train_ss = [s for s in data if s != SESSION] if mode == "LOSO" else list(data)
    Xtr = np.vstack([data[s][0] for s in train_ss])
    ytr = np.concatenate([data[s][1] for s in train_ss])
    keep = ytr >= 0
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=10,
                             n_estimators=300, num_leaves=63,
                             learning_rate=0.05, verbose=-1)
    clf.fit(Xtr[keep], ytr[keep])
    P_mc = clf.predict_proba(Xe)  # (130, 10)
    mc_gtv = P_mc[t, gt_seq]
    print(f"  [{mode}] P_mc at GT node: median={np.median(mc_gtv):.3f} "
          f"max={mc_gtv.max():.3f}  top-1 acc={(P_mc.argmax(1)==gt_seq).mean():.3f}")
    fused = np.maximum(base_gtv, mc_gtv)
    report(f"[{mode}] champion + P_mc or_max @0.2", fused >= 0.25)
    rescued = (mc_gtv[uncov] >= 0.25).sum()
    print(f"    rescues {rescued}/{len(uncov)} baseline-uncovered steps at bar 0.25")
