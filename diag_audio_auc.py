"""T-audio-0: is the audio problem calibration or classification?

For each session, builds the pooled classifier's RAW (uncalibrated) and CALIBRATED
P_audio and computes, per node with audio data:
  - ROC-AUC of the score separating "object at this node" vs "object elsewhere"
    (rank-based; needs no threshold). AUC >> 0.5 => the information exists and the
    fix is calibration/thresholding (cheap). AUC ~ 0.5 => the classifier itself is
    uninformative there (needs data/features).
  - score distributions at GT vs non-GT steps, raw and calibrated.
  - best achievable GT-step coverage at ANY per-node bar with false-positive rate
    <= 10% at that node (what an ideal per-node recalibration could unlock).

Usage: track_mdp_env/bin/python diag_audio_auc.py [session ...]
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import joblib

from examples.finetune_deterministic import build_iobt_gt_sequence, build_p_audio_iobt_10

CLF = "results/pooled/pooled_clf_20260521_164149.pkl"
CALIB = "results/pooled/pooled_calibrators_20260521_164149.pkl"
NODE_IDS = list(range(1, 11))


def auc(pos, neg):
    """Rank-based ROC-AUC (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_cov_at_fpr(pos, neg, max_fpr=0.10):
    """Max GT coverage achievable with per-node bar keeping FPR <= max_fpr."""
    if len(pos) == 0:
        return float("nan"), float("nan")
    bar = np.quantile(neg, 1 - max_fpr) if len(neg) else 0.0
    return (pos > bar).mean(), bar


def analyze(session):
    sd = os.path.join(os.getcwd(), "iobt_data_10", session)
    gt_seq, gps_mask = build_iobt_gt_sequence(sd, step_s=0.5)
    calib = joblib.load(CALIB)
    P_raw = build_p_audio_iobt_10(CLF, sd, NODE_IDS, calibrators=None, step_s=0.5)[gps_mask]
    P_cal = build_p_audio_iobt_10(CLF, sd, NODE_IDS, calibrators=calib, step_s=0.5)[gps_mask]

    from pathlib import Path
    print("=" * 78)
    print(f"SESSION {session}  ({len(gt_seq)} GT steps)")
    print(f"{'node':>5} {'gt_n':>5} {'AUC_raw':>8} {'AUC_cal':>8} "
          f"{'raw@GT p50/p90':>15} {'raw@~GT p50/p90':>16} {'cov@fpr10%':>10} {'bar':>6}")
    for k, nid in enumerate(NODE_IDS):
        if not (Path(sd) / f"node{nid}_respeaker.flac").exists():
            print(f"{nid:>5}  (no FLAC)")
            continue
        at = gt_seq == k
        pos, neg = P_raw[at, k], P_raw[~at, k]
        posc = P_cal[at, k]
        cov, bar = best_cov_at_fpr(pos, neg)
        print(f"{nid:>5} {at.sum():>5} {auc(pos, neg):>8.3f} "
              f"{auc(posc, P_cal[~at, k]):>8.3f} "
              f"{np.median(pos) if len(pos) else float('nan'):>7.3f}/"
              f"{np.percentile(pos, 90) if len(pos) else float('nan'):<6.3f} "
              f"{np.median(neg):>8.3f}/{np.percentile(neg, 90):<6.3f} "
              f"{cov:>10.3f} {bar:>6.3f}")
    # overall: what fraction of GT steps could ideal per-node bars cover?
    covs = []
    for k, nid in enumerate(NODE_IDS):
        if not (Path(sd) / f"node{nid}_respeaker.flac").exists():
            continue
        at = gt_seq == k
        if at.sum() == 0:
            continue
        cov, _ = best_cov_at_fpr(P_raw[at, k], P_raw[~at, k])
        covs.append((at.sum(), cov))
    tot = sum(n for n, _ in covs)
    w = sum(n * c for n, c in covs) / tot if tot else float("nan")
    print(f"  weighted ideal audio coverage over audio-capable GT steps: {w:.3f} "
          f"({tot}/{len(gt_seq)} steps have audio at GT)")
    print()


if __name__ == "__main__":
    for s in sys.argv[1:] or ["20250812_165739", "20250812_091600", "20250813_154313"]:
        analyze(s)
