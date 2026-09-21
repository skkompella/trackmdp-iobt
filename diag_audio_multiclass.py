"""E-audio-1: 10-node MULTICLASS audio movement classifier (cross-node joint features).

Motivation (diag_audio_auc.py, 2026-08-28): the per-node-binary pooled classifier has
near-zero localization power (AUC ~0.3-0.66, many below chance; ideal coverage at
FPR<=10% only 8-21%). The old 6-node pipeline's MULTICLASS movement classifier used all
nodes' signals jointly (plus pairwise diffs) and was node-discriminative by construction
— never ported to the 10-node system. This tests that port.

Design:
  - features per GPS step: for each of 10 nodes, per-session z-scored raw (M,10) ->
    causal rolling mean+std (M,20), concatenated -> (M,200); +10 presence-mask cols.
    Missing FLAC -> zeros + mask=0.
  - label: GT node index (from the per-node binary labels; steps whose GT node has no
    FLAC are dropped -> metrics are over audio-capable GT steps, same basis as the
    binary baseline's ideal-coverage numbers).
  - LightGBM multiclass softmax over 10 nodes.
  - LOSO folds: hold out 20250812_091600 (unseen everywhere) and 20250813_154313.
  - metrics: top-1/top-3 accuracy, per-node one-vs-rest AUC of P[:, gt_node], and
    ideal coverage of P_gt at per-node bars with FPR<=10% — directly comparable to
    the binary baseline (091600: 0.211, 154313: 0.081, 165739 in-domain: 0.177).

Usage: track_mdp_env/bin/python diag_audio_multiclass.py
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from pathlib import Path

from collection.train_node_classifiers_10 import (
    aggregate_rolling, load_session_raw_per_node_10, WINDOW_STEPS,
)

DATA = Path("iobt_data_10")
NODE_IDS = list(range(1, 11))
SESSIONS = [
    "20240806_085959", "20240806_094213", "20240806_094548", "20240806_100933",
    "20240806_101417", "20240806_101909", "20240806_102439", "20240806_102918",
    "20250812_091600", "20250812_165739", "20250813_154313",
]
TEST_FOLDS = ["20250812_091600", "20250813_154313"]


CACHE = Path(os.environ.get("MC_CACHE", "/tmp/track_mdp_logs/mc_feat_cache"))
CACHE.mkdir(parents=True, exist_ok=True)


def build_session_xy(session):
    """-> X (M, 210+45), y (M,) node idx 0-9 or -1 if GT node has no audio.

    v2: adds 45 pairwise cross-node differences of each node's rolled amplitude
    (column 0), mirroring the 6-node multiclass classifier's pairwise-diff features.
    Per-session results cached as npz (delete the cache dir after feature changes).
    """
    cpath = CACHE / f"{session}.npz"
    if cpath.exists():
        z = np.load(cpath)
        return z["X"], z["y"]
    try:
        per_node = load_session_raw_per_node_10(DATA / session)
    except Exception as e:
        print(f"  [skip] {session}: {e}")
        return None, None
    M = len(next(iter(per_node.values()))[0])
    feats, mask = [], []
    y = np.full(M, -1, dtype=np.int64)
    for k, nid in enumerate(NODE_IDS):
        if nid in per_node:
            raw, y_bin = per_node[nid]
            mu, sd = raw.mean(0), raw.std(0) + 1e-9   # per-session z-score
            rolled = aggregate_rolling((raw - mu) / sd, WINDOW_STEPS)  # (M,20)
            feats.append(rolled)
            mask.append(np.ones((M, 1)))
            y[y_bin == 1] = k
        else:
            feats.append(np.zeros((M, 20)))
            mask.append(np.zeros((M, 1)))
    amp = np.stack([feats[k][:, 0] for k in range(len(NODE_IDS))], axis=1)  # (M,10)
    diffs = [amp[:, i] - amp[:, j]
             for i in range(len(NODE_IDS)) for j in range(i + 1, len(NODE_IDS))]
    X = np.hstack(feats + mask + [np.stack(diffs, axis=1)])
    np.savez_compressed(CACHE / f"{session}.npz", X=X, y=y)
    return X, y


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    import lightgbm as lgb
    data = {}
    for s in SESSIONS:
        print(f"[load] {s}", flush=True)
        X, y = build_session_xy(s)
        if X is not None:
            data[s] = (X, y)

    for test in TEST_FOLDS:
        if test not in data:
            continue
        train_ss = [s for s in data if s != test]
        Xtr = np.vstack([data[s][0] for s in train_ss])
        ytr = np.concatenate([data[s][1] for s in train_ss])
        keep = ytr >= 0
        Xtr, ytr = Xtr[keep], ytr[keep]
        Xte, yte = data[test]
        keep_te = yte >= 0
        Xte, yte = Xte[keep_te], yte[keep_te]

        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=10, n_estimators=300,
            num_leaves=63, learning_rate=0.05, verbose=-1,
        )
        clf.fit(Xtr, ytr)
        P = clf.predict_proba(Xte)  # (M,10)

        top1 = (P.argmax(1) == yte).mean()
        order = np.argsort(-P, axis=1)[:, :3]
        top3 = np.mean([yte[i] in order[i] for i in range(len(yte))])

        print("=" * 72)
        print(f"TEST FOLD {test}  (train: {len(train_ss)} sessions, "
              f"{len(ytr)} steps; test: {len(yte)} audio-capable GT steps)")
        print(f"  top-1 accuracy: {top1:.3f}   top-3: {top3:.3f}   "
              f"(chance top-1 ~ 0.1)")
        covs = []
        print(f"  {'node':>5} {'gt_n':>5} {'AUC':>7} {'cov@fpr10%':>11} {'bar':>6}")
        for k, nid in enumerate(NODE_IDS):
            at = yte == k
            if at.sum() == 0:
                continue
            pos, neg = P[at, k], P[~at, k]
            bar = np.quantile(neg, 0.9)
            cov = (pos > bar).mean()
            covs.append((at.sum(), cov))
            print(f"  {nid:>5} {at.sum():>5} {auc(pos, neg):>7.3f} {cov:>11.3f} {bar:>6.3f}")
        tot = sum(n for n, _ in covs)
        w = sum(n * c for n, c in covs) / tot if tot else float("nan")
        base = {"20250812_091600": 0.211, "20250813_154313": 0.081}.get(test)
        print(f"  weighted ideal audio coverage: {w:.3f}  "
              f"(binary-pooled baseline: {base})")


if __name__ == "__main__":
    main()
