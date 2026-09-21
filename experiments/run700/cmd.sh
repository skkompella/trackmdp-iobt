#!/bin/bash
# Run 700 — GENERALIZATION E1a: frozen-policy eval of run 610 (tl=1 accuracy champion)
# on unseen session 20250812_091600 (repaired YOLO for nodes 2/4/7 applied 2026-08-28).
# --iterations 1: the Baseline accuracy line IS the result (frozen policy); the single
# PPO iter is discarded. Analytic oracle bound for this config on this session: 76.90%
# (diag_unseen_coverage.py, smooth+/-2 or_max thr0.2, tl=1). Old runs 237/240 scored
# 68.5-70.7% with the pre-phase-2 recipe and broken YOLO data.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 610 --new-run 700 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --iobt-gt-session 20250812_091600 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 1 \
    > /tmp/track_mdp_logs/run700.log 2>&1
