#!/bin/bash
# Run 609 — phase-2 family #3: P_cam temporal smoothing (±2 bins = ±1.0s max-pool)
# Launched 2026-08-27 23:22 by gnhf iteration 3. Analytic GT-node lock coverage
# 73.85% -> 83.85% (diag_phase2_coverage.txt, cam-only thr0.4 row, +/-2 bins).
# Camera-only fusion => --calibrators correctly NOT required.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 227 --new-run 609 \
    --soft-reward --fusion-mode camera \
    --soft-scale 0.8 --soft-threshold 0.4 \
    --cam-smooth-bins 2 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run609.log 2>&1
