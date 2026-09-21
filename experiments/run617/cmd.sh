#!/bin/bash
# Run 617 — PHASE 3 Track C: wider camera smoothing (+/-4 bins) on the Track A
# winner (sensor_rew=-0.10, run 613's config).
# The penalty sweep settled non-monotonically (612: -0.25 -> 94.20% @ 1.52;
# 613: -0.10 -> 94.85% @ 1.71; 614: -0.05 -> 94.05% @ 2.05), so the accuracy
# route to 97% runs through LOCK COVERAGE, not reward shaping.
# Analytic pre-check (experiments/diag_phase3_precheck.txt): +/-4-bin (2.0s)
# max-pool of P_cam raises or_max@0.2 lock coverage 90.00% -> 94.62% (123/130,
# residual gaps [2,2,2,1]) and the oracle bound at tl=3 to 100%. Legitimacy:
# GT dwell median 10 bins, 83% of dwells >= 4 bins — a 2s carry rarely crosses
# a node transition.
# Identical otherwise to run 613: run-611 tl=3 base, or_max + calibrators
# (MANDATORY), scale 0.8, threshold 0.2, 200 iters, --gps-eval.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 617 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 4 \
    --time-limit 3 \
    --sensor-rew -0.10 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run617.log 2>&1
