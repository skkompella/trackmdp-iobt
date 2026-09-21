#!/bin/bash
# Run 618 — PHASE 3 final stack: threshold 0.15 on top of run 617's config
# (cam-smooth 4 + sensor-rew -0.10, run-611 tl=3 base).
# Run 617 hit 96.25% @ 1.58 (all-time record); remaining gap to 97% ~ 1 step.
# Analytic (verified post-617): smooth+/-4 + thr 0.15 (lock bar 0.1875) gives
# coverage 127/130 = 97.69%, residual gaps [2,1], oracle tl=3 = 100%.
# CAVEAT (documented in experiments/diag_phase3_precheck.txt): at bar 0.1875
# the calibrated-audio FP rate at non-GT nodes rises 25.7% -> 65.0%. The eval
# metric is unaffected (state machine only consults the GT node), but the
# soft-confirm approximation is less deployment-honest — this run's number
# must carry that caveat wherever it is reported.
# or_max => --calibrators MANDATORY. Fine-tune from 611 base, never a champion.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 618 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.15 \
    --cam-smooth-bins 4 \
    --time-limit 3 \
    --sensor-rew -0.10 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run618.log 2>&1
