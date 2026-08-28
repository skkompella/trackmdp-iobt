#!/bin/bash
# Run 610 — phase-2 family #4: COMBINATION of the family-#2 and family-#3 winners.
#   or_max fusion + soft-threshold 0.2 (lock bar 0.25, audio can rescue camera gaps)
#   + --cam-smooth-bins 2 (±1.0s max-pool of P_cam before fusion)
# Analytic GT-node lock coverage: 90.00% (117/130), residual gaps [10, 8, 3]
# (diag_phase2_coverage.txt, +/-2 bins row, or_max thr0.2 column).
# or_max touches P_audio => --calibrators is MANDATORY (see run 606 trap).
# Base run 227 kept for comparability with runs 608/609.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 227 --new-run 610 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run610.log 2>&1
