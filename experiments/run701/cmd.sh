#!/bin/bash
# Run 701 — GENERALIZATION E1b: frozen-policy eval of run 612 (tl=3 efficiency champion)
# on unseen session 20250812_091600 (repaired YOLO). Tests the prediction that tl=3 is
# the dominant unseen-session lever: oracle bound 87.04% at tl=3 vs 76.90% at tl=1 for
# the same signal (diag_unseen_coverage.py). --time-limit 3 REQUIRED to match the
# checkpoint's obs/action space.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 612 --new-run 701 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --time-limit 3 \
    --iobt-gt-session 20250812_091600 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 1 \
    > /tmp/track_mdp_logs/run701.log 2>&1
