#!/bin/bash
# Run 608 — phase-2 family #2: or_max fusion, soft-threshold 0.2 (lock bar 0.25).
# RECONSTRUCTED after the fact from experiments/run608/notes.json + RESUME.md
# (the original launch by gnhf iteration 1 saved no cmd.sh); config fields match.
# Result: baseline 93.00%, best 93.70% at iter 34.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 227 --new-run 608 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run608.log 2>&1
