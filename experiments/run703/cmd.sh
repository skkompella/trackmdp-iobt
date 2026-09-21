#!/bin/bash
# Run 703 — GENERALIZATION E5b: ADAPTATION CONTROL — fine-tune the tl=1 base (run 227)
# on unseen session 20250812_091600 for 100 iterations with the full phase-2 recipe.
# Mirrors historical run 237 (which scored 70.7% with the old recipe + broken YOLO) but
# with or_max@0.2 + calibrators + cam-smooth + repaired data. Vs run 702 this isolates
# the tl=3 contribution to adaptation. Oracle bound at tl=1: 76.90%.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 227 --new-run 703 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --iobt-gt-session 20250812_091600 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 100 \
    > /tmp/track_mdp_logs/run703.log 2>&1
