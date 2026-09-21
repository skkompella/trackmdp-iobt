#!/bin/bash
# Run 705 — GENERALIZATION E-cam-2b: ADAPTATION with the relaxed camera filter —
# fine-tune the tl=1 base (run 227) on unseen 20250812_091600 for 100 iterations,
# full phase-2 recipe + relaxed camera (see run 704). This is the full "new
# environment + new object + new movement pattern, after some training time"
# scenario with the data pipeline fixed. Oracle bound (tl=1): 84.27%.
# Compare: run 703 (same but 'car' filter), run 704 (frozen, relaxed).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 227 --new-run 705 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --cam-classes car,truck,bus,train,boat,airplane,motorcycle \
    --cam-min-conf 0.25 \
    --iobt-gt-session 20250812_091600 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 100 \
    > /tmp/track_mdp_logs/run705.log 2>&1
