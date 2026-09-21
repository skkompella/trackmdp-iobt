#!/bin/bash
# Run 704 — GENERALIZATION E-cam-2a: frozen-policy eval of run 610 on unseen
# 20250812_091600 with the RELAXED camera filter (--cam-classes vehicle set,
# --cam-min-conf 0.25). diag_camera_audit found the session's object is detected as
# 'truck' (median conf 0.77-0.88); the hardcoded 'car' filter kept only 10.8% of
# GT-bin detections. Relaxed: lock coverage 30.3% -> 52.3%, oracle bound (tl=1)
# 76.90% -> 84.27%. Compare vs run 700 (same eval, 'car' filter): 69.90%.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 610 --new-run 704 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --cam-classes car,truck,bus,train,boat,airplane,motorcycle \
    --cam-min-conf 0.25 \
    --iobt-gt-session 20250812_091600 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 1 \
    > /tmp/track_mdp_logs/run704.log 2>&1
