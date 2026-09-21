#!/bin/bash
# Run 620 — PHASE 3 ceiling-seek: sensor_rew = 0.0 (NO energy penalty at all),
# honest threshold 0.2, cam-smooth 4, run-611 tl=3 base.
# Run 619 reached 97.00% @ 2.84 sensors with sensor_rew=-0.01. The analytic
# oracle bound for this coverage stack (cam-smooth-4, or_max@0.2, tl=3) is
# 100.00% (experiments/diag_phase3_precheck.txt), so the 97%->ceiling gap is
# PURELY POLICY-SIDE: even -0.01 slightly discourages activating the GT node on
# a few tracked steps. Removing the penalty entirely lets PPO maximize pure
# tracking accuracy — expected to widen activation past 2.84 sensors and push
# accuracy toward the 100% oracle (target: beat 97.00%).
# Honest: threshold stays 0.2 (bar 0.25), no audio-FP caveat (unlike run 618).
# or_max => --calibrators MANDATORY. Fine-tune from the 611 base, never a champion.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 620 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 4 \
    --time-limit 3 \
    --sensor-rew 0.0 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run620.log 2>&1
