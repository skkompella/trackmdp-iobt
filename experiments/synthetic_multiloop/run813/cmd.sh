#!/bin/bash
# Run 813 — ONLINE TABULAR TD (library restart).
# Archive the finished segment's table, restore the best-scoring archive. Heuristic selection, not matched-expert — documented in change_detection.py.
#
# Regime chosen by the lever experiments (801-807): max_sensors=2 sits below the
# loop family's maximum successor ambiguity of 4, so the policy cannot hedge over
# every candidate successor and must infer which loop is active. tl=1 + grid
# preset throughout, matching run 241's configuration.
#
# References in this regime:
#   PPO on all 10 loops (run 803)        31.53% @ 1.75 sensors
#   pooled tabular prior (stationary)    70.35% @ 2.00 sensors
#   single-loop tabular (lower bound)    80.23%
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/train_multiloop_online.py \
    --num-loops 10 --loop-seed 20260921 \
    --new-run 813 \
    --time-limit 1 --max-sensors 2 --reward-preset grid \
    --episodes-per-loop 50 --passes 5 \
    --alpha 0.1 --gamma 0.95 --lam 0.8 --epsilon 0.1 --optimistic-init 1.0 \
    --restart-strategy library \
    > /tmp/track_mdp_logs/run813.log 2>&1
