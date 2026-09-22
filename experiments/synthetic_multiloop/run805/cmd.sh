#!/bin/bash
# Run 805 — LEVER EXPERIMENT (sensor_rew).
# sensor_rew -0.16 -> -0.50. Soft pressure; a 6-sensor detection nets 1.0-3.0 = -2.0, so wide activation is now clearly worse than missing (0.0).
#
# Context: run 241 tracks all 10 loops at the 99.00% measurement ceiling using
# 4.26 sensors, because max successor ambiguity across the loop family is 4 and
# max_sensors is 6 — so hedging over every candidate successor always works and
# no adaptation is ever needed. These runs find which lever pushes activation
# below 4, the point at which a policy must commit and therefore must infer
# which loop it is on.
#
# Baseline shared by 801-807: tl=1 and the grid_env reward preset, i.e. run
# 241's regime. NOT run 800's tl=3 + iobt preset, which confounded the result.
# Serialize these runs: two Ray instances thrash the box (index.md:105).
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/train_multiloop_synthetic.py \
    --num-loops 10 --loop-seed 20260921 \
    --new-run 805 \
    --time-limit 1 \
    --reward-preset grid \
    --max-sensors 6 \
    --sensor-rew -0.50 \
    --max-iterations 600 --eval-interval 5 --patience 30 \
    --eval-episodes-per-loop 5 --final-eval-episodes-per-loop 20 \
    > /tmp/track_mdp_logs/run805.log 2>&1
