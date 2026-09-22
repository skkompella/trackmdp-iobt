#!/bin/bash
# Run 814 — ONLINE TABULAR TD (ORACLE RESTART — control for run 810).
# Restarts at the KNOWN switch points instead of using the detector, so the
# detector's error rate is removed entirely. This separates two explanations for
# run 810's poor showing (47.51% vs the 71.62% no-detector baseline):
#   (a) restarting itself destroys knowledge shared across loops, or
#   (b) the GLR detector simply over-fires (810 logged 60 false alarms in 107).
# If the oracle arm still trails the no-detector baseline, the answer is (a) and
# no amount of detector tuning rescues restart-on-change for this problem.
#
# Same regime as 810-813: max_sensors=2, tl=1, grid preset, cold restart.
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/train_multiloop_online.py \
    --num-loops 10 --loop-seed 20260921 \
    --new-run 814 \
    --time-limit 1 --max-sensors 2 --reward-preset grid \
    --episodes-per-loop 50 --passes 5 \
    --alpha 0.1 --gamma 0.95 --lam 0.8 --epsilon 0.1 --optimistic-init 1.0 \
    --oracle-restart --restart-strategy cold \
    > /tmp/track_mdp_logs/run814.log 2>&1
