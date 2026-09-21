#!/bin/bash
# Run 613 — PHASE 3 Track A1: sensor-penalty rebalance, accuracy end.
# The run-612 diagnosis: tl=3 removed missing-state churn, but the hardcoded
# sensor_rew=-0.25 then dominated the reward, so PPO spent the slack on energy
# (1.52 sensors/step, 94.20%) instead of accuracy. This run weakens the penalty
# to -0.10 (new --sensor-rew flag) on the identical run-612 recipe, projecting
# sensors/step back up toward 3-4 and accuracy into the 96-98% band
# (oracle bound for this coverage at tl=3: 98.46%).
# sensor_rew is reward-only (no obs/action space change) => restoring from the
# run-611 tl=3 base is safe. or_max => --calibrators MANDATORY (run 606 trap).
# Never fine-tune from a champion (610/612) — always the 611 base (run-234 lesson).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 613 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --time-limit 3 \
    --sensor-rew -0.10 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run613.log 2>&1
