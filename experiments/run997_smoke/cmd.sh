#!/bin/bash
# Run 997 — THROWAWAY smoke test for the new --sensor-rew flag (phase 3).
# 5 iterations from the run-611 tl=3 base with sensor_rew=-0.10; verifies:
#   (1) [sensor-rew] banner prints and RealIoBTEnv gets the override,
#   (2) checkpoint restore works with the flag present (obs/action space
#       unchanged — sensor_rew is reward-only),
#   (3) training + eval loop completes.
# NOT a reportable result. Checkpoint runs/agent_run997_ppo is disposable.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 997 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --time-limit 3 \
    --sensor-rew -0.10 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 5 \
    > /tmp/track_mdp_logs/run997_smoke.log 2>&1
