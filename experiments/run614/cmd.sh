#!/bin/bash
# Run 614 — PHASE 3 Track A2: sensor-penalty rebalance, weakest penalty (-0.05).
# Run 613 (-0.10) recovered accuracy to 94.85% but activation only rose
# 1.49 -> 1.71 sensors/step — PPO is still deep in the frugal basin. This run
# halves the penalty again to test whether -0.05 finally trades energy back
# toward 3-4 sensors and pushes accuracy into the 96-98% band (oracle 98.46%).
# Per run 613's findings: expect a modest rise; if this also under-activates,
# the next lever is coverage (cam-smooth 4, run 617) or max_sensors 8 (run 616),
# not further penalty softening.
# sensor_rew is reward-only (no obs/action space change) => restoring from the
# run-611 tl=3 base is safe. or_max => --calibrators MANDATORY (run 606 trap).
# Never fine-tune from a champion (610/612/613) — always the 611 base (run-234 lesson).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 614 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --time-limit 3 \
    --sensor-rew -0.05 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run614.log 2>&1
