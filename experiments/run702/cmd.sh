#!/bin/bash
# Run 702 — GENERALIZATION E5a: ADAPTATION test — fine-tune the tl=3 GENERIC base
# (run 611) on unseen session 20250812_091600 for 100 iterations. This is the user's
# target scenario: "work well on a new movement pattern in a new environment after some
# training time". Adapting from the generic base, NOT from a session-tuned champion:
# run 240 (adapted from champion 229) underperformed run 237 (adapted from base 227),
# and same-session chaining overfits (run 234). Oracle bound at tl=3: 87.04%.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 702 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --time-limit 3 \
    --iobt-gt-session 20250812_091600 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 100 \
    > /tmp/track_mdp_logs/run702.log 2>&1
