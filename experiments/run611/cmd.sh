#!/bin/bash
# Run 611 — phase-2 family #1 stage 1: NEW TOPO BASE with time_limit=3.
# Retrains the run-241 recipe (topo prior, from scratch, soft-reward audio fusion,
# 500 iters) with --time-limit 3, because obs/action space depends on
# time_limit_max so the existing run-227/241 bases (time_limit_max=1) are
# INCOMPATIBLE.
# NOTE: this base run's own accuracy numbers are NOT reportable results — the
# base recipe deliberately reproduces run 227/241 (uncalibrated audio, like the
# historical bases); only the subsequent calibrated fine-tune (run 612) counts.
# Smoke-tested first at 10 iters (run 998, /tmp/track_mdp_logs/run998_smoke_tl3.log).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --scratch --new-run 611 \
    --soft-reward --fusion-mode audio \
    --iobt-prior topo \
    --time-limit 3 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --gps-eval --iterations 500 \
    > /tmp/track_mdp_logs/run611.log 2>&1
