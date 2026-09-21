#!/bin/bash
# Run 619 — PHASE 3: the TRUE NON-FRUGAL arm. sensor_rew=-0.01 (near-zero
# energy penalty) at the HONEST threshold 0.2, cam-smooth 4, run-611 tl=3 base.
# Runs 612-618 all self-throttled to ~1.4-2.1 sensors/step because even -0.10
# per sensor outweighs the marginal soft-detection reward; run 618 proved the
# residual gap to 97% is policy-side (GT node missing from the tiny activated
# set on a few tracked steps), not coverage-side (oracle 100% at this stack).
# Near-zero penalty lets accuracy pressure dominate: the eval metric only
# needs the GT node IN the activated set while tracked, so wide activation
# converts oracle headroom directly into accuracy. Projection: 97-98% at
# 4-6 sensors/step — the user's stated non-frugal target band.
# No honesty caveat: threshold stays 0.2 (bar 0.25, above audio FP levels).
# or_max => --calibrators MANDATORY. Fine-tune from 611 base, never a champion.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 619 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 4 \
    --time-limit 3 \
    --sensor-rew -0.01 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run619.log 2>&1
