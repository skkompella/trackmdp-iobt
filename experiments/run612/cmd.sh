#!/bin/bash
# Run 612 — phase-2 family #1+#4: the FULL COMBINATION with time_limit=3.
#   time_limit=3 (tracker tolerates 3 consecutive failed confirmations)
#   + or_max fusion + soft-threshold 0.2 (lock bar 0.25, audio rescues camera gaps)
#   + --cam-smooth-bins 2 (±1.0s max-pool of P_cam before fusion)
# Fine-tunes from run 611 (the tl=3 topo base — run 227/241 are INCOMPATIBLE,
# their obs/action space assumes time_limit_max=1).
# Oracle-policy upper bound for this config: 98.46% (diag_phase2_coverage.py
# section (c): coverage 90.00%, tl=3) vs 96.92% for run 610 (same coverage, tl=1).
# or_max touches P_audio => --calibrators is MANDATORY (see run 606 trap).
# LAUNCH ONLY AFTER run 611 completes (checkpoint runs/agent_run611_ppo).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p /tmp/track_mdp_logs
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --env iobt --run 611 --new-run 612 \
    --soft-reward --fusion-mode or_max \
    --soft-scale 0.8 --soft-threshold 0.2 \
    --cam-smooth-bins 2 \
    --time-limit 3 \
    --iobt-gt-session 20250812_165739 \
    --pooled-clf results/pooled/pooled_clf_20260521_164149.pkl \
    --calibrators results/pooled/pooled_calibrators_20260521_164149.pkl \
    --gps-eval --iterations 200 \
    > /tmp/track_mdp_logs/run612.log 2>&1
