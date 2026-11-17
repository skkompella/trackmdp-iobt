#!/usr/bin/env bash
# Robustness of the converged PPO Track-MDP policy to ONE change of movement model.
# Branch run 1530 (5x5, movement seed 1, no sensor budget; checkpoint at iteration 470,
# 99.5% over its last 10 evals) onto a NEW random 9-candidate/4-slot model (seed 2).
# Everything else per standard.md and identical to 1530.
#   1540  from 1530's checkpoint (weights + optimizer), evaluated at iteration 0 before training
#   1541  control: from scratch on the same new model (how much the old policy helps)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
B=$(cd ../budget_test && pwd)/agent_run1530_ppo/current
COMMON="--nrows 5 --ncols 5 --iterations 500 --eval-interval 10 --eval-episodes 100
  --max-ep-steps 0 --movement original --seed 2 --max-sensors 6 --sensor-budget none"
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1540 $COMMON \
  --init-checkpoint $B > run1540.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1540 (from 1530)"; sleep 30
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1541 $COMMON \
  > run1541.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1541 (from scratch)"
