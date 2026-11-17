#!/usr/bin/env bash
# General base, 5x5 (run 1520): same settings as run 1510 (10x10) - equal-probability Moore walk
# (stay + 8 neighbours, exit 0.005), no episode cap, port PPO settings, k=9, 1000 iterations,
# eval every 10 iterations over 100 episodes, unseeded.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../../track_mdp_env/bin/python ../../../examples/training_grid_rect.py --run 1520 \
  --nrows 5 --ncols 5 --iterations 1000 --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --movement moore > run1520.log 2>&1 < /dev/null &
echo "launched run 1520 (pid $!)"
