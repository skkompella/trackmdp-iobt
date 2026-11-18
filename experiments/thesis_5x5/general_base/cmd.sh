#!/usr/bin/env bash
# 5x5 general base under standard.md (replaces run 1520): equal-probability Moore walk
# (every in-bounds 3x3 candidate equally likely, exit 0.005 -- the expected 4-slot model),
# seed 1, port PPO, no sensor budget, upstream metric, eval 100 episodes every 10 iterations.
# Length: max(300 iterations, 40 iterations after accuracy has plateaued), then finish open
# episodes. Plateau = no rise >= 0.5 points in the rolling 3-evaluation mean. Cap 2000.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../../track_mdp_env/bin/python ../../../examples/training_grid_rect.py --run 1560 \
  --nrows 5 --ncols 5 --iterations 300 --plateau-patience 40 --plateau-min-delta 0.005 \
  --max-iterations 2000 --finish-episodes --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --movement moore --seed 1 --sensor-budget none > run1560.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1560 (5x5 general base, plateau rule)"
