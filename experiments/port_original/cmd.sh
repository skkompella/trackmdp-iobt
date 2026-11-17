#!/usr/bin/env bash
# RLlib port with original Track-MDP settings: 10x10, 2005 iterations, eval every 10 (100 episodes),
# no training episode cap (episodes end on target exit), original 3x3 random movement.
# Kept from the port: its metrics, k=9, lr 1e-4, 4 workers, unseeded.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1500 \
  --nrows 10 --ncols 10 --iterations 2005 --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --movement original > run1500.log 2>&1 < /dev/null &
echo "launched run 1500 (pid $!)"
