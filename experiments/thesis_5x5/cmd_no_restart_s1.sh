#!/usr/bin/env bash
# Thesis experiment, 5x5, run 1: NO RESTART (standard.md section 7, schedule v1).
# (P0,0) -> (P1,150) -> (P2,300) -> (P3,450); at least 600 iterations, then training
# continues until every episode in progress at iteration 600 has ended (--finish-episodes).
# P0-P3 = O0-O3, original 9-candidate/4-slot scheme, --matrix-seed 1.
# Starts COLD (random initial policy); PPO keeps training through every switch.
# Everything else per standard.md: infinite horizon, port PPO settings, no sensor
# budget, upstream accuracy, eval 100 episodes every 10 iterations + iteration 0 +
# right after each switch + at the end, current/best checkpoints.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1551 \
  --nrows 5 --ncols 5 --iterations 600 --finish-episodes --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --seed 1 --sensor-budget none \
  --matrices O0,O1,O2,O3 --matrix-seed 1 --switch-at 150,300,450 \
  > run1551.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1551 (no restart, cold start)"
