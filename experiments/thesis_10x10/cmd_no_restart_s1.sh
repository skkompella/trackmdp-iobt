#!/usr/bin/env bash
# Thesis experiment, 10x10, run 1: NO RESTART (standard.md section 7). Rerun of 1570
# (stopped at iteration 2730 for a power-off) with identical settings:
# (P0,0) -> (P1,700) -> (P2,1400) -> (P3,2100), >= 2800 iterations then finish open
# episodes; P0-P3 = O0-O3 = seeds 1-4 (--matrix-seed 1); cold start; standard settings.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1571 \
  --nrows 10 --ncols 10 --iterations 2800 --finish-episodes --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --seed 1 --sensor-budget none \
  --matrices O0,O1,O2,O3 --matrix-seed 1 --switch-at 700,1400,2100 \
  > run1571.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1571 (10x10, no restart; rerun of 1570) pid $!"
