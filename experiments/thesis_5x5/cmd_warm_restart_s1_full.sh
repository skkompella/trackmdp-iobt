#!/usr/bin/env bash
# FULL restart (fresh algorithm), warm start from the prior. Thesis experiment, 5x5: CHANGE DETECTION + WARM RESTART (standard.md section 7).
# Identical to run 1551 (no restart): (P0,0) -> (P1,150) -> (P2,300) -> (P3,450), >= 600
# iterations then finish open episodes; P0-P3 = O0-O3 = seeds 1-4 (--matrix-seed 1); cold
# start; standard settings. Plus: transition-stream detector (delta_F 0.01, 169 streams,
# window 2000, min 30 samples) on the rollout workers' tau=0 observations; on every alarm
# the policy weights are reset to the 5x5 general base (run 1560, best checkpoint).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1554 \
  --nrows 5 --ncols 5 --iterations 600 --finish-episodes --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --seed 1 --sensor-budget none \
  --matrices O0,O1,O2,O3 --matrix-seed 1 --switch-at 150,300,450 \
  --init-checkpoint $(pwd)/general_base/agent_run1560_ppo/best \
  --detector transition --delta-f 0.01 --restart warm --restart-checkpoint $(pwd)/general_base/agent_run1560_ppo/best \
  > run1554.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1554 (warm restart)"
