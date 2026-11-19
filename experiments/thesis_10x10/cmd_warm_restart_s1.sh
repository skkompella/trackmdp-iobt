#!/usr/bin/env bash
# Thesis experiment, 10x10, run 2: CHANGE DETECTION + WARM RESTART (standard.md section 7).
# Identical to run 1570 (no restart): (P0,0) -> (P1,700) -> (P2,1400) -> (P3,2100), >= 2800
# iterations then finish open episodes; P0-P3 = O0-O3 = seeds 1-4 (--matrix-seed 1);
# standard settings. STARTS FROM THE PRIOR (--init-checkpoint 1580 best). Plus: transition-stream detector (delta_F 0.01, 784 streams on 10x10,
# window 2000, min 30 samples) on the rollout workers' tau=0 observations; on every alarm the
# algorithm is fully restarted (new networks, optimizer, KL coeff, rollout workers/envs) and loaded with the 10x10 general base (run 1580, best checkpoint, 69.8%).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run 1572 \
  --nrows 10 --ncols 10 --iterations 2800 --finish-episodes --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --seed 1 --sensor-budget none \
  --matrices O0,O1,O2,O3 --matrix-seed 1 --switch-at 700,1400,2100 \
  --init-checkpoint $(pwd)/general_base/agent_run1580_ppo/best \
  --detector transition --delta-f 0.01 --restart warm \
  --restart-checkpoint $(pwd)/general_base/agent_run1580_ppo/best \
  > run1572.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1572 (10x10 warm restart) pid $!"
