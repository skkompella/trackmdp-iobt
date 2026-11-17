#!/usr/bin/env bash
# 10x10 PPO + DAL warm, k=5, cost -0.04, delta = 1/T, 1/T^2, 1/T^3 with T = 15,360 training
# episodes (15 segments x 200 iterations x 512 steps / 100 steps per episode).
# Compare with 1107 (delta 0.01) and 1403 (delta 1/sqrt(T)).   runs 1404, 1405, 1406
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors; O=experiments/random_matrix/grid10; L=/tmp/track_mdp_logs
r=1404
for DELTA in 6.5104e-05 4.2386e-09 2.7595e-13; do
  setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env grid --nrows 10 --ncols 10 --moore \
    --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors 5 --sensor-rew -0.04 \
    --restart-strategy warm --prior-checkpoint $(pwd)/$P/ppo_grid10m_C_k5_c0.04 --delta $DELTA \
    --out-dir $O/run$r > $L/g10_run$r.log 2>&1 < /dev/null &
  echo "$(date +%T) launched run $r delta=$DELTA"
  r=$((r+1)); sleep 90
done
