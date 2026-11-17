#!/usr/bin/env bash
# 10x10 PPO + DAL warm restart with delta = 1/sqrt(T), T = training episodes in the run:
#   15 segments x 200 iterations x 512 steps / 100 steps per episode = 15,360 -> delta = 8.0687e-3
# Same design as runs 1104-1107 (which used delta = 0.01).
#   1400 (k4,-0.16)  1401 (k4,-0.04)  1402 (k5,-0.16)  1403 (k5,-0.04)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors; O=experiments/random_matrix/grid10; L=/tmp/track_mdp_logs
DELTA=8.0687e-3
mkdir -p $L
r=1400
for k in 4 5; do for c in 0.16 0.04; do
  setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env grid --nrows 10 --ncols 10 --moore \
    --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors $k --sensor-rew -$c \
    --restart-strategy warm --prior-checkpoint $(pwd)/$P/ppo_grid10m_C_k${k}_c${c} --delta $DELTA \
    --out-dir $O/run$r > $L/g10_run$r.log 2>&1 < /dev/null &
  echo "$(date +%T) launched run $r k=$k cost=-$c delta=$DELTA"
  r=$((r+1)); sleep 90
done; done
