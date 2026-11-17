#!/usr/bin/env bash
# 1) General base (run 1510): equal-probability Moore walk (stay + 8 neighbours, exit 0.005),
#    10x10, port PPO settings, k=9, no episode cap, 1000 iterations.
# 2) Then the switching experiment with 300 iterations per movement model (O0-O3 x 3 passes):
#    1611 PPO from scratch | 1612 from general base, no restarts | 1613 from general base + DAL warm restart to it
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
GB=experiments/port_switching/general_base
O=experiments/port_switching
if [ ! -f $GB/agent_run1510_ppo/rllib_checkpoint.json ] || [ ! -f $GB/DONE ]; then
  (cd $GB && ../../../track_mdp_env/bin/python ../../../examples/training_grid_rect.py --run 1510 \
     --nrows 10 --ncols 10 --iterations 1000 --eval-interval 10 --eval-episodes 100 \
     --max-ep-steps 0 --movement moore > run1510.log 2>&1) && touch $GB/DONE
fi
[ -f $GB/DONE ] || { echo "base training failed"; exit 1; }
B=$(pwd)/$GB/agent_run1510_ppo
COMMON="--env grid --nrows 10 --ncols 10 --moore --moore-moves --exit-prob 0.005 --max-ep-steps 0
  --ppo-preset port --max-sensors 9 --sensor-rew -0.16 --matrices O0,O1,O2,O3 --matrix-seed 20261114
  --iterations-per-segment 300 --passes 3"
launch () { r=$1; shift
  setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py $COMMON \
    --new-run $r --out-dir $O/run$r "$@" > $O/run$r.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $r"; sleep 60; }
launch 1611 --no-detector
launch 1612 --no-detector --init-checkpoint $B
launch 1613 --init-checkpoint $B --restart-strategy warm --prior-checkpoint $B --delta 0.01
