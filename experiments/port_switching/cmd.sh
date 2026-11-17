#!/usr/bin/env bash
# Switching experiment with the RLlib-port base model (run 1500 @ iteration 630, 96.88%).
# 10x10, original Track-MDP movement (O0-O3: 4 random 3x3 destinations per cell, seed 20261114),
# exit 0.005 and no training episode cap, port PPO settings, k=9, cost -0.16.
# O0 -> O1 -> O2 -> O3, 3 passes (11 switches), 50 iterations per segment (200k steps).
#   1601  PPO from scratch, no detector         (the earlier "PPO, no restarts" arm)
#   1602  PPO from the base model, no detector  (fine-tune this model straight through)
#   1603  PPO from the base model + DAL warm restart to it (GLR, delta 0.01)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
B=$(pwd)/experiments/port_original/base_ckpt_96
O=experiments/port_switching
COMMON="--env grid --nrows 10 --ncols 10 --moore --moore-moves --exit-prob 0.005 --max-ep-steps 0
  --ppo-preset port --max-sensors 9 --sensor-rew -0.16 --matrices O0,O1,O2,O3 --matrix-seed 20261114
  --iterations-per-segment 50 --passes 3"
launch () { r=$1; shift
  setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py $COMMON \
    --new-run $r --out-dir $O/run$r "$@" > $O/run$r.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $r"; sleep 60; }
launch 1601 --no-detector
launch 1602 --no-detector --init-checkpoint $B
launch 1603 --init-checkpoint $B --restart-strategy warm --prior-checkpoint $B --delta 0.01
