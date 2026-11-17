#!/usr/bin/env bash
# 5x5 switching experiment, same design as 1611-1613 (10x10):
# original Track-MDP movement O0-O3 (seed 20261114) at 5x5, exit 0.005, no episode cap,
# port PPO settings, k=9, cost -0.16, 300 iterations per model, 3 passes (11 switches).
# Base: general_base_5x5 (run 1520, equal-probability Moore walk, 85.5% at iteration 1000).
#   1621 PPO from scratch | 1622 from base, no restarts | 1623 from base + DAL warm restart to it (delta 0.01)
# Memory-aware: each run starts only when >= 5 GB is available (each needs ~3-4 GB).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
B=$(pwd)/experiments/port_switching/general_base_5x5/agent_run1520_ppo
O=experiments/port_switching
COMMON="--env grid --nrows 5 --ncols 5 --moore --moore-moves --exit-prob 0.005 --max-ep-steps 0
  --ppo-preset port --max-sensors 9 --sensor-rew -0.16 --matrices O0,O1,O2,O3 --matrix-seed 20261114
  --iterations-per-segment 300 --passes 3"
free_gb () { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
launch () { r=$1; shift
  until [ "$(free_gb)" -ge 5 ]; do sleep 120; done
  setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py $COMMON \
    --new-run $r --out-dir $O/run$r "$@" > $O/run$r.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $r (free $(free_gb) GB)"; sleep 120; }
launch 1621 --no-detector
launch 1622 --no-detector --init-checkpoint $B
launch 1623 --init-checkpoint $B --restart-strategy warm --prior-checkpoint $B --delta 0.01
echo "$(date +%T) all 5x5 runs launched"
