#!/usr/bin/env bash
# Resume run 1623 (stopped by an external SIGTERM on 2026-11-15 21:37) from its last checkpoint.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
B=$(pwd)/experiments/port_switching/general_base_5x5/agent_run1520_ppo
O=experiments/port_switching
setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env grid --nrows 5 --ncols 5 \
  --moore --moore-moves --exit-prob 0.005 --max-ep-steps 0 --ppo-preset port --max-sensors 9 --sensor-rew -0.16 \
  --matrices O0,O1,O2,O3 --matrix-seed 20261114 --iterations-per-segment 300 --passes 3 \
  --new-run 1623 --out-dir $O/run1623 --init-checkpoint $B --restart-strategy warm --prior-checkpoint $B \
  --delta 0.01 --resume >> $O/run1623.log 2>&1 < /dev/null &
echo "resumed 1623"
