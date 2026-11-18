#!/usr/bin/env bash
# Queued after the 5x5 restart runs 1552 (warm) and 1553 (cold) finish (waits on their PIDs).
#  1) run 1570: 10x10, 3 switches, NO restart -- design of 5x5 run 1551 with 700-iteration
#     segments: (P0,0)->(P1,700)->(P2,1400)->(P3,2100), >= 2800 iterations then finish episodes
#  2) run 1580: 10x10 equal-probability general base, ONLY if warm (1552) beat cold (1553)
#     on mean accuracy over the whole run (upstream metric, all evaluations).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=../../track_mdp_env/bin/python
for pid in "$@"; do while kill -0 $pid 2>/dev/null; do sleep 60; done; done
echo "$(date +%T) 1552/1553 finished"

setsid nohup $PY ../../examples/training_grid_rect.py --run 1570 \
  --nrows 10 --ncols 10 --iterations 2800 --finish-episodes --eval-interval 10 --eval-episodes 100 \
  --max-ep-steps 0 --seed 1 --sensor-budget none \
  --matrices O0,O1,O2,O3 --matrix-seed 1 --switch-at 700,1400,2100 \
  > run1570.log 2>&1 < /dev/null &
echo "$(date +%T) launched 1570 (10x10, no restart)"
$PY ../../examples/random_matrix_oracle.py --standard --matrices O0,O1,O2,O3 --matrix-seed 1 \
  --nrows 10 --ncols 10 --episodes 300 --costs 0.16 --out oracle_standard_s1.json > oracle.log 2>&1

$PY - <<'PY' > warm_vs_cold.txt
import pandas as pd
w = pd.read_csv("../thesis_5x5/agent_run1552_ppo/metrics_history.csv")
c = pd.read_csv("../thesis_5x5/agent_run1553_ppo/metrics_history.csv")
wm, cm = w.accuracy.mean(), c.accuracy.mean()
print(f"warm 1552 mean accuracy {wm*100:.2f}% | cold 1553 mean accuracy {cm*100:.2f}%")
print("DECISION:", "warm outperforms cold -> train 10x10 base" if wm > cm else "warm does not outperform cold -> no 10x10 base")
PY
cat warm_vs_cold.txt
if command grep -q "warm outperforms" warm_vs_cold.txt; then
  (cd general_base && setsid nohup ../../../track_mdp_env/bin/python ../../../examples/training_grid_rect.py --run 1580 \
    --nrows 10 --ncols 10 --iterations 300 --plateau-patience 40 --plateau-min-delta 0.005 \
    --max-iterations 2000 --finish-episodes --eval-interval 10 --eval-episodes 100 \
    --max-ep-steps 0 --movement moore --seed 1 --sensor-budget none \
    > run1580.log 2>&1 < /dev/null &)
  echo "$(date +%T) launched 1580 (10x10 general base)"
fi
