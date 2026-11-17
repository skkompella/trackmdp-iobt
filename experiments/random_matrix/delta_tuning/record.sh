#!/usr/bin/env bash
# Record the eval hit/miss stream PPO's detector sees (10x10, k=5, cost -0.04, R0-R4 x 3 passes),
# with warm restarts at the TRUE switch times, i.e. the stream a perfect DAL detector would leave.
# Used to tune delta = T^-p offline (tune_delta.py) instead of one training run per delta.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors; O=experiments/random_matrix/delta_tuning
setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env grid --nrows 10 --ncols 10 --moore \
  --new-run 1410 --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors 5 --sensor-rew -0.04 \
  --no-detector --restart-strategy warm --oracle-restart --prior-checkpoint $(pwd)/$P/ppo_grid10m_C_k5_c0.04 \
  --save-stream $O/stream_oracle_k5_c0.04.npz --out-dir $O/run1410 > /tmp/track_mdp_logs/g10_run1410.log 2>&1 < /dev/null &
echo "$(date +%T) launched recording run 1410"
# Second stream (run 1411): no restarts at all, so each switch hits the OLD policy, as it does in a
# real DAL run before the alarm. Started after 1410 to stay within memory.
until [ -f experiments/random_matrix/delta_tuning/run1410/online_summary.json ] || ! pgrep -f "new-run 1410" >/dev/null; do sleep 60; done
setsid nohup ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env grid --nrows 10 --ncols 10 --moore \
  --new-run 1411 --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors 5 --sensor-rew -0.04 \
  --no-detector --save-stream $O/stream_norestart_k5_c0.04.npz --out-dir $O/run1411 \
  > /tmp/track_mdp_logs/g10_run1411.log 2>&1 < /dev/null &
echo "$(date +%T) launched recording run 1411"
