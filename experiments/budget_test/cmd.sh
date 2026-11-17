#!/usr/bin/env bash
# Sensor-budget comparison on 5x5, everything else per standard.md:
# infinite horizon (exit 0.005, no cap), original 9-candidate/4-slot movement (seed 1),
# PPO port settings, upstream accuracy metric, best/current checkpoints + metric history.
# k = 6 (upstream's only training value).
#   1530  --sensor-budget none : upstream, k never applied (train or eval)
#   1531  --sensor-budget env  : at most 6 tracking sensors per step, enforced in the env (train + eval)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for spec in "1530 none" "1531 env"; do set -- $spec
  setsid nohup ../../track_mdp_env/bin/python ../../examples/training_grid_rect.py --run $1 \
    --nrows 5 --ncols 5 --iterations 1000 --eval-interval 10 --eval-episodes 100 \
    --max-ep-steps 0 --movement original --seed 1 --max-sensors 6 --sensor-budget $2 \
    > run$1.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $1 (sensor budget $2)"; sleep 30
done
