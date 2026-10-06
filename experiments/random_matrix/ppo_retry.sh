#!/usr/bin/env bash
# Re-run PPO jobs that crashed at startup (RLlib workers failed to come up while
# 8 Ray clusters started at once). 2 at a time, staggered, up to 4 attempts.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors
L=/tmp/track_mdp_logs
attempt () {   # $1 = label, rest = command; retries until exit 0 and no traceback
  label=$1; shift
  for try in 1 2 3 4; do
    "$@" > $L/rm_$label.log 2>&1; rc=$?
    if [ $rc -eq 0 ] && ! grep -q Traceback $L/rm_$label.log; then
      echo "$(date +%T) $label ok (try $try)"; return 0; fi
    echo "$(date +%T) $label failed rc=$rc (try $try)"; sleep 90
  done
  return 1
}
prior () {
  e=$1 k=$2 c=$3; tag=ppo_${e}_C_k${k}_c${c}
  attempt prior_$tag ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py \
    --env $e --new-run 0 --matrices C --iterations-per-segment 1200 --passes 1 \
    --max-sensors $k --sensor-rew -$c --no-detector --save-checkpoint $P/$tag \
    --out-dir $P/${tag}_train && touch $P/$tag.DONE
}
arm () {
  r=$1 e=$2 k=$3 c=$4
  attempt run$r ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py \
    --env $e --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 \
    --passes 3 --max-sensors $k --sensor-rew -$c --no-detector \
    --out-dir experiments/random_matrix/$e/run$r
}
export -f attempt prior arm; export P L
printf '%s\n' "prior grid 4 0.16" "prior grid 5 0.04" "prior grid 5 0.16" \
  "prior iobt 4 0.16" "prior iobt 5 0.04" "arm 1000 grid 4 0.16" \
  "arm 1001 grid 4 0.04" "arm 1003 grid 5 0.04" \
  | xargs -P 2 -L 1 bash -c 'sleep $((RANDOM % 60)); "$0" "$@"'
echo "$(date +%T) RETRY QUEUE DONE"
