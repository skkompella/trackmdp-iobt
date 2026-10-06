#!/usr/bin/env bash
# Memory-aware scheduler for the random-matrix PPO jobs. Replaces ppo_queue.sh
# and ppo_retry.sh, which started too many Ray clusters at once (each PPO job
# is ~2 GB; at 9-10 concurrent the box ran out of memory and workers died).
#
# A job is DONE when its output exists (checkpoint for base models, summary for
# arms) and RUNNING when a matching python process exists. Anything else is
# launched -- at most MAX_RUNNING at a time, only with MIN_FREE_GB available --
# and retried up to MAX_TRIES times. Base models go first; a DAL-warm arm waits
# for its base model.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors
L=/tmp/track_mdp_logs
PY=./track_mdp_env/bin/python
MAX_RUNNING=6; MIN_FREE_GB=5; MAX_TRIES=4
declare -A tries

running_with () {   # count python processes whose args contain $1
  ps -eo args | awk -v p="$1" 'index($0, "track_mdp_env/bin/python") && index($0, p) && !index($0, "awk")' | wc -l
}
n_running () { running_with "train_switching_grid_ppo.py"; }
free_gb () { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }

jobs=()
for e in grid iobt; do for k in 4 5; do for c in 0.16 0.04; do
  jobs+=("prior $e $k $c 0"); done; done; done
r=1000
for arm in none warm; do for e in grid iobt; do for k in 4 5; do for c in 0.16 0.04; do
  jobs+=("$arm $e $k $c $r"); r=$((r+1)); done; done; done; done
# priority: base models, then warm arms (unblocked as base models land), then none
order=("${jobs[@]:0:8}" "${jobs[@]:16:8}" "${jobs[@]:8:8}")

while true; do
  pending=0
  for j in "${order[@]}"; do
    set -- $j; kind=$1 e=$2 k=$3 c=$4 r=$5
    tag=ppo_${e}_C_k${k}_c${c}
    if [ "$kind" = prior ]; then
      done_file=$P/$tag/rllib_checkpoint.json; key="--save-checkpoint $P/$tag "
    else
      done_file=experiments/random_matrix/$e/run$r/online_summary.json; key="--new-run $r "
    fi
    [ -f "$done_file" ] && continue
    pending=$((pending+1))
    [ "$(running_with "$key")" -gt 0 ] && continue
    [ "$kind" = warm ] && [ ! -f $P/$tag/rllib_checkpoint.json ] && continue
    [ "${tries[$j]:-0}" -ge $MAX_TRIES ] && continue
    [ "$(n_running)" -ge $MAX_RUNNING ] && break
    [ "$(free_gb)" -lt $MIN_FREE_GB ] && break
    tries[$j]=$(( ${tries[$j]:-0} + 1 ))
    if [ "$kind" = prior ]; then
      setsid nohup $PY examples/train_switching_grid_ppo.py --env $e --new-run 0 \
        --matrices C --iterations-per-segment 1200 --passes 1 --max-sensors $k \
        --sensor-rew -$c --no-detector --save-checkpoint $P/$tag \
        --out-dir $P/${tag}_train > $L/rm_prior_$tag.log 2>&1 < /dev/null &
    else
      if [ "$kind" = none ]; then extra="--no-detector"
      else extra="--restart-strategy warm --prior-checkpoint $(pwd)/$P/$tag --delta 0.01"; fi
      setsid nohup $PY examples/train_switching_grid_ppo.py --env $e --new-run $r \
        --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 \
        --max-sensors $k --sensor-rew -$c $extra \
        --out-dir experiments/random_matrix/$e/run$r > $L/rm_run$r.log 2>&1 < /dev/null &
    fi
    echo "$(date +%T) launched $j (try ${tries[$j]})"
    sleep 90      # let it allocate before judging memory again
  done
  if [ $pending -eq 0 ]; then echo "$(date +%T) ALL PPO DONE"; break; fi
  sleep 60
done
