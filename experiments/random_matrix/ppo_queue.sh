#!/usr/bin/env bash
# PPO jobs for the random-matrix switching experiment. Runs detached, 8 at a time.
#   1. 8 base models: equal-probability matrix C, 1200 iterations, per env/k/cost
#   2. 8 PPO no-DAL arms     (R0..R4 x 3 passes x 200 iterations)
#   3. 8 PPO DAL-warm arms   (wait for their own base model's DONE marker)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors
L=/tmp/track_mdp_logs
mkdir -p $P $L
jobs_file=$(mktemp)
for e in grid iobt; do for k in 4 5; do for c in 0.16 0.04; do
  echo "prior $e $k $c" >> $jobs_file
done; done; done
r=1000
for arm in none warm; do
for e in grid iobt; do for k in 4 5; do for c in 0.16 0.04; do
  echo "$arm $e $k $c $r" >> $jobs_file; r=$((r+1))
done; done; done; done

run_job () {
  set -- $1
  kind=$1 e=$2 k=$3 c=$4 r=$5
  if [ "$kind" = prior ]; then
    tag=ppo_${e}_C_k${k}_c${c}
    ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env $e \
      --new-run 0 --matrices C --iterations-per-segment 1200 --passes 1 \
      --max-sensors $k --sensor-rew -$c --no-detector --save-checkpoint $P/$tag \
      --out-dir $P/${tag}_train > $L/rm_prior_$tag.log 2>&1 \
      && touch $P/$tag.DONE
    echo "$(date +%T) prior $tag rc=$?"
    return
  fi
  out=experiments/random_matrix/$e/run$r
  if [ "$kind" = none ]; then
    extra="--no-detector"
  else
    tag=ppo_${e}_C_k${k}_c${c}
    until [ -f $P/$tag.DONE ]; do sleep 60; done
    extra="--restart-strategy warm --prior-checkpoint $(pwd)/$P/$tag --delta 0.01"
  fi
  ./track_mdp_env/bin/python examples/train_switching_grid_ppo.py --env $e \
    --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 \
    --passes 3 --max-sensors $k --sensor-rew -$c $extra --out-dir $out \
    > $L/rm_run$r.log 2>&1
  echo "$(date +%T) run $r ($kind $e k$k c-$c) rc=$?"
}
export -f run_job; export P L
xargs -P 8 -I{} bash -c 'run_job "{}"' < $jobs_file
echo "$(date +%T) ALL PPO DONE"
