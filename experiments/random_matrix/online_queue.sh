#!/usr/bin/env bash
# Online tabular jobs for the random-matrix switching experiment (detached).
#   1. 8 base tables: equal-probability matrix C, 6000 episodes, per env/k/cost
#   2. 16 arms: {none, DAL-warm} x env x k x cost on R0..R4 x 3 passes x 1000 eps
# Learner: alpha 0.05, gamma 0.9, lambda 0, eps 0.1, optimistic 5 (chosen by a
# pilot on R0/R1: IoBT's own defaults scored 48% on stochastic movement).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors
L=/tmp/track_mdp_logs
HP="--alpha 0.05 --gamma 0.9 --lam 0.0 --epsilon 0.1 --optimistic-init 5.0"
mkdir -p $P $L

script_for () {   # env -> script + env-specific flags
  if [ "$1" = grid ]; then
    echo "examples/train_switching_grid_online.py --action-mode local --episodes-per-segment"
  else
    echo "examples/train_multiloop_online.py --episodes-per-loop"
  fi
}

prior_job () {
  e=$1 k=$2 c=$3; tag=tab_${e}_C_k${k}_c${c}
  ./track_mdp_env/bin/python $(script_for $e) 6000 --matrices C --passes 1 \
    --max-sensors $k --sensor-rew -$c --no-detector $HP --new-run 0 \
    --save-q $P/$tag.npy --out-dir $P/${tag}_train > $L/rm_prior_$tag.log 2>&1
  echo "$(date +%T) prior $tag rc=$?"
}
arm_job () {
  arm=$1 e=$2 k=$3 c=$4 r=$5
  if [ "$arm" = none ]; then extra="--no-detector"
  else extra="--restart-strategy warm --prior-path $P/tab_${e}_C_k${k}_c${c}.npy --delta 1e-4"; fi
  ./track_mdp_env/bin/python $(script_for $e) 1000 --matrices R0,R1,R2,R3,R4 \
    --passes 3 --max-sensors $k --sensor-rew -$c $extra $HP --new-run $r \
    --greedy-eval-every 100 --greedy-eval-episodes 10 \
    --out-dir experiments/random_matrix/$e/run$r > $L/rm_run$r.log 2>&1
  echo "$(date +%T) run $r (online $arm $e k$k c-$c) rc=$?"
}
export -f script_for prior_job arm_job; export P L HP

for e in grid iobt; do for k in 4 5; do for c in 0.16 0.04; do echo "$e $k $c"; done; done; done \
  | xargs -P 8 -L 1 bash -c 'prior_job $0 $1 $2'
r=1016
for arm in none warm; do for e in grid iobt; do for k in 4 5; do for c in 0.16 0.04; do
  echo "$arm $e $k $c $r"; r=$((r+1)); done; done; done; done \
  | xargs -P 8 -L 1 bash -c 'arm_job $0 $1 $2 $3 $4'
echo "$(date +%T) ALL ONLINE DONE"
