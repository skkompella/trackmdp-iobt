#!/usr/bin/env bash
# Resumable scheduler for the transition-stream detector validation.
#   experiments/transition_dal/run_phase.sh null|sweep|arms
# Skips finished jobs (online_summary.json), resumes paused ones (ckpt/LATEST),
# runs at most MAX_JOBS at once, and stops launching while PAUSE_ALL exists
# (see pause_all.sh). Writes "ALL <PHASE> DONE" to <phase>.status at the end.
set -u
cd "$(dirname "$0")/../.."
PHASE=${1:?usage: run_phase.sh null|sweep|arms}
D=experiments/transition_dal
PY=./track_mdp_env/bin/python
MAX_JOBS=${MAX_JOBS:-8}; MAX_TRIES=3; MIN_FREE_GB=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $D/logs $D/$PHASE
rm -f $D/PAUSE_ALL
STATUS=$D/$PHASE.status
: > $STATUS

BASE="--max-sensors 5 --sensor-rew -0.04 --max-ep-steps 100 --time-limit 1
      --episodes-per-segment 1000 --matrix-seed 20261006
      --alpha 0.05 --gamma 0.9 --lam 0.0 --epsilon 0.1 --optimistic-init 5.0
      --greedy-eval-every 100 --greedy-eval-episodes 10"
PRIOR_grid=experiments/random_matrix/priors/tab_grid_C_k5_c0.04.npy
PRIOR_iobt=experiments/random_matrix/priors/tab_iobt_C_k5_c0.04.npy
SWITCHING="--matrices R0,R1,R2,R3,R4 --passes 3"

declare -a TAGS ARGS
add () { TAGS+=("$1"); ARGS+=("$2"); }
case $PHASE in
  null)
    for e in grid iobt; do for s in 1 2 3; do
      add "${e}_transition_monitor_s$s" "--env $e --train-seed $s --matrices R0 --passes 15 \
        --detector transition --detector-mode monitor --probe-rate 0.2 --save-streams"
    done; done;;
  sweep)
    for e in grid iobt; do for r in 0 0.05 0.1 0.2 0.3; do
      add "${e}_rho${r}_s1" "--env $e --train-seed 1 $SWITCHING \
        --detector transition --detector-mode monitor --probe-rate $r"
    done; done;;
  arms)
    [ -f $D/rho_star.env ] || { echo "missing $D/rho_star.env (Task 9)"; exit 1; }
    source $D/rho_star.env          # defines RHO_grid and RHO_iobt
    for e in grid iobt; do
      rho_var=RHO_$e; rho=${!rho_var}; prior_var=PRIOR_$e; prior=${!prior_var}
      for s in 1 2 3; do
        common="--env $e --train-seed $s $SWITCHING"
        add "${e}_none_s$s"            "$common --detector none"
        add "${e}_oracle-cold_s$s"     "$common --detector oracle --restart-strategy cold"
        add "${e}_oracle-warm_s$s"     "$common --detector oracle --restart-strategy warm --prior-path $prior"
        add "${e}_transition-cold_s$s" "$common --detector transition --probe-rate $rho --restart-strategy cold"
        add "${e}_transition-warm_s$s" "$common --detector transition --probe-rate $rho --restart-strategy warm --prior-path $prior"
      done
    done;;
  *) echo "unknown phase $PHASE"; exit 1;;
esac

free_gb () { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
running_out () { ps -eo args | grep -F -- "--out-dir $1" | grep -v grep | grep -c train_online_dal.py; }
n_running () { ps -eo args | grep -v grep | grep train_online_dal.py | grep -c -- "--out-dir $D/$PHASE/"; }
declare -A tries

echo "$(date +%T) phase $PHASE: ${#TAGS[@]} jobs, MAX_JOBS=$MAX_JOBS"
while true; do
  pending=0
  for k in "${!TAGS[@]}"; do
    tag=${TAGS[$k]}; out=$D/$PHASE/$tag
    [ -f $out/online_summary.json ] && continue
    pending=$((pending+1))
    [ -f $D/PAUSE_ALL ] && continue
    [ "$(running_out $out)" -gt 0 ] && continue
    [ "${tries[$tag]:-0}" -ge $MAX_TRIES ] && continue
    [ "$(n_running)" -ge $MAX_JOBS ] && continue
    [ "$(free_gb)" -lt $MIN_FREE_GB ] && continue
    resume=""; [ -f $out/ckpt/LATEST ] && resume="--resume"
    tries[$tag]=$(( ${tries[$tag]:-0} + 1 ))
    setsid nohup $PY examples/train_online_dal.py $BASE ${ARGS[$k]} --out-dir $out $resume \
      >> $D/logs/$PHASE.$tag.log 2>&1 < /dev/null &
    echo "$(date +%T) launched $tag ${resume:+(resume)} try ${tries[$tag]}"
    sleep 2
  done
  if [ $pending -eq 0 ]; then echo "ALL ${PHASE^^} DONE" | tee -a $STATUS; break; fi
  if [ -f $D/PAUSE_ALL ] && [ "$(n_running)" -eq 0 ]; then
    echo "$(date +%T) PAUSED ($pending jobs left); rerun run_phase.sh $PHASE to resume" | tee -a $STATUS
    break
  fi
  stuck=1
  for k in "${!TAGS[@]}"; do
    tag=${TAGS[$k]}; out=$D/$PHASE/$tag
    [ -f $out/online_summary.json ] && continue
    { [ "${tries[$tag]:-0}" -lt $MAX_TRIES ] || [ "$(running_out $out)" -gt 0 ]; } && stuck=0
  done
  if [ $stuck -eq 1 ]; then echo "$(date +%T) GAVE UP: jobs failed $MAX_TRIES times; see $D/logs" | tee -a $STATUS; break; fi
  sleep 20
done
