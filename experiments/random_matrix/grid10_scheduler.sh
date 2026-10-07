#!/usr/bin/env bash
# 10x10 random-matrix switching experiment: 5 random matrices (R0-R4), 3 passes,
# online and PPO, with and without DAL warm restart, k in {4,5}, cost in {-0.16,-0.04}.
# Same design as the 5x5/IoBT runs (results.md) at 10x10. PPO uses --moore (local
# action window): with one action per cell (100) it never learned (flat ~2.5%).
#
#   runs 1100-1103 PPO no DAL      1104-1107 PPO + DAL warm
#   runs 1108-1111 online no DAL   1112-1115 online + DAL warm
#   order within each block: (k4,-0.16) (k4,-0.04) (k5,-0.16) (k5,-0.04)
#
# Memory-aware: PPO jobs (~2 GB) only start with fewer than MAX_PPO running and
# MIN_FREE_GB available; online jobs (~0.6 GB) up to MAX_ONLINE. A job is DONE when
# its output exists, RUNNING when a matching python process exists; retried up to
# MAX_TRIES times. Base models first; DAL arms wait for their own base model.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors
O=experiments/random_matrix/grid10
L=/tmp/track_mdp_logs
PY=./track_mdp_env/bin/python
G="--nrows 10 --ncols 10"
HP="--alpha 0.05 --gamma 0.9 --lam 0.0 --epsilon 0.1 --optimistic-init 5.0"
MAX_PPO=6; MAX_ONLINE=8; MIN_FREE_GB=5; MAX_TRIES=4
mkdir -p $P $O $L
declare -A tries

running_with () {
  ps -eo args | awk -v p="$1" 'index($0, "track_mdp_env/bin/python") && index($0, p) && !index($0, "awk")' | wc -l
}
free_gb () { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }

jobs=()
for kind in ppo_prior tab_prior; do for k in 4 5; do for c in 0.16 0.04; do
  jobs+=("$kind $k $c 0"); done; done; done
r=1100
for arm in ppo_none ppo_warm tab_none tab_warm; do for k in 4 5; do for c in 0.16 0.04; do
  jobs+=("$arm $k $c $r"); r=$((r+1)); done; done; done

while true; do
  pending=0
  for j in "${jobs[@]}"; do
    set -- $j; kind=$1 k=$2 c=$3 r=$4
    ptag=ppo_grid10m_C_k${k}_c${c}; ttag=tab_grid10_C_k${k}_c${c}
    case $kind in
      ppo_prior) done_file=$P/$ptag/rllib_checkpoint.json; key="--save-checkpoint $P/$ptag ";;
      tab_prior) done_file=$P/$ttag.npy;                     key="--save-q $P/$ttag.npy ";;
      *)         done_file=$O/run$r/online_summary.json;      key="--new-run $r ";;
    esac
    [ -f "$done_file" ] && continue
    pending=$((pending+1))
    [ "$(running_with "$key")" -gt 0 ] && continue
    [ "$kind" = ppo_warm ] && [ ! -f $P/$ptag/rllib_checkpoint.json ] && continue
    [ "$kind" = tab_warm ] && [ ! -f $P/$ttag.npy ] && continue
    [ "${tries[$j]:-0}" -ge $MAX_TRIES ] && continue
    case $kind in
      ppo_*) [ "$(running_with train_switching_grid_ppo.py)" -ge $MAX_PPO ] && continue
             [ "$(free_gb)" -lt $MIN_FREE_GB ] && continue;;
      *)     [ "$(running_with train_switching_grid_online.py)" -ge $MAX_ONLINE ] && continue
             [ "$(free_gb)" -lt 2 ] && continue;;
    esac
    tries[$j]=$(( ${tries[$j]:-0} + 1 ))
    case $kind in
      ppo_prior) cmd="examples/train_switching_grid_ppo.py --env grid $G --moore --new-run 0 --matrices C --iterations-per-segment 1200 --passes 1 --max-sensors $k --sensor-rew -$c --no-detector --save-checkpoint $P/$ptag --out-dir $P/${ptag}_train"; log=prior_$ptag;;
      tab_prior) cmd="examples/train_switching_grid_online.py $G --action-mode local --episodes-per-segment 6000 --matrices C --passes 1 --max-sensors $k --sensor-rew -$c --no-detector $HP --new-run 0 --save-q $P/$ttag.npy --out-dir $P/${ttag}_train"; log=prior_$ttag;;
      ppo_none)  cmd="examples/train_switching_grid_ppo.py --env grid $G --moore --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors $k --sensor-rew -$c --no-detector --out-dir $O/run$r"; log=run$r;;
      ppo_warm)  cmd="examples/train_switching_grid_ppo.py --env grid $G --moore --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors $k --sensor-rew -$c --restart-strategy warm --prior-checkpoint $(pwd)/$P/$ptag --delta 0.01 --out-dir $O/run$r"; log=run$r;;
      tab_none)  cmd="examples/train_switching_grid_online.py $G --action-mode local --episodes-per-segment 1000 --matrices R0,R1,R2,R3,R4 --passes 3 --max-sensors $k --sensor-rew -$c --no-detector $HP --greedy-eval-every 100 --greedy-eval-episodes 10 --new-run $r --out-dir $O/run$r"; log=run$r;;
      tab_warm)  cmd="examples/train_switching_grid_online.py $G --action-mode local --episodes-per-segment 1000 --matrices R0,R1,R2,R3,R4 --passes 3 --max-sensors $k --sensor-rew -$c --restart-strategy warm --prior-path $P/$ttag.npy --delta 1e-4 $HP --greedy-eval-every 100 --greedy-eval-episodes 10 --new-run $r --out-dir $O/run$r"; log=run$r;;
    esac
    setsid nohup $PY $cmd > $L/g10_$log.log 2>&1 < /dev/null &
    echo "$(date +%T) launched $j (try ${tries[$j]})"
    case $kind in ppo_*) sleep 90;; *) sleep 5;; esac
  done
  if [ $pending -eq 0 ]; then echo "$(date +%T) ALL GRID10 DONE"; break; fi
  sleep 60
done
