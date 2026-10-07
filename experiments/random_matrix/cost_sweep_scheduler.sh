#!/usr/bin/env bash
# k=5 sensor-cost sweep with DAL warm restart, all three environments.
# Waits for the 10x10 k=6 runs to finish first.
#   online + DAL: costs 0.01 0.02 0.06 0.08 0.10 0.12  -> runs 1200-1229
#   PPO + DAL:    costs 0.02 0.08 0.12                  -> runs 1250-1279
#   run = base + 10*env_index + cost_index, env order: grid (5x5), iobt, grid10
# Costs -0.04 and -0.16 already exist (runs 1011/1010, 1015/1014, 1107/1106 PPO;
# 1027/1026, 1031/1030, 1115/1114 online). Same design as results.md.
cd "$(dirname "$0")/../.."
until grep -q "ALL GRID10 K6 DONE" experiments/random_matrix/grid10_k6_scheduler.status; do sleep 120; done
echo "$(date +%T) k=6 done, starting cost sweep"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=experiments/random_matrix/priors
L=/tmp/track_mdp_logs
PY=./track_mdp_env/bin/python
HP="--alpha 0.05 --gamma 0.9 --lam 0.0 --epsilon 0.1 --optimistic-init 5.0"
MAX_PPO=6; MAX_ONLINE=8; MIN_FREE_GB=5; MAX_TRIES=4; K=5
declare -A tries
running_with () {
  ps -eo args | awk -v p="$1" 'index($0, "track_mdp_env/bin/python") && index($0, p) && !index($0, "awk")' | wc -l
}
free_gb () { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }

envs=(grid iobt grid10)
jobs=()
for ei in 0 1 2; do ci=0; for c in 0.01 0.02 0.06 0.08 0.10 0.12; do
  jobs+=("tab_prior ${envs[$ei]} $c 0"); jobs+=("tab_warm ${envs[$ei]} $c $((1200+10*ei+ci))"); ci=$((ci+1)); done; done
for ei in 0 1 2; do ci=0; for c in 0.02 0.08 0.12; do
  jobs+=("ppo_prior ${envs[$ei]} $c 0"); jobs+=("ppo_warm ${envs[$ei]} $c $((1250+10*ei+ci))"); ci=$((ci+1)); done; done

online_cmd () {   # env -> script + flags (episode-count flag last)
  case $1 in
    grid)   echo "examples/train_switching_grid_online.py --action-mode local --episodes-per-segment";;
    grid10) echo "examples/train_switching_grid_online.py --nrows 10 --ncols 10 --action-mode local --episodes-per-segment";;
    iobt)   echo "examples/train_multiloop_online.py --episodes-per-loop";;
  esac
}
ppo_env () {
  case $1 in
    grid)   echo "--env grid";;
    grid10) echo "--env grid --nrows 10 --ncols 10 --moore";;
    iobt)   echo "--env iobt";;
  esac
}
ppo_tag () { case $1 in grid10) echo "ppo_grid10m_C_k${K}_c$2";; *) echo "ppo_$1_C_k${K}_c$2";; esac; }

while true; do
  pending=0
  for j in "${jobs[@]}"; do
    set -- $j; kind=$1 e=$2 c=$3 r=$4
    ttag=tab_${e}_C_k${K}_c${c}; ptag=$(ppo_tag $e $c)
    case $kind in
      tab_prior) done_file=$P/$ttag.npy; key="--save-q $P/$ttag.npy ";;
      ppo_prior) done_file=$P/$ptag/rllib_checkpoint.json; key="--save-checkpoint $P/$ptag ";;
      *)         done_file=experiments/random_matrix/$e/run$r/online_summary.json; key="--new-run $r ";;
    esac
    [ -f "$done_file" ] && continue
    pending=$((pending+1))
    [ "$(running_with "$key")" -gt 0 ] && continue
    [ "$kind" = tab_warm ] && [ ! -f $P/$ttag.npy ] && continue
    [ "$kind" = ppo_warm ] && [ ! -f $P/$ptag/rllib_checkpoint.json ] && continue
    [ "${tries[$j]:-0}" -ge $MAX_TRIES ] && continue
    case $kind in
      ppo_*) [ "$(running_with train_switching_grid_ppo.py)" -ge $MAX_PPO ] && continue
             [ "$(free_gb)" -lt $MIN_FREE_GB ] && continue;;
      *)     [ "$(running_with _online.py)" -ge $MAX_ONLINE ] && continue
             [ "$(free_gb)" -lt 2 ] && continue;;
    esac
    tries[$j]=$(( ${tries[$j]:-0} + 1 ))
    case $kind in
      tab_prior) cmd="$(online_cmd $e) 6000 --matrices C --passes 1 --max-sensors $K --sensor-rew -$c --no-detector $HP --new-run 0 --save-q $P/$ttag.npy --out-dir $P/${ttag}_train";;
      tab_warm)  cmd="$(online_cmd $e) 1000 --matrices R0,R1,R2,R3,R4 --passes 3 --max-sensors $K --sensor-rew -$c --restart-strategy warm --prior-path $P/$ttag.npy --delta 1e-4 $HP --greedy-eval-every 100 --greedy-eval-episodes 10 --new-run $r --out-dir experiments/random_matrix/$e/run$r";;
      ppo_prior) cmd="examples/train_switching_grid_ppo.py $(ppo_env $e) --new-run 0 --matrices C --iterations-per-segment 1200 --passes 1 --max-sensors $K --sensor-rew -$c --no-detector --save-checkpoint $P/$ptag --out-dir $P/${ptag}_train";;
      ppo_warm)  cmd="examples/train_switching_grid_ppo.py $(ppo_env $e) --new-run $r --matrices R0,R1,R2,R3,R4 --iterations-per-segment 200 --passes 3 --max-sensors $K --sensor-rew -$c --restart-strategy warm --prior-checkpoint $(pwd)/$P/$ptag --delta 0.01 --out-dir experiments/random_matrix/$e/run$r";;
    esac
    tag=$( [ "$r" = 0 ] && echo "${kind}_${e}_c$c" || echo "run$r" )
    setsid nohup $PY $cmd > $L/sweep_$tag.log 2>&1 < /dev/null &
    echo "$(date +%T) launched $j (try ${tries[$j]})"
    case $kind in ppo_*) sleep 90;; *) sleep 5;; esac
  done
  if [ $pending -eq 0 ]; then echo "$(date +%T) ALL COST SWEEP DONE"; break; fi
  sleep 60
done
