#!/usr/bin/env bash
# Pause every running transition_dal job: stop the scheduler from launching,
# ask each running job to checkpoint and exit, and wait until all have.
# After this it is safe to kill anything left (pkill -f run_phase.sh).
# Resume later by rerunning experiments/transition_dal/run_phase.sh <phase>.
set -u
cd "$(dirname "$0")/../.."
D=experiments/transition_dal
touch $D/PAUSE_ALL
outs=$(ps -eo args | grep -v grep | grep train_online_dal.py \
       | grep -o -- "--out-dir [^ ]*" | awk '{print $2}' | sort -u)
if [ -z "$outs" ]; then echo "no running jobs"; exit 0; fi
for o in $outs; do touch "$o/PAUSE"; echo "pause requested: $o"; done
for o in $outs; do
  while ps -eo args | grep -v grep | grep train_online_dal.py | grep -q -- "--out-dir $o\( \|$\)"; do
    sleep 2
  done
  echo "paused: $o ($(cat "$o/status.json" 2>/dev/null | tr -d '\n '))"
done
echo "all jobs paused; the scheduler will exit on its own (or pkill -f run_phase.sh)"
