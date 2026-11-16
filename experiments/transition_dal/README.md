# transition_dal — validating the transition-stream change detector

Design: `docs/superpowers/specs/2026-11-15-transition-dal-detector-design.md`.
Plan: `docs/superpowers/plans/2026-11-15-transition-dal-detector.md`.
Runner: `examples/train_online_dal.py` (code in `src/core/online_dal.py`).

Tabular SARSA, k = 5, sensor cost -0.04, 1000 episodes x 100 steps per segment,
5x5 grid and IoBT. One 15-segment run takes about 12 minutes (measured
0.045-0.048 s/episode with the detector on, 2026-11-15).

## Phases

| Phase | Command | Jobs | What it shows |
|---|---|---|---|
| null | `run_phase.sh null` | 6 (2 envs x 3 seeds), R0 only, monitor mode, streams saved | false-alarm rate under no change; ordered vs shuffled replay (`examples/replay_streams.py`) |
| sweep | `run_phase.sh sweep` | 10 (2 envs x rho in {0, .05, .1, .2, .3}), R0..R4 x 3, monitor mode | detection delay per switch vs probe rate; picks rho* |
| arms | `run_phase.sh arms` | 30 (2 envs x 5 arms x 3 seeds), restart mode | accuracy: none, oracle cold/warm, transition cold/warm |

`arms` needs `rho_star.env` (`RHO_grid=...`, `RHO_iobt=...`), written after the sweep.
At 8 parallel jobs each phase takes under an hour of wall time; replaying one null run's
streams takes up to an hour.

Summaries: `examples/summarize_dal_validation.py --runs 'experiments/transition_dal/<phase>/*' --out experiments/transition_dal/<phase>_summary.md`.

## Pause, kill, resume

```bash
experiments/transition_dal/pause_all.sh     # stop launching, checkpoint every running job, wait
pkill -f run_phase.sh                       # optional: the scheduler exits by itself once all are paused
# ... later ...
setsid nohup experiments/transition_dal/run_phase.sh <phase> >> experiments/transition_dal/logs/<phase>.sched.log 2>&1 &
```

A single job can be paused with `touch <out-dir>/PAUSE` or `kill -TERM <pid>`: it
finishes the current episode, checkpoints, writes `status.json` = paused, and exits 0.
The scheduler resumes jobs with `--resume`; a resumed run is bit-identical to an
uninterrupted one (`tests/test_online_dal.py`). Jobs also checkpoint every 200 episodes,
so even a `kill -9` loses at most 200 episodes.

Per-run outputs: `online_summary.json`, `online_history.csv`, `alarms.json`
(alarms with the firing stream, and the true switches), `status.json`, `ckpt/`,
and `streams.npz` for null runs. `ckpt/` and `streams.npz` are not committed.
