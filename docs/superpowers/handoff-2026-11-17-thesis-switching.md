# Handoff — thesis switching experiments (PPO, standard.md §7)

_Saved 2026-11-17 21:35 CST before the user powered the machine off for a flight.
Branch: `transition-dal`. Latest commits: `87c8f6f` (this thread), `60baf18`
(detector plumbing from the parallel detector session), `c32c1e7`._

Read first: `standard.md` (the binding experiment standard; §7 = thesis experiments).
The parallel detector thread has its own handoff:
`docs/superpowers/handoff-2026-11-17-ppo-detector.md`.

## What the thesis experiment is

Track-MDP PPO tracking a target whose movement model P switches without warning.
Three arms, identical except for what happens on a detected change:

| Arm | Start | On an alarm |
|---|---|---|
| No restart | cold (random policy) | nothing, PPO keeps training |
| Cold restart | cold | reset weights to the run's initial random policy |
| Warm restart | cold | reset weights to the general base (equal-probability prior) |

Detector: transition-stream GLR (`src/core/ppo_transition.py`, δ_F = 0.01, 169
streams on 5×5). All via `examples/training_grid_rect.py` (`--matrices`,
`--switch-at`, `--finish-episodes`, `--detector transition --restart warm|cold`).

## Runs and their state at power-off

| Run | What | State |
|---|---|---|
| 1551 | 5×5, seeds 1–4, switches 150/300/450, no restart | **done** |
| 1560 | 5×5 general base (equal-prob Moore, plateau rule) | **done**: plateaued ~70% (best 71.5% @260) |
| 1552 | 5×5 warm restart to 1560 best | **running** at power-off (~iter 590 of ≥600; 96.5% last eval, 3 alarms) |
| 1553 | 5×5 cold restart | **running** at power-off (~iter 600; 87.6% last eval, 3 alarms) |
| 1570 | 10×10, seeds 1–4, 700-iteration segments (switches 700/1400/2100, ≥2800), no restart | **queued** (`experiments/thesis_10x10/queue.sh`, waits on PIDs of 1552/1553) |
| 1580 | 10×10 general base, launched **only if** warm 1552 beats cold 1553 on mean accuracy | **queued** (same script; decision written to `experiments/thesis_10x10/warm_vs_cold.txt`) |

Powering off kills all of these. Runs 1552/1553 may or may not have finished
(they were ~10 minutes from the end). After power-on:
1. Check `experiments/thesis_5x5/run1552.log` / `run1553.log` for "Best accuracy"
   (finished) — if not finished, they must be rerun (`cmd_warm_restart_s1.sh`,
   `cmd_cold_restart_s1.sh`); `training_grid_rect.py` has no mid-run resume.
2. The queue (`experiments/thesis_10x10/queue.sh <pid1552> <pid1553>`) died with
   the machine; relaunch it (with any dead PIDs it proceeds immediately). The user
   said they will say when to pause/resume runs — ask before relaunching.

## Results so far (5×5, upstream accuracy metric)

Oracle (standard, per-cell, knows P): 100% (99.47% tracked) at ~3 cells/step on
every model.

- **Single switch (1540 vs 1541):** converged 1530 policy dropped 99.5% → 67.8% on
  a new P and had only recovered to 74.7% after 110 iterations; cold PPO on the same
  P reached 97.2%. A converged policy adapts slower than learning from scratch.
- **No restart, 3 switches (1550 seed 20261117; 1551 seeds 1–4):** P0 learned cold
  (1551: 100% by iteration 130); each later model learned worse: segment-end
  1551 = 100 → 92.3 → 86.6 → 79.2% (sensors 4.2 → 6.7); 1550 = 91.6 → 90.7 → 89.4 →
  78.4%. Same decline on two model sets.
- **Segment length:** cold PPO reaches the 99% plateau at ~130–140 iterations
  (1530, 1551 P0) → 150-iteration segments on 5×5. 10×10 is much slower (1500: 96%
  at 630) → 700-iteration segments (user's choice).
- **Sensor budget test (1530 vs 1531):** enforcing k=6 in the env slowed early
  learning but converged to the same place → decided: no hard budget.
- **General base problem:** equal-probability walk needs all ~9 candidates watched;
  oracle shows full coverage is reward-optimal (100%), but PPO plateaus ~70% (1560)
  — a PPO training failure, not a limit. Domain randomisation would NOT fix it (a
  memoryless policy faces the same averaged movement). Candidate fixes (later):
  lower sensor cost / curriculum, entropy bonus.

## Open questions for the user

- Results of 1552 vs 1553 (warm vs cold) — not yet reported.
- Better general base candidates — user said "later".
- README.md says Camp Buckner for the 10-node data; per memory the 10-node data is
  Graces Quarters — README needs that fix.

## Gotchas learned this thread

- Never `pkill -f`/`pgrep -f` with a pattern that also appears in your own (or a
  watcher's) command line — it matched and killed/blocked our own shells three
  times. Wait on PIDs (`kill -0 $pid`) or snapshot `ps` to a file and filter on
  argv[0].
- Splitting shared files hunk-by-hunk with `git apply --unidiff-zero` misplaced
  code (fixed commit `60baf18`). Rebuild the file content explicitly and diff it
  against the previous commit instead.
- Python stdout to a file is buffered: launch with `PYTHONUNBUFFERED=1`.
- Uncommitted, not ours: `RESUME.md`, `src/core/iobt_environment.py` (pre-session),
  the confusion-test files (`ppo_detector_eval*`, other thread).
