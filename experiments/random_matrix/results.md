# Random-matrix switching experiment: online vs PPO, with and without DAL (runs 1000-1031)

## Setup

- **Environments:** 5 random transition matrices (R0-R4) on the IoBT graph and 5 on
  the 5x5 grid. Each row ~ Dirichlet(1) over stay + neighbours (seed 20261006).
- **Schedule:** R0 -> R1 -> R2 -> R3 -> R4, 3 passes (15 segments, 14 switches).
  Online: 1000 episodes per segment. PPO: 200 iterations per segment (~1024 episodes).
- **Online learner:** tabular SARSA, alpha 0.05, gamma 0.9, lambda 0, eps 0.1,
  optimistic 5. Full subset table on IoBT, local (reachable-cell) table on the grid.
- **DAL:** GLR on the per-step hit stream (online delta 1e-4, PPO delta 0.01 at batch
  resolution), warm restart to an equal-probability (matrix C) base model trained per
  env / k / cost: 8 tables (6000 episodes) and 8 PPO checkpoints (1200 iterations).
- **Reference:** matrix-aware tracker watching the top-m predicted cells
  (`random_matrix_oracle.py`); "best" = the m with the highest reward, "full" = m = k.
- Accuracy: online = greedy mean / end-of-segment; PPO = greedy per-iteration eval,
  mean / last-20-iterations. "cells" = cells switched on per tracked step at segment end.
  One seed per run.

## Results

| env | k | cost | ref best / full | online | online + DAL | PPO | PPO + DAL |
|---|---:|---:|---|---|---|---|---|
| grid | 4 | -0.16 | 94.6 / 96.0 | 78.0 / 78.8, 2.80c | 75.8 / 76.8, 3.20c, r473 (13/14) | 50.7 / 56.4, 3.93c | 76.1 / 77.6, 3.92c, r201 (14/14) |
| grid | 4 | -0.04 | 96.0 / 96.0 | **92.2 / 92.7**, 3.47c | **92.3 / 92.7**, 3.76c, r96 (9/14) | 45.5 / 49.3, 3.98c | 56.2 / 57.4, 4.00c, r152 (13/14) |
| grid | 5 | -0.16 | 94.6 / 99.0 | 73.2 / 74.0, 2.74c | 68.3 / 69.4, 2.97c, r609 (14/14) | 57.8 / 63.5, 4.73c | **97.8 / 98.6**, 4.42c, r46 (7/14) |
| grid | 5 | -0.04 | 96.9 / 99.0 | 94.6 / 95.3, 3.74c | **95.6 / 95.5**, 3.96c, r71 (8/14) | 56.0 / 59.3, 4.81c | 90.9 / 92.4, 4.63c, r173 (14/14) |
| iobt | 4 | -0.16 | 91.3 / 96.7 | 75.4 / 79.7, 3.42c | 73.6 / 75.2, 3.66c, r166 (11/14) | 81.9 / 85.0, 3.61c | **83.5 / 85.3**, 3.78c, r178 (14/14) |
| iobt | 4 | -0.04 | 96.7 / 96.7 | 92.5 / 93.1, 3.86c | **93.6 / 93.3**, 3.95c, r47 (8/14) | 79.8 / 82.8, 3.81c | 83.5 / 86.9, 3.78c, r130 (12/14) |
| iobt | 5 | -0.16 | 91.3 / 98.7 | 69.4 / 75.1, 3.77c | 68.3 / 72.4, 3.89c, r234 (13/14) | 90.8 / 93.3, 3.79c | **95.8 / 96.8**, 3.97c, r45 (10/14) |
| iobt | 5 | -0.04 | 97.5 / 98.7 | 94.6 / 94.8, 4.61c | **97.6 / 97.6**, 4.65c, r31 (4/14) | 88.9 / 91.6, 4.27c | 95.5 / 96.4, 4.03c, r122 (13/14) |

Online runs: 1016-1023 (no DAL), 1024-1031 (DAL warm). PPO: 1000-1007 (no DAL),
1008-1015 (DAL warm), in table order.

Base models (greedy): online tables mean on R0-R4 / PPO checkpoints final on C:

| env | k | cost | online base | PPO base |
|---|---:|---:|---:|---:|
| grid | 4 | -0.16 | 59.1 | 79.2 |
| grid | 4 | -0.04 | 88.4 | 60.4 |
| grid | 5 | -0.16 | 60.8 | 99.0 |
| grid | 5 | -0.04 | 95.9 | 91.7 |
| iobt | 4 | -0.16 | 70.8 | 87.0 |
| iobt | 4 | -0.04 | 87.0 | 84.2 |
| iobt | 5 | -0.16 | 60.9 | 98.6 |
| iobt | 5 | -0.04 | 98.3 | 98.5 |

## Findings

1. **Which learner wins depends on the sensor cost.** At cost -0.04 the online learner
   is best or tied in 4 of 4 settings (92-98%, within 0-4 points of the reference).
   At cost -0.16 PPO + DAL is best in 3 of 4 (76-98%); the online learner under-watches
   (2.7-3.8 cells) and stalls at 69-78%, and cannot even learn the equal-probability
   matrix at that cost (its base tables score 61-71%).
2. **PPO without DAL is weak on random matrices**, especially on the grid (45-58%).
   Training continually through the switches never settles.
3. **DAL warm gain tracks base-model headroom** (base minus learner alone):
   correlation 0.997 for PPO (8/8 signs agree), 0.78 for online (6/8), 0.95 overall.
   PPO + DAL gains +1.6 to +40 points; it is as good as the checkpoint it resets to.
4. **PPO base-model quality is erratic**: the same 1200-iteration recipe gave 60% (grid
   k4 -0.04) to 99% (grid k5 -0.16). That variance, not the detector, sets PPO + DAL's
   result.
5. **The detector over-fires** on stochastic movement: 31-609 restarts for 14 switches.
   With a good base model the extra resets are harmless; with a bad one they hurt
   (online at -0.16 loses 1-5 points).
6. Best per setting: grid k5 -0.16 PPO + DAL 97.8%; IoBT k5 -0.04 online + DAL 97.6%,
   matching the reward-optimal reference.

## Caveats

- One seed per run; base-model variance (finding 4) suggests seeds matter a lot for PPO.
- PPO delta (0.01) and online delta (1e-4) were not calibrated on these streams.
- PPO accuracy is from one greedy episode per iteration (noisy; averaged over 20).
- Grid PPO warm runs 832/835 from the earlier A/B experiment used algo.restore(),
  which crashed rollout workers on restore; RLlib restarted them, so those runs may
  have lost samples. Fixed here (set_weights in place, commit 38179b1).

---

# 10x10 grid (runs 1100-1115)

Same design; 10x10 random matrices (seed 20261006). PPO uses `--moore` (25-cell local
action window); the full 100-cell action space never learned (2.5% flat over 1200
iterations). Base models: online tables on C (greedy mean on R0-R4) / PPO on C (final).

| k | cost | ref best | online | online + DAL | PPO | PPO + DAL | base online / PPO |
|---:|---:|---:|---|---|---|---|---|
| 4 | -0.16 | 93.7 | 60.25 / 61.67, 2.89c | 54.50 / 52.72, 2.99c, r902 (14/14) | **67.75 / 68.27**, 3.96c | 53.81 / 53.23, 3.99c, r187 (14/14) | 48.2 / 49.9 |
| 4 | -0.04 | 93.7 | **85.46 / 86.93**, 3.75c | 77.29 / 76.71, 3.81c, r590 (14/14) | 71.00 / 70.96, 3.96c | 73.23 / 72.39, 3.96c, r214 (14/14) | 73.9 / 72.2 |
| 5 | -0.16 | 99.0 | 55.11 / 55.77, 2.79c | 45.05 / 43.70, 2.93c, r908 (14/14) | 91.28 / 93.13, 4.65c | **96.70 / 97.01**, 4.61c, r91 (8/14) | 44.2 / 94.5 |
| 5 | -0.04 | 99.0 | 94.15 / 95.43, 4.26c | **97.23 / 97.41**, 4.55c, r118 (12/14) | 92.86 / 94.33, 4.66c | **97.10 / 97.33**, 4.61c, r121 (12/14) | 95.7 / 95.6 |

Online: 1108-1111 (no DAL), 1112-1115 (DAL). PPO: 1100-1103 (no DAL), 1104-1107 (DAL).
Headroom (base - alone) vs DAL gain: correlation 0.98, sign correct 8/8.
