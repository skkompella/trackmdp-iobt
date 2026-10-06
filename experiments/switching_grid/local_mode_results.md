# Grid with the "local" exact-subset learner (runs 910-917)

Switching A,B x 3 passes x 1000 episodes, no detector, greedy evaluation (10
episodes per checkpoint, every 50 episodes). Tabular SARSA over subsets of the
cells reachable since the object was last seen (5 cells at delay 0, 13 at delay 1).
"cells/trk" = cells switched on per tracked step at segment end, last pass.

| run | k | cost | hyperparams | mean | @50 | @1000 | pass-2 end | cells/trk | histogram (cells: share of tracked steps) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 910 | 4 | -0.16 | grid (a.02 g.9 l0 e.15 opt5) | 81.92% | 77.93% | 84.48% | 84.50% | 2.88 | 1:4% 2:20% 3:58% 4:18% |
| 911 | 4 | -0.16 | iobt (a.1 g.95 l.8 e.1 opt0) | 87.73% | 93.08% | 89.77% | 88.15% | 2.90 | 1:8% 2:5% 3:78% 4:10% |
| **912** | 4 | -0.04 | grid | **95.22%** | 93.35% | 95.95% | 94.80% | 3.38 | 3:62% 4:38% |
| 913 | 4 | -0.04 | iobt | 81.91% | 83.28% | 84.58% | 91.85% | 3.29 | 2:9% 3:51% 4:39% |
| 914 | 5 | -0.16 | grid | 81.01% | 77.20% | 80.47% | 79.50% | 2.77 | 0:3% 1:3% 2:22% 3:61% 4:9% 5:2% |
| 915 | 5 | -0.16 | iobt | 88.16% | 88.95% | 93.10% | 93.80% | 3.16 | 1:1% 2:8% 3:70% 4:17% 5:5% |
| **916** | 5 | -0.04 | grid | **94.74%** | 94.68% | 95.42% | 95.00% | 3.35 | 3:67% 4:31% 5:2% |
| 917 | 5 | -0.04 | iobt | 79.35% | 74.35% | 72.60% | 79.75% | 2.65 | 2:46% 3:43% 4:11% 5:1% |

Factored baselines (cost -0.16, grid hyperparams): 871 k=4 73.49% mean / 86.80%
pass-2 end; 872 k=5 73.19% / 89.40%. Best factored at cost -0.04: 72-74% mean.
Oracle ceilings: 92.49% at cost -0.16, 98.29% at cost -0.04.

- Exact local subsets fix the credit problem: at cost -0.04 mean accuracy goes
  from ~73% (factored) to ~95%, and it is already ~93-95% 50 episodes after each
  switch, so adaptation is fast.
- k=5 adds nothing over k=4: the reward optimum at cost -0.04 uses 4 cells.
- Remaining gap (95% vs 98.3%): the agent switches on 3 cells on ~2/3 of tracked
  steps. At cost -0.04 the 4th cell (p=0.05) is near break-even, so its value is
  hard to separate from 3 cells.
- The IoBT hyperparameters (alpha 0.1, lambda 0.8) are unstable here at cost -0.04.
- Single seed per configuration.
