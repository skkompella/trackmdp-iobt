# Online learning curves vs sensor budget (IoBT runs 862-863 + 858/860; grid runs 870-873)

No detector, greedy evaluation (10 episodes per checkpoint). IoBT: 10 loops x 5
passes x 1000 episodes, tabular SARSA (subset), checkpoints every 100 episodes;
k=5/6 are runs 860/858. Grid: matrices A,B x 3 passes x 1000 episodes, factored
tabular (alpha 0.02, eps 0.15), checkpoints every 50 episodes.

## IoBT -- greedy accuracy vs episodes into a segment (mean over 50 segments)

| episode | k=3 | k=4 | k=5 | k=6 |
|---:|---:|---:|---:|---:|
| 100 | 92.26% | 93.39% | 94.02% | 95.30% |
| 200 | 98.43% | 96.91% | 97.14% | 98.00% |
| 400 | 98.72% | 98.39% | 99.00% | 99.00% |
| 1000 | 98.33% | 98.06% | 99.00% | 99.00% |

Sensors/step at segment end: k=3 2.17, k=4 2.58 (not recorded for k=5/6).

## Grid -- greedy accuracy vs episodes into a segment (mean over 6 segments)

| episode | k=3 | k=4 | k=5 | k=6 |
|---:|---:|---:|---:|---:|
| 50 | 54.10% | 61.43% | 52.65% | 58.67% |
| 200 | 63.57% | 69.03% | 54.93% | 68.68% |
| 400 | 66.28% | 67.02% | 69.00% | 76.33% |
| 600 | 76.37% | 76.62% | 84.68% | 82.33% |
| 800 | 79.03% | 85.57% | 82.68% | 85.47% |
| 1000 | 76.93% | 80.73% | 84.92% | 86.18% |

End-of-segment accuracy by pass:

| pass | k=3 | k=4 | k=5 | k=6 |
|---:|---:|---:|---:|---:|
| 0 | 67.30% | 71.60% | 77.60% | 82.65% |
| 1 | 78.95% | 83.80% | 87.75% | 90.15% |
| 2 | 84.55% | 86.80% | 89.40% | 85.75% |

## Reading

- IoBT: accuracy saturates at the 99% ceiling within 200-400 episodes at every k.
  Loops are deterministic, so the learner converges to a specialist; more steps add
  nothing beyond that point.
- Grid: accuracy is still rising at 1000 episodes and across passes, and ends at
  77-90% -- well below the 99% the matrix-C base model reaches. More training
  should keep helping here.
- Single seed per k; 10-episode checkpoints are noisy (a few points).
