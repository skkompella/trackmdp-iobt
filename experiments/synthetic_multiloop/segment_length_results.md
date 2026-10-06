# IoBT online runs: longer segments, greedy evaluation (runs 854-861)

Same setup as runs 850-853 (10 loops x 5 passes, tabular SARSA, delta 1e-4,
equal-probability base models), but with 200 or 1000 episodes per loop instead of
50, and with `--greedy-eval-every` on: every (episodes_per_loop / 10) episodes the
current table is scored greedily (10 episodes) on the current loop. "1st chk" is
the first greedy checkpoint after each switch (20 episodes in at 200/loop, 100 in
at 1000/loop), averaged over the 49 switches.

| run | eps/loop | k | arm | greedy mean | segment end | 1st chk | sensors | restarts | caught | false |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 854 | 200 | 6 | none | 98.72% | 99.00% | 98.66% | 4.17 | 0 | - | - |
| 855 | 200 | 6 | warm | 98.59% | 98.80% | 98.80% | 4.90 | 24 | 17/49 | 7 |
| 856 | 200 | 5 | none | 97.76% | 98.52% | 97.06% | 3.84 | 0 | - | - |
| 857 | 200 | 5 | warm | 96.88% | 97.32% | 96.47% | 4.53 | 30 | 15/49 | 15 |
| 858 | 1000 | 6 | none | 98.47% | 99.00% | 95.22% | 3.33 | 0 | - | - |
| 859 | 1000 | 6 | warm | 98.97% | 99.00% | 99.00% | 4.90 | 30 | 22/49 | 8 |
| 860 | 1000 | 5 | none | 98.24% | 99.00% | 94.21% | 3.01 | 0 | - | - |
| 861 | 1000 | 5 | warm | 98.42% | 99.00% | 98.09% | 4.53 | 72 | 34/49 | 38 |

Findings:

- Scored greedily, the learner on its own reaches the 99% ceiling by the end of
  segments once each loop gets 200+ episodes. The old 50-episode numbers (scored
  while exploring) understated it by roughly 6-8 points.
- Warm restart does not raise mean accuracy materially (-0.9 to +0.5 points).
- Warm restart removes the post-switch dip at 1000 episodes/loop (first checkpoint
  95.22% -> 99.00% at k=6, 94.21% -> 98.09% at k=5). The no-detector dip is deeper
  at 1000 than at 200 because the learner has specialised further on the previous
  loop (3.0-3.3 sensors vs 3.8-4.2).
- Warm restart costs 0.7-1.5 extra sensors/step: the restarted agent stays close to
  the base model's hedge (~5 sensors at k=6) and does not specialise back down
  within a segment, while the no-detector learner falls to ~3 sensors.

Single seed per arm; greedy checkpoints use 10 episodes each.
