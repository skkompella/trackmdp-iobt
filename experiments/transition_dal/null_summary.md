Tolerance: 300 episodes

| run | env | detector | mode | rho | seed | caught | false alarms | median delay (ep) | greedy mean % | greedy end % | sensors/step | probe frac | restarts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_transition_monitor_s1 | grid | transition | monitor | 0.20 | 1 | 0/0 | 0 | - | 93.40 | 93.35 | 3.63 | 0.01 | 0 |
| grid_transition_monitor_s2 | grid | transition | monitor | 0.20 | 2 | 0/0 | 0 | - | 93.55 | 93.93 | 3.65 | 0.01 | 0 |
| grid_transition_monitor_s3 | grid | transition | monitor | 0.20 | 3 | 0/0 | 0 | - | 93.58 | 94.11 | 3.63 | 0.01 | 0 |
| iobt_transition_monitor_s1 | iobt | transition | monitor | 0.20 | 1 | 0/0 | 0 | - | 93.42 | 94.08 | 4.40 | 0.04 | 0 |
| iobt_transition_monitor_s2 | iobt | transition | monitor | 0.20 | 2 | 0/0 | 0 | - | 94.27 | 94.47 | 4.30 | 0.05 | 0 |
| iobt_transition_monitor_s3 | iobt | transition | monitor | 0.20 | 3 | 0/0 | 0 | - | 94.15 | 94.05 | 4.37 | 0.04 | 0 |

## Ordered vs shuffled replay (examples/replay_streams.py, 3 shuffles)

| run | samples | dF=0.01 ordered / shuffled | dF=0.1 | dF=0.5 |
|---|---|---|---|---|
| grid_transition_monitor_s1 | 4,634,868 | 0 / [0, 0, 0] | 0 / [0, 0, 0] | 0 / [1, 1, 0] |
| grid_transition_monitor_s2 | 4,656,912 | 0 / [0, 0, 0] | 0 / [0, 0, 0] | 1 / [0, 0, 0] |
| grid_transition_monitor_s3 | 4,643,517 | 0 / [1, 1, 0] | 0 / [1, 1, 0] | 1 / [1, 1, 0] |
| iobt_transition_monitor_s1 | 4,279,959 | 0 / [0, 0, 0] | 0 / [0, 1, 0] | 0 / [0, 1, 0] |
| iobt_transition_monitor_s2 | 4,381,041 | 0 / [0, 0, 0] | 0 / [0, 1, 1] | 0 / [0, 2, 2] |
| iobt_transition_monitor_s3 | 4,424,041 | 0 / [0, 0, 0] | 0 / [0, 0, 0] | 0 / [0, 1, 0] |

Gate (plan Task 8): at most 1 alarm per env over 3 seeds (observed 0 and 0), and |ordered - mean(shuffled)| <= 2 sqrt(mean) + 2 at dF 0.1 and 0.5 for every run: **PASS**.
Old pooled hit stream on a comparable stationary grid stream: one alarm every 3,700 steps at delta 0.01 ordered, none shuffled.
