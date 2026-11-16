# Phase 3 — accuracy arms (mean ± std over seeds 1-3)

Tabular SARSA, k = 5, cost -0.04, R0..R4 x 3 passes (14 switches), 1000 episodes/segment. rho = 0.05 for transition arms. Warm = restore the equal-probability (C) base table.

| env | arm | greedy mean % | greedy end % | sensors/step | caught/14 | false alarms | restarts | probe frac |
|---|---|---|---|---|---|---|---|---|
| grid | none | 94.64 ± 0.05 | 95.37 ± 0.36 | 3.84 ± 0.00 | 0.0 | 0.0 | 0.0 | 0.0000 |
| grid | oracle-cold | 90.99 ± 0.17 | 92.66 ± 0.69 | 3.55 ± 0.02 | 0.0 | 0.0 | 14.0 | 0.0000 |
| grid | transition-cold | 91.24 ± 0.34 | 93.17 ± 0.34 | 3.57 ± 0.04 | 14.0 | 0.0 | 14.0 | 0.0010 |
| grid | oracle-warm | 95.20 ± 0.12 | 94.99 ± 0.26 | 3.86 ± 0.00 | 0.0 | 0.0 | 14.0 | 0.0000 |
| grid | transition-warm | 94.97 ± 0.28 | 94.61 ± 0.60 | 3.84 ± 0.02 | 14.0 | 0.0 | 14.0 | 0.0002 |
| iobt | none | 94.73 ± 0.27 | 95.20 ± 0.39 | 4.55 ± 0.06 | 0.0 | 0.0 | 0.0 | 0.0000 |
| iobt | oracle-cold | 88.56 ± 0.34 | 89.80 ± 0.07 | 4.53 ± 0.04 | 0.0 | 0.0 | 14.0 | 0.0000 |
| iobt | transition-cold | 88.48 ± 0.58 | 89.76 ± 0.38 | 4.53 ± 0.01 | 14.0 | 0.0 | 14.0 | 0.0048 |
| iobt | oracle-warm | 97.69 ± 0.06 | 97.78 ± 0.12 | 4.64 ± 0.01 | 0.0 | 0.0 | 14.0 | 0.0000 |
| iobt | transition-warm | 97.64 ± 0.14 | 97.77 ± 0.14 | 4.63 ± 0.01 | 14.0 | 0.0 | 14.0 | 0.0008 |

"caught" and "false alarms" apply to the transition arms only (oracle arms restart at the known switches and raise no alarms).

- The detector matches the oracle: every transition run caught 14/14 switches with 0 false alarms, and its accuracy is within 0.25 points of the oracle arm with the same restart strategy in all four (env, strategy) pairs.
- Whether restarting helps is a property of the restart, not the detector: cold restarts cost 3.4 (grid) and 6.2 (IoBT) points against never restarting; warm restarts to the C base table gain 0.3 (grid) and 2.9 (IoBT) points.
