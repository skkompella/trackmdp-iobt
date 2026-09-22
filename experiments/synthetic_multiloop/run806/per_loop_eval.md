# Synthetic multi-loop run 806

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 6 of 10, reward_preset grid, sensor_rew -1.0
- Converged: True (best iteration 135 of 285)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   29.25% |         5.70 |
|    1 |   6 | 0-1-6-8-9-7 |   65.30% |         4.52 |
|    2 |   7 | 0-4-5-3-2-6-1 |   55.90% |         5.71 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   69.30% |         5.21 |
|    4 |   7 | 0-1-2-6-8-9-7 |   56.65% |         5.14 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         5.16 |
|    6 |   6 | 0-4-3-2-6-1 |   65.60% |         5.67 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   55.00% |         5.34 |
|    8 |   3 | 2-3-6 |   66.00% |         5.99 |
|    9 |   7 | 0-4-2-3-6-1-7 |   42.55% |         5.43 |

**Mean 60.46%** (min 29.25% on loop 0, max 99.00%, sd 17.33pp) at 5.39 sensors/step
