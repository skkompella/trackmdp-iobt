# Synthetic multi-loop run 805

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 6 of 10, reward_preset grid, sensor_rew -0.5
- Converged: True (best iteration 60 of 175)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   55.65% |         4.87 |
|    1 |   6 | 0-1-6-8-9-7 |   33.45% |         5.16 |
|    2 |   7 | 0-4-5-3-2-6-1 |   70.70% |         4.73 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   59.70% |         5.10 |
|    4 |   7 | 0-1-2-6-8-9-7 |   42.90% |         5.01 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         4.87 |
|    6 |   6 | 0-4-3-2-6-1 |   66.00% |         4.85 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   66.00% |         4.90 |
|    8 |   3 | 2-3-6 |   46.20% |         5.44 |
|    9 |   7 | 0-4-2-3-6-1-7 |   42.55% |         4.87 |

**Mean 58.21%** (min 33.45% on loop 1, max 99.00%, sd 17.85pp) at 4.98 sensors/step
