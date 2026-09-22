# Synthetic multi-loop run 801

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 6 of 10, reward_preset grid, sensor_rew -0.16
- Converged: True (best iteration 20 of 170)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   29.00% |         5.86 |
|    1 |   6 | 0-1-6-8-9-7 |   32.65% |         5.52 |
|    2 |   7 | 0-4-5-3-2-6-1 |   70.65% |         5.72 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   69.60% |         5.60 |
|    4 |   7 | 0-1-2-6-8-9-7 |   42.40% |         5.57 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         5.72 |
|    6 |   6 | 0-4-3-2-6-1 |   65.95% |         5.67 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   44.05% |         5.67 |
|    8 |   3 | 2-3-6 |   46.20% |         5.67 |
|    9 |   7 | 0-4-2-3-6-1-7 |   70.65% |         5.86 |

**Mean 57.02%** (min 29.00% on loop 0, max 99.00%, sd 20.62pp) at 5.69 sensors/step
