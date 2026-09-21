# Synthetic multi-loop run 800

- Loops: 10 sampled with seed 20260921
- time_limit 3, max_sensors 6 of 10, sensor_rew -0.25
- Converged: True (best iteration 20 of 170)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   28.30% |         5.71 |
|    1 |   6 | 0-1-6-8-9-7 |   82.35% |         5.17 |
|    2 |   7 | 0-4-5-3-2-6-1 |   42.30% |         5.71 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   89.00% |         5.40 |
|    4 |   7 | 0-1-2-6-8-9-7 |   56.50% |         5.71 |
|    5 |   7 | 0-7-1-2-3-5-4 |   84.90% |         5.58 |
|    6 |   6 | 0-4-3-2-6-1 |   49.15% |         5.67 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   65.80% |         5.34 |
|    8 |   3 | 2-3-6 |   66.00% |         5.34 |
|    9 |   7 | 0-4-2-3-6-1-7 |   70.60% |         5.43 |

**Mean 63.49%** (min 28.30% on loop 0, max 89.00%, sd 18.62pp) at 5.51 sensors/step
