# Synthetic multi-loop run 804

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 1 of 10, reward_preset grid, sensor_rew -0.16
- Converged: True (best iteration 275 of 415)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |    6.95% |         1.00 |
|    1 |   6 | 0-1-6-8-9-7 |    0.55% |         1.00 |
|    2 |   7 | 0-4-5-3-2-6-1 |   13.75% |         1.00 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   69.20% |         1.00 |
|    4 |   7 | 0-1-2-6-8-9-7 |   14.45% |         1.00 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         1.00 |
|    6 |   6 | 0-4-3-2-6-1 |    0.75% |         1.00 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   21.75% |         1.00 |
|    8 |   3 | 2-3-6 |    0.25% |         1.00 |
|    9 |   7 | 0-4-2-3-6-1-7 |   14.40% |         1.00 |

**Mean 24.11%** (min 0.25% on loop 8, max 99.00%, sd 31.47pp) at 1.00 sensors/step
