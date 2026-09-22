# Synthetic multi-loop run 802

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 3 of 10, reward_preset grid, sensor_rew -0.16
- Converged: True (best iteration 25 of 175)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   42.50% |         3.00 |
|    1 |   6 | 0-1-6-8-9-7 |    1.40% |         3.00 |
|    2 |   7 | 0-4-5-3-2-6-1 |   28.15% |         3.00 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   59.30% |         3.00 |
|    4 |   7 | 0-1-2-6-8-9-7 |   14.85% |         3.00 |
|    5 |   7 | 0-7-1-2-3-5-4 |   84.95% |         3.00 |
|    6 |   6 | 0-4-3-2-6-1 |   32.50% |         3.00 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   54.95% |         3.00 |
|    8 |   3 | 2-3-6 |   99.00% |         3.00 |
|    9 |   7 | 0-4-2-3-6-1-7 |   57.05% |         3.00 |

**Mean 47.47%** (min 1.40% on loop 1, max 99.00%, sd 28.56pp) at 3.00 sensors/step
