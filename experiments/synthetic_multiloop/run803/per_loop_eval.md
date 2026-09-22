# Synthetic multi-loop run 803

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 2 of 10, reward_preset grid, sensor_rew -0.16
- Converged: True (best iteration 80 of 230)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   26.95% |         1.72 |
|    1 |   6 | 0-1-6-8-9-7 |    1.25% |         1.84 |
|    2 |   7 | 0-4-5-3-2-6-1 |    7.15% |         1.79 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   69.20% |         1.60 |
|    4 |   7 | 0-1-2-6-8-9-7 |   28.20% |         1.86 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         1.43 |
|    6 |   6 | 0-4-3-2-6-1 |    0.50% |         1.83 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   21.65% |         1.78 |
|    8 |   3 | 2-3-6 |   46.20% |         1.77 |
|    9 |   7 | 0-4-2-3-6-1-7 |   15.15% |         1.85 |

**Mean 31.53%** (min 0.50% on loop 6, max 99.00%, sd 30.10pp) at 1.75 sensors/step
