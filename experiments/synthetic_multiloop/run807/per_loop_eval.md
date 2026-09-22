# Synthetic multi-loop run 807

- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 6 of 10, reward_preset grid, sensor_rew -2.0
- Converged: True (best iteration 20 of 170)
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   28.65% |         5.45 |
|    1 |   6 | 0-1-6-8-9-7 |   33.55% |         5.67 |
|    2 |   7 | 0-4-5-3-2-6-1 |   70.75% |         5.44 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   79.10% |         5.60 |
|    4 |   7 | 0-1-2-6-8-9-7 |   70.65% |         5.29 |
|    5 |   7 | 0-7-1-2-3-5-4 |   84.65% |         5.43 |
|    6 |   6 | 0-4-3-2-6-1 |   81.45% |         5.35 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   44.20% |         5.45 |
|    8 |   3 | 2-3-6 |    0.30% |         5.27 |
|    9 |   7 | 0-4-2-3-6-1-7 |   42.50% |         5.43 |

**Mean 53.58%** (min 0.30% on loop 8, max 84.65%, sd 26.56pp) at 5.44 sensors/step
