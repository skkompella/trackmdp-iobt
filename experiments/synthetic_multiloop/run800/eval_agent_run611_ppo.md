# run611 evaluated on the run-800 loop set

- Checkpoint: `/home/hari/Documents/Projects/trackmdp-iobt-gnhf-worktrees/objective-push-track-876cb3/runs/agent_run611_ppo` (no training)
- Loops: 10 sampled with seed 20260921
- time_limit 3, max_sensors 6 of 10, sensor_rew -0.25
- Eval: 10 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   27.40% |         2.86 |
|    1 |   6 | 0-1-6-8-9-7 |   31.70% |         1.73 |
|    2 |   7 | 0-4-5-3-2-6-1 |    8.20% |         2.38 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |    9.50% |         2.60 |
|    4 |   7 | 0-1-2-6-8-9-7 |    7.10% |         2.40 |
|    5 |   7 | 0-7-1-2-3-5-4 |   28.10% |         2.38 |
|    6 |   6 | 0-4-3-2-6-1 |   17.40% |         2.53 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   10.90% |         2.28 |
|    8 |   3 | 2-3-6 |   32.40% |         2.05 |
|    9 |   7 | 0-4-2-3-6-1-7 |    7.20% |         2.53 |

**Mean 17.99%** (min 7.10% on loop 4, max 32.40%, sd 10.20pp) at 2.37 sensors/step
