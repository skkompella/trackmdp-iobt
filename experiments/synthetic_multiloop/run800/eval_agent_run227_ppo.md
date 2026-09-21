# run227 evaluated on the run-800 loop set

- Checkpoint: `/home/hari/Documents/Projects/trackmdp-iobt-gnhf-worktrees/objective-push-track-876cb3/runs/agent_run227_ppo` (no training)
- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 6 of 10, sensor_rew -0.25
- Eval: 10 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   99.00% |         4.87 |
|    1 |   6 | 0-1-6-8-9-7 |   99.00% |         4.68 |
|    2 |   7 | 0-4-5-3-2-6-1 |   99.00% |         4.87 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   99.00% |         4.91 |
|    4 |   7 | 0-1-2-6-8-9-7 |   99.00% |         4.59 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         5.01 |
|    6 |   6 | 0-4-3-2-6-1 |   49.50% |         5.01 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   88.00% |         5.01 |
|    8 |   3 | 2-3-6 |   99.00% |         4.68 |
|    9 |   7 | 0-4-2-3-6-1-7 |   84.90% |         5.01 |

**Mean 91.54%** (min 49.50% on loop 6, max 99.00%, sd 14.88pp) at 4.86 sensors/step
