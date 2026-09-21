# run620 evaluated on the run-800 loop set

- Checkpoint: `/home/hari/Documents/Projects/trackmdp-iobt-gnhf-worktrees/objective-push-track-876cb3/runs/agent_run620_ppo` (no training)
- Loops: 10 sampled with seed 20260921
- time_limit 3, max_sensors 6 of 10, sensor_rew -0.25
- Eval: 10 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   15.60% |         3.61 |
|    1 |   6 | 0-1-6-8-9-7 |   16.60% |         5.13 |
|    2 |   7 | 0-4-5-3-2-6-1 |   15.20% |         3.60 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   39.10% |         4.40 |
|    4 |   7 | 0-1-2-6-8-9-7 |   26.90% |         4.85 |
|    5 |   7 | 0-7-1-2-3-5-4 |   14.30% |         4.00 |
|    6 |   6 | 0-4-3-2-6-1 |   17.40% |         4.81 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   11.00% |         4.76 |
|    8 |   3 | 2-3-6 |   33.30% |         3.03 |
|    9 |   7 | 0-4-2-3-6-1-7 |   35.40% |         4.52 |

**Mean 22.48%** (min 11.00% on loop 7, max 39.10%, sd 9.69pp) at 4.27 sensors/step
