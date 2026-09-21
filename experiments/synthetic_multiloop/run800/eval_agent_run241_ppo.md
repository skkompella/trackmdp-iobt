# run241 evaluated on the run-800 loop set

- Checkpoint: `/home/hari/Documents/Projects/trackmdp-iobt-gnhf-worktrees/objective-push-track-876cb3/runs/agent_run241_ppo` (no training)
- Loops: 10 sampled with seed 20260921
- time_limit 1, max_sensors 6 of 10, sensor_rew -0.25
- Eval: 20 episodes per loop, fixed seed 12345, dedicated env

| Loop | Len | Nodes | Accuracy | Sensors/step |
|------|-----|-------|----------|--------------|
|    0 |   7 | 0-4-5-3-6-2-1 |   99.00% |         4.30 |
|    1 |   6 | 0-1-6-8-9-7 |   99.00% |         4.02 |
|    2 |   7 | 0-4-5-3-2-6-1 |   99.00% |         4.30 |
|    3 |  10 | 0-7-9-8-6-1-2-3-5-4 |   99.00% |         4.12 |
|    4 |   7 | 0-1-2-6-8-9-7 |   99.00% |         4.02 |
|    5 |   7 | 0-7-1-2-3-5-4 |   99.00% |         4.44 |
|    6 |   6 | 0-4-3-2-6-1 |   99.00% |         4.52 |
|    7 |   9 | 0-4-2-3-6-8-9-7-1 |   99.00% |         4.24 |
|    8 |   3 | 2-3-6 |   99.00% |         4.02 |
|    9 |   7 | 0-4-2-3-6-1-7 |   99.00% |         4.58 |

**Mean 99.00%** (min 99.00% on loop 0, max 99.00%, sd 0.00pp) at 4.26 sensors/step
