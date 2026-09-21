# Existing checkpoints evaluated on the run-800 loop set

Same 10 loops (seed 20260921), same evaluator, dedicated seeded env, no training.
Each checkpoint evaluated at ITS OWN time_limit — the observation space encodes
time_limit_max, so a tl=1 policy cannot be loaded into a tl=3 env.

| Model | tl | Trained on | Mean acc | sd | Sensors/step |
|-------|----|------------|---------:|---:|-------------:|
| **run 241** | 1 | topo random walk, 500 iters | **99.00%** | 0.00pp | 4.26 |
| run 227 | 1 | topo random walk, 400 iters | 91.54% | 14.88pp | 4.86 |
| run 800 | 3 | these 10 loops, plateau-stopped at 170 | 63.49% | 18.62pp | 5.51 |
| run 620 | 3 | live audio+camera fusion (session 165739) | 22.48% | 9.69pp | 4.27 |
| run 611 | 3 | live audio, tl=3 base | 17.99% | 10.20pp | 2.37 |

## 99.00% is the measurement ceiling, not a score

evaluate_policy starts every episode in missing_state. That first step is counted
in the denominator (total_steps) but can never count in the numerator, because
total_found only increments when `not was_missing`. So the maximum achievable
accuracy is (max_ep_steps - 1) / max_ep_steps.

Verified by sweeping episode length against run 241:

| episode length | run 241 accuracy | (steps-1)/steps |
|---------------:|-----------------:|----------------:|
| 50 | 98.00% | 98.00% |
| 100 | 99.00% | 99.00% |
| 200 | 99.50% | 99.50% |
| 400 | 99.75% | 99.75% |

Exact at every length. Run 241 never misses a single trackable step on any of the
10 loops. It is a perfect tracker here, not a 99% one.

## What this says

The multi-loop task is EASY for Track-MDP. A policy trained long enough on the
topology's random walk solves it perfectly and zero-shot — it never saw these
loops, but every loop is a path through IOBT_MAP, so learning the transition
structure covers all of them at once. The 400 -> 500 iteration difference between
run 227 and run 241 is the difference between uneven (91.54%, sd 14.88) and
perfect (99.00%, sd 0.00), so the last stretch of training is what buys
consistency across loops.

Run 800's 63.49% is therefore a property of its configuration, not of the task:

1. **time_limit 3 vs 1.** Every tl=3 checkpoint scores badly here (800: 63%,
   620: 22%, 611: 18%) while both tl=1 topo models score 91-99%.
2. **Reward constants.** Run 241's env (TopoIoBTEnv -> iobt_env) inherits the
   grid_env defaults: tracking_rew 1.0, miss 0, sensor_rew -0.16. Run 800 used
   RealIoBTEnv's live-tuned values (1.5 / -0.5 / -0.25), chosen for comparability
   with the live runs. Under those, detecting with all 6 sensors on nets exactly
   0.00, so PPO is pushed to narrow activation it cannot afford.
3. **Training target.** The topo random walk covers every valid transition; 10
   fixed loops are a narrower and apparently harder optimization target.

Run 620 and 611 scoring 18-22% is its own finding: the live-data champions have
overfit to one session's trajectory and frugal activation, and carry nothing over
to clean synthetic loops.

Note: run 241's on-disk checkpoint is the FINAL iteration (500), not its recorded
best (439), because of the overwrite bug fixed in 7320e69. It hits the ceiling
anyway, so that run had no meaningful decay.

## Next

Retrain the multi-loop agent at tl=1 with grid_env reward constants and a 500+
iteration budget. If it reaches the ceiling, the diagnosis above is confirmed and
run 800's config was simply wrong.
