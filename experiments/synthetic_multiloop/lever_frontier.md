# Which lever controls activation? (runs 801-807)

Shared baseline: tl=1, `grid` reward preset, 10 loops (seed 20260921), PPO from
scratch, plateau early-stop, deterministic seeded eval at 20 episodes/loop on
the restored best checkpoint. Only the lever varies.

| Run | Lever | max_sens | sensor_rew | Mean acc | Sensors/step | sd | Worst loop |
|-----|-------|---------:|-----------:|---------:|-------------:|---:|-----------:|
| 801 | control | 6 | -0.16 | 57.02% | 5.69 | 20.62pp | 29.00% |
| 802 | max_sensors | 3 | -0.16 | 47.47% | **3.00** | 28.56pp | 1.40% |
| 803 | max_sensors | 2 | -0.16 | 31.53% | **1.75** | 30.10pp | 0.50% |
| 804 | max_sensors | 1 | -0.16 | 24.11% | **1.00** | 31.47pp | 0.25% |
| 805 | sensor_rew | 6 | -0.50 | 58.21% | 4.98 | 17.85pp | 33.45% |
| 806 | sensor_rew | 6 | -1.00 | 60.46% | 5.39 | 17.33pp | 29.25% |
| 807 | sensor_rew | 6 | -2.00 | 53.58% | 5.44 | 26.56pp | 0.30% |

## Verdict: max_sensors, decisively

**max_sensors is monotonic and exact.** 5.69 -> 3.00 -> 1.75 -> 1.00. At k=3 and
k=1 the cap binds precisely, so activation is set rather than merely nudged.

**sensor_rew barely moves activation and is non-monotonic.** Steepening it 12.5x
(-0.16 -> -2.00) buys only **-0.25 sensors/step**, and -1.00 uses MORE than
-0.50. It never drops below 4.98, so it never crosses the ambiguity threshold of
4 — it cannot create the regime we need at any setting tested.

### Why sensor_rew fails (the mechanism)

It appears on *both* sides of the trade. A missing-state step is charged
`n_grid * sensor_rew` = `16 * sensor_rew` regardless of the action taken, so
scaling the penalty scales the cost of BEING LOST in lockstep with the cost of
activating:

| sensor_rew | tracked hit, 6 on | tracked miss, 0 on | missing-state step |
|-----------:|------------------:|-------------------:|-------------------:|
| -0.16 | +0.04 | 0.00 | -2.56 |
| -0.50 | -2.00 | 0.00 | -8.00 |
| -1.00 | -5.00 | 0.00 | -16.00 |
| -2.00 | -11.00 | 0.00 | -32.00 |

Naively, at -2.00 doing nothing (0.00) beats a 6-sensor detection (-11.00) and
the policy should go inactive. It does not, because going inactive means losing
the object, and a missing step costs -32.00. The ratio is roughly preserved
under scaling, so the optimal policy is close to scale-invariant. **This refutes
the prediction made before the runs** that the sensor_rew arm would collapse
toward inactivity; the arm is flat, not degenerate.

## The control settles an older question

Run 801 is the same tl=1 + grid configuration as run 241 and is evaluated on the
same loops. Run 241 scores **99.00%** (the ceiling). Run 801 scores **57.02%**.
The only difference is the training target: run 241 trained on the topology's
random walk, run 801 on the 10 loops themselves.

Three candidate causes were proposed for run 800's weakness — time_limit, the
reward preset, and the training target. Runs 801-807 hold tl=1 and the grid
preset fixed, matching run 241 exactly, and still reach only 57%. **The training
target is the decisive cause.** Training on a narrow set of loops is markedly
worse than training on the transition structure that generates them, even though
the loops are what gets evaluated.

## Operating point for the online experiments

**max_sensors = 2** (run 803). It sits below the ambiguity threshold of 4, so
hedging is impossible at the worst nodes and the policy must infer which loop is
active — the regime online adaptation is meant to address. It leaves clear
headroom (PPO manages 31.53%) rather than the near-floor of k=1, and the tabular
learner is known to reach the ceiling in the easy regime, so a poor result there
will be attributable to the regime rather than the implementation.

k=1 (run 804) is held in reserve as the maximally demanding setting. Note it
needed best-iteration 275 against 20-80 for the others, i.e. far more training —
a useful signal that commitment is genuinely harder to learn.
