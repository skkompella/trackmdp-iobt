# 5x5 grid, two opposed transition matrices: online vs PPO x change detection

Object drifts north-east under matrix A, south-west under B, on the schedule
A,B,A,B,A,B. Every arm gets **102,400 environment steps per segment** — 1,024
tabular episodes of 100 steps, or 200 PPO iterations at batch 512. 5x5 grid,
`max_sensors=6`, `time_limit=1`, `sensor_rew=-0.16`. Ceiling is 99.00%.

## The 2x2 that was asked for

| | no detection | detection, COLD restart | detection, WARM restart |
|---|---:|---:|---:|
| **tabular online** | 66.18% | 18.84% | **82.13%** |
| **PPO** | 71.03% | 30.08% | **95.51%** |

## Full results

| Run | Learner | Arm | Mean acc | Final | Sensors | Restarts | Caught | False |
|-----|---------|-----|---------:|------:|--------:|---------:|-------:|------:|
| 832 | PPO | CD warm | **95.51%** | **98.67%** | **3.83** | 12 | 1/5 | 11 |
| 822 | tabular | CD warm | 82.13% | 83.39% | 5.59 | 55 | 4/5 | 51 |
| 824 | tabular | warm, ORACLE timing | 77.04% | 79.83% | 5.20 | 5 | 5/5 | 0 |
| 830 | PPO | no detector | 71.03% | 81.42% | 4.55 | 0 | — | — |
| 820 | tabular | no detector | 66.18% | 72.73% | 4.64 | 0 | — | — |
| 823 | tabular | cold, ORACLE timing | 47.56% | 65.35% | 5.38 | 5 | 5/5 | 0 |
| 831 | PPO | CD cold | 30.08% | 26.59% | 5.99 | 25 | 4/5 | 21 |
| 821 | tabular | CD cold | 18.84% | 18.37% | 5.72 | 71 | 5/5 | 66 |

## Three findings

### 1. Detection+restart helps BOTH learners — but only with a warm target

Warm beats its own no-detection baseline by **+15.95pp** (tabular) and
**+24.48pp** (PPO). Cold *loses* by 47pp and 41pp respectively. This is the
same conclusion the multi-loop post-mortem reached, now with the sign flipped
on the headline: what you restart **to** decides everything.

Restart to nothing is harmful even when perfectly timed — the cold oracle (run
823, 5/5 caught, zero false alarms) still trails the baseline, 47.56% vs 66.18%.

### 2. PPO beats the tabular learner here, reversing the IoBT result

PPO wins every matched arm: 71.03 vs 66.18 (no detection), 95.51 vs 82.13
(warm), 30.08 vs 18.84 (cold). On the 10-node IoBT loops the tabular learner was
2.3x PPO. The difference is problem size and representation: 25 cells and 51
states give PPO's network something to generalise across, while the factored
tabular learner is linear in cells and had to drop eligibility traces entirely.

Run 832 is the best result of the project on this task: **98.67% final at 3.83
sensors** — the highest accuracy AND the lowest energy of any arm. It starts
from the restored hedge and specialises *downward* in sensor count while holding
accuracy.

### 3. The benefit is NOT detection. It is periodic restoration.

Run 832 caught **1 of 5** switches, with 11 false alarms, and still won by
24.5pp. That prompted control run 824: warm restart fired at the 5 *known*
switches instead of on detection.

```
  822  warm, GLR timing    (4/5 caught, 51 false, 55 restarts)   82.13%
  824  warm, ORACLE timing (5/5 caught,  0 false,  5 restarts)   77.04%
```

**The worse-timed detector scored higher.** Detection timing is not the
mechanism. Each restart resets the learner to a 99% generalist before it can
specialise into something worse, so spurious restarts *help*. A plain "reset to
the prior every N episodes", with no detector at all, would likely match this.

That is a deflating result for the change-detection hypothesis and it should be
stated plainly: across two experiments now, the detector has never been shown to
earn its keep. What earns its keep is having a good generalist to fall back to.

## Structural caveat

The equal-weight agent scores **99.00% on A, B and C alike**: with 6 sensors it
covers all <=5 destinations, and every matrix's support is a subset of
{stay} u neighbours. So a universal hedge exists here, as it did for the loops —
and the warm arms are, in effect, restoring that hedge.

Specialised agents genuinely conflict (an A-trained agent scores 87.52% on A and
18.48% on B), so the regimes are real. But `max_sensors <= 4` would remove the
hedge, since 4 sensors cannot cover 5 destinations, and that is the version of
this experiment where commitment is actually forced. Untested.

## Caveats

- **PPO detects at batch resolution** (512 steps) against the tabular learner's
  per step, because PPO's rollouts live in worker processes. Inherent.
- **Latency favours the wrong arms**, as before: it is measured against each
  segment's own plateau, so a learner that settles quickly into a bad plateau
  scores well. Read accuracy.
- **One detector tuning** was tested (delta 0.01, min_samples 30). Given finding
  3, tuning it is unlikely to be the interesting lever.
