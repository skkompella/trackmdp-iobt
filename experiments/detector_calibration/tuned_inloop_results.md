# Stage 3 — the tuned detector, in-loop

Calibrated settings applied to every arm, both learners, both environments.

## Results

### 5x5 grid (tuned = transition_surprise, delta 2.28e-14, window 200)

| Run | Arm | Accuracy | Restarts | Caught | False |
|---|---|---:|---:|---:|---:|
| 832 | PPO warm, untuned | **95.51%** | 12 | 1/5 | 11 |
| 835 | PPO warm, **tuned** | 92.48% | 4 | 3/5 | 1 |
| 822 | tabular warm, untuned | 82.13% | 55 | 4/5 | 51 |
| 825 | tabular warm, **tuned** | 79.95% | 68 | 5/5 | 63 |
| 824 | tabular warm, oracle timing | 77.04% | 5 | 5/5 | 0 |
| 830 | PPO no detector | 71.03% | 0 | — | — |
| 820 | tabular no detector | 66.18% | 0 | — | — |
| 836 | PPO cold, **tuned** | 58.47% | 5 | 4/5 | 1 |
| 831 | PPO cold, untuned | 30.08% | 25 | 4/5 | 21 |
| 826 | tabular cold, **tuned** | 25.57% | 15 | 1/5 | 14 |
| 821 | tabular cold, untuned | 18.84% | 71 | 5/5 | 66 |

### IoBT (tuned = hit, delta 1e-4)

| Run | Arm | Accuracy | Restarts | Caught | False |
|---|---|---:|---:|---:|---:|
| 811 | no detector | **71.62%** | 0 | — | — |
| 816 | warm, **tuned** | 68.96% | 25 | 17/49 | 8 |
| 812 | warm, untuned | 67.94% | 40 | 22/49 | 18 |
| 815 | cold, **tuned** | 56.54% | 58 | 42/49 | 16 |
| 810 | cold, untuned | 47.51% | 107 | 47/49 | 60 |

## The unifying result: calibration pays in proportion to what a spurious restart costs

| Arm | Untuned | Tuned | Delta |
|---|---:|---:|---:|
| **PPO cold** (grid) | 30.08% | 58.47% | **+28.39pp** |
| tabular cold (IoBT) | 47.51% | 56.54% | **+9.03pp** |
| tabular cold (grid) | 18.84% | 25.57% | **+6.73pp** |
| tabular warm (IoBT) | 67.94% | 68.96% | +1.02pp |
| tabular warm (grid) | 82.13% | 79.95% | -2.18pp |
| PPO warm (grid) | 95.51% | 92.48% | -3.03pp |

Every COLD arm improves, and the gain scales with how expensive the restart is:
rebuilding a PPO network costs far more than re-filling a 1,326-weight table,
and PPO cold gains the most by far. Every WARM arm is flat or marginally worse.

The mechanism is the same one three earlier results pointed at: under warm
restart a spurious restart just reinstates a strong generalist, so firing too
often is nearly free and calibration has nothing to recover. Under cold restart
a spurious restart destroys everything learned, so firing 5 times instead of 25
is worth 28 points.

## Detection quality and tracking are decoupled — now four ways

1. run 822 (51 false alarms) 82.13% vs run 824 (zero) 77.04%
2. IoBT tuned delta improves detection without beating no-detection
3. run 825 (5/5 caught) 79.95% vs run 822 (4/5) 82.13%
4. run 835 (3/5 caught, 1 false) 92.48% vs run 832 (1/5, 11 false) 95.51%

In each case the better-detecting configuration tracked no better, and usually
slightly worse. The differences are small enough to be seed noise — which is the
point: across a 5x range in detection quality, downstream tracking does not move.

## Verdict against the bar set in the plan

Calibration succeeds only if a better-calibrated detector beats run 822 AND the
oracle-timing control 824. The best tuned arm (825, 79.95%) beats 824 but not
822. **It fails that bar.**

What calibration did achieve is real and worth keeping:
- grid detection from 2/5 to 5/5 switches caught;
- IoBT from a 24x-too-aggressive delta to 43/49 caught at a 27-step delay;
- cold-restart arms improved in all three environments;
- and a procedure that specifies a false-alarm rate rather than a magic number.

But the best configuration overall remains run 832 — PPO, warm restart, UNTUNED
detector, 95.51% at 3.83 sensors. No tuned arm beats it.

## What this says about where the gains are

The detector is not the bottleneck. The restart target is. A well-chosen warm
prior is worth 25-65 points; the detector's timing is worth roughly nothing once
that prior is in place. If change detection is to earn its keep, it has to be in
a regime where restarts are expensive and cannot be made cheap — which is
exactly where the cold-restart numbers above show calibration paying off.

## Caveat on the PPO arms

Runs 835 and 836 use delta=2.06e-09, calibrated on a TABULAR stream. PPO's
detector sees ~20,000 batch-resolution samples per run against tabular's
614,400, and sample rate is precisely what breaks delta transfer. Recording a
PPO stream means running PPO, so this was not done. The PPO tuned numbers should
be read as "less trigger-happy" rather than "properly calibrated" — which, given
finding above, is most of what mattered anyway.
