# IoBT with the equal-probability base model

## The problem this fixes

The IoBT warm-restart arms were restoring `prior_k2.npy`, trained on the pooled
**loops**. That is the analogue of training the grid's base model on matrices A
and B rather than on C. It scored 70.35% — *worse* than the 71.62% the learner
reached unaided — so every restart was a downgrade, and warm restart lost.

The correct base model is the IoBT analogue of matrix C: an **equal-probability
random walk** over IOBT_MAP, staying put or moving to any adjacent node with
equal probability. This is the movement model run 241 was trained on.

## Sensor budget decides whether a good base model exists

IOBT_MAP node degrees run 2 to 5, so the walk has up to **6** destinations
(node 4: five neighbours plus stay). The loop family has max **4** successors
per node. Training the equal-probability agent and evaluating it on the ten
loops:

| max_sensors | On equal-prob | On the loops | Worst loop |
|---:|---:|---:|---:|
| 2 | 37.07% | 29.80% | 0.80% |
| 3 | 62.50% | 53.24% | 0.90% |
| 4 | 84.37% | 77.23% | 56.70% |
| 5 | 94.20% | **91.69%** | 56.80% |
| 6 | 96.97% | **93.77%** | 84.85% |

The jump from 53% to 77% at k=4 is the loop-family ambiguity threshold, exactly
as predicted. A genuinely strong universal base model needs k >= 5.

## Warm restart retried

| k | Base model | No detector | Warm | Gain |
|---:|---:|---:|---:|---:|
| 2 (old pooled prior, 70.35%) | — | 71.62% | 68.96% | **-2.66pp** |
| 5 | 91.69% | 86.74% | 87.65% | **+0.91pp** |
| 6 | 93.77% | 90.52% | 90.62% | **+0.10pp** |

**The sign flips.** With a base model that is actually better than what the
learner reaches alone, warm restart stops hurting.

## The law that covers every result so far

Warm-restart gain tracks **headroom** = base model minus learner-alone:

| Setting | Base from | Base | Alone | Headroom | Warm gain |
|---|---|---:|---:|---:|---:|
| grid PPO k=6 | matrix C | 99.00% | 71.03% | +27.97pp | +24.48pp |
| grid tabular k=6 | matrix C | 99.00% | 66.18% | +32.82pp | +15.95pp |
| IoBT tabular k=5 | equal-prob | 91.69% | 86.74% | +4.95pp | +0.91pp |
| IoBT tabular k=6 | equal-prob | 93.77% | 90.52% | +3.25pp | +0.10pp |
| IoBT tabular k=2 | pooled loops | 70.35% | 71.62% | -1.27pp | -2.66pp |
| grid tabular k=2 | matrix C @ k=2 | 31.09% | 45.55% | -14.46pp | -2.20pp |

correlation(headroom, gain) = **0.901**, and the sign of the headroom predicts
the sign of the gain in all six rows.

## Why IoBT's gain is small even when positive

Headroom is only 3-5pp there, because at k>=5 the learner already reaches
86-90% on the loops by itself. On the 5x5 grid the learner alone manages just
66-71% against a 99% base model, leaving 28-33pp of headroom — and that is where
the large wins come from.

So change detection with warm restart is worth exactly as much as the base model
is better than self-taught performance. It is not a property of the detector.
