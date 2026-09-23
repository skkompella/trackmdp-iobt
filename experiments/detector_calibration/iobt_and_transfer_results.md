# Detector calibration — IoBT, and transfer from the 5x5 grid

IoBT streams: 250,000 environment steps, **49 switches every 5,000 steps**
(10 loops x 5 passes), max_sensors 2, against a matched stationary reference.

## IoBT operating point (ARL0 >= 5,000)

| Signal | ARL0 | Mean delay | Caught | delta |
|---|---:|---:|---:|---:|
| **hit** | 100,000 | **27** | **43/49** | 1.0e-04 |
| miss_run | 5,882 | 1,238 | 35/49 | 1.0e-22 |
| sensors | 12,500 | 1,690 | 31/49 | 6.3e-12 |
| reward | 18,182 | 1,292 | 24/49 | 6.3e-12 |
| staleness | 25,000 | 1,169 | 22/49 | 2.5e-08 |
| transition_surprise | 2,041 | 4,400 | 1/49 | (below target) |

**On IoBT the current signal is the best by a wide margin** — 43 of 49 switches
at a 27-step delay. Detection is not the bottleneck here at all.

The original IoBT runs (810-814) used delta=0.01, which measured out at
ARL0 4,166 — roughly **24x too trigger-happy**. Calibrated to delta=1e-4 the
same signal gives ARL0 100,000 and catches 43/49. That is the fix for IoBT.

## The ranking INVERTS between environments

| Signal | 5x5 grid | IoBT |
|---|---|---|
| `hit` | **worst** (2/5) | **best** (43/49, 27-step delay) |
| `transition_surprise` | **best** (5/5) | **worst** (1/49) |

### Why: it is about hedging, not about the detector

On the 5x5 grid `max_sensors=6` covers all <=5 possible destinations, and every
matrix's support is a subset of {stay} u neighbours — so a hedging policy exists
and a switch often does not change tracking success at all. Three of the five
switches moved the hit rate by <=5.4pp, below the 5-10pp detection floor. The
change is real but invisible in that stream.

On IoBT `max_sensors=2` against loops with up to 4 possible successors makes
hedging impossible. A switch immediately breaks a specialised policy, so the
hit rate drops hard and fast — hence the 27-step delay.

**Detectability in the success stream is a property of whether the policy can
hedge, not of the detector.** That is the generalisable finding, and it predicts
which signal to use before running anything: if the sensor budget covers the
successor ambiguity, `hit` will be blind and an environment-side signal is
needed; if it does not, `hit` is the sharpest signal available.

## Transfer: nothing transfers

| Config | Signal | delta | On 5x5 | On IoBT |
|---|---|---:|---|---|
| 5x5-tuned | transition_surprise | 2.3e-14 | 5/5, ARL0 25,000 | 46/49, **ARL0 266** |
| IoBT-tuned | hit | 1.0e-04 | 5/5, **ARL0 2,532** | 43/49, ARL0 100,000 |

Each cross-applied configuration "catches" switches in the other environment
only by firing 10-400x too often. Nor does the normalised form transfer: the
tuned ARL0 is 0.24 segment-lengths on the grid and 20 segment-lengths on IoBT,
almost two orders of magnitude apart.

What transfers is the **procedure**, not any constant:

1. record a stationary reference stream — without it no false-alarm rate exists;
2. sweep delta and compare candidates at matched ARL0, never at matched delta;
3. choose the operating point from the downstream cost asymmetry, which here
   favours recall because warm restart makes false alarms nearly free;
4. pick the signal with the largest at-switch effect, which depends on whether
   the policy can hedge.

## Recommendations

| Environment | Signal | delta | Expect |
|---|---|---:|---|
| IoBT (10-node loops, max_sensors 2) | `hit` | 1e-4 | 43/49, 27-step delay |
| 5x5 grid (max_sensors 6) | `transition_surprise` | 2.3e-14 | 5/5, ~17.8k-step delay |
| 5x5 grid, speed priority | `reward` | see curve | 4/5, ~10.9k-step delay |

## Caveat

At delta=1e-4 the `hit` signal also reaches 5/5 on the grid, at ARL0 2,532 —
about 40 false alarms per segment. Given that warm restart is nearly indifferent
to false alarms (run 822: 51 alarms, 82.13%; run 824: zero, 77.04%) that may
even be acceptable downstream, but it is no longer detection in any meaningful
sense; it is periodic reset with extra steps. The honest reading is that a
detector operating there is not earning its keep, which is the same conclusion
run 824 reached by a different route.
