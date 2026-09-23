# Detector calibration — 5x5 grid

Streams recorded over 614,400 environment steps (6 segments x 1024 episodes),
5 switches, plus a matched **stationary** run with no switches that supplies the
false-alarm reference. All configurations compared at matched ARL0, never at
matched delta.

## Operating characteristic

Each cell: switches caught out of 5 / mean detection delay in environment steps.

| Signal | ARL0>=100,000 | ARL0>=50,000 | ARL0>=25,000 | ARL0>=10,000 |
|---|---|---|---|---|
| **transition_surprise** | — | — | **5/5  17,844** | **5/5  17,844** |
| reward | 4/5  30,453 | 4/5  30,453 | 4/5  **10,880** | 4/5  10,880 |
| miss_run | — | 4/5  35,389 | 4/5  35,389 | 4/5  35,389 |
| **hit (current)** | 2/5  32,471 | 2/5  32,471 | **2/5  32,471** | 4/5  42,000 |
| staleness | 2/5  62,600 | 2/5  62,600 | 4/5  13,630 | 4/5  13,630 |
| sensors | — | — | 2/5  48,890 | 2/5  41,854 |

## Findings

### 1. The signal currently in use is the worst of the six

`hit` catches **2 of 5** switches at every false-alarm rate down to
ARL0 25,000, and only reaches 4/5 by dropping to ARL0 10,000 — at which point
its delay is the worst in the table (42,000 steps). Every other candidate
dominates it somewhere.

This matches the synthetic gate exactly. At ARL0 20,000 the detection floor is
between 5 and 10pp, and the measured at-switch effects on this stream were
+39.1, -1.0, +9.8, -0.4 and +5.4pp: one comfortable, one marginal, three below
the floor. The two invisible switches are invisible because the tracker's
success rate genuinely does not change across them.

### 2. transition_surprise is the only signal that catches all five

5/5 at ARL0 25,000 with a delay of 17,844 steps. It cannot be pushed above
ARL0 25,000 — its own estimator is non-stationary, since surprise falls as the
transition model is learned — but it does not need to be. See below.

### 3. The right operating point is NOT a high ARL0

A false alarm is nearly free in this system, because the restart target is what
matters, not the timing: run 822 fired **51 false alarms** and scored 82.13%,
while run 824 fired **zero** and scored 77.04%. Under warm restart a spurious
restart just reinstates a strong generalist.

So the cost asymmetry runs the other way from the textbook case: missed switches
are expensive, false alarms are not. The operating point should favour recall,
which puts ARL0 ~25,000 and `transition_surprise` at the top, with `reward` the
choice if delay matters more than the fifth switch.

### 4. `reward` is a free upgrade over `hit`

4/5 at 10,880 steps, needs no new machinery, and is already computed every step.
It carries strictly more information than the binary hit: how many sensors were
spent as well as whether the object was found.

## Recommendation for the 5x5 grid

| Priority | Signal | delta | ARL0 | Expect |
|---|---|---|---|---|
| Recall | `transition_surprise` | 2.28e-14 | 25,000 | 5/5, ~17.8k-step delay |
| Speed | `reward` | (see curve) | 25,000 | 4/5, ~10.9k-step delay |

Both beat the current `hit` at 2/5.

## Method notes and corrections

- `max_window` was held at 200 rather than swept, to keep a full sweep to
  minutes instead of an hour. It is a parameter that could still matter.
- Two reporting errors were made and fixed while producing this table. The
  first ranked signals whose configurations sat at wildly different
  false-alarm rates, because the code silently substituted a fallback row when
  a signal missed the target — the exact matched-delta mistake this sweep
  exists to prevent. The second concluded that `transition_surprise` could not
  be stabilised; in fact it needed a delta near 1e-14, far below the original
  grid's floor of 1e-12, and it works fine at ARL0 25,000.
