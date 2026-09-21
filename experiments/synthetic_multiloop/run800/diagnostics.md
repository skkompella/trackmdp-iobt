# Run 800 diagnostics — why the per-loop spread is so wide

Best checkpoint: iteration 20 of 170. Mean 63.49% @ 5.51 sensors/step,
but the spread across loops is 28.30% to 89.00% (sd 18.62pp).

## Successor ambiguity

At each node, how many different nodes can come next across the 10-loop family.
This is what the agent cannot disambiguate: it never observes which loop is active.

| node | distinct successors | which |
|------|--------------------|-------|
| 0 | 3 | 1, 4, 7 |
| 1 | 4 | 0, 2, 6, 7 |
| 2 | 3 | 1, 3, 6 |
| 3 | 3 | 2, 5, 6 |
| 4 | 4 | 0, 2, 3, 5 |
| 5 | 2 | 3, 4 |
| 6 | 3 | 1, 2, 8 |
| 7 | 3 | 0, 1, 9 |
| 8 | 2 | 6, 9 |
| 9 | 2 | 7, 8 |

Per loop, ranked by accuracy:

| loop | len | mean ambiguity | accuracy | sensors |
|------|-----|----------------|----------|---------|
| 3 | 10 | 2.90 | 89.00% | 5.40 |
| 5 | 7 | 3.14 | 84.90% | 5.58 |
| 1 | 6 | 2.83 | 82.35% | 5.17 |
| 9 | 7 | 3.29 | 70.60% | 5.43 |
| 8 | 3 | 3.00 | 66.00% | 5.34 |
| 7 | 9 | 3.00 | 65.80% | 5.34 |
| 4 | 7 | 2.86 | 56.50% | 5.71 |
| 6 | 6 | 3.33 | 49.15% | 5.67 |
| 2 | 7 | 3.14 | 42.30% | 5.71 |
| 0 | 7 | 3.14 | 28.30% | 5.71 |

- corr(mean ambiguity, accuracy) = **-0.405**
- corr(loop length, accuracy)    = **+0.203**

Ambiguity explains some of the spread and in the expected direction, but only
some: loops 0, 2 and 5 share an identical mean ambiguity of 3.14 and score
28.30%, 42.30% and 84.90%. So node-level ambiguity is not the whole story --
the ordering of ambiguous nodes within a loop matters too, and that is not
captured here. Loop length is NOT the driver: the longest loop (3, a
Hamiltonian cycle over all 10 nodes) is the easiest, and the shortest (8, len 3)
sits mid-table.

Loops 0 and 2 are nearly the same cycle -- [0,4,5,3,6,2,1] vs [0,4,5,3,2,6,1],
differing only in whether 6 or 2 comes first after 3 -- yet they differ by 14pp.
That pair is the cleanest evidence that the policy resolves the family
inconsistently rather than hedging uniformly.

## Why training decayed after iteration 20

Reward arithmetic at sensor_rew = -0.25, max_sensors = 6:

| action | net reward |
|--------|-----------|
| detect with all 6 sensors on | **0.00** |
| detect with 1 sensor on      | +1.25 |
| miss with 0 sensors on       | -0.50 |

Full activation is worth exactly nothing -- the 6-sensor cost (1.5) precisely
cancels the tracking reward (1.5). Only *precise* activation pays. PPO therefore
pushes toward narrower activation, but successor ambiguity means it cannot be
precise, so it activates less and misses more: from iteration 20 to 170,
sensors/step fell 5.51 -> 3.22 while mean accuracy fell 63.5% -> 35.2%. Both
axes got worse together, which is the signature of a penalty that is too strong
rather than a genuine accuracy/energy trade.

Next step: sweep --sensor-rew (0.0, -0.05, -0.10) from scratch. At 0.0 the
ceiling for this setup should be visible; the frontier between there and -0.25
is where a useful operating point lives.
