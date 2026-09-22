# Online learning on a sequential loop schedule (runs 810-814)

Tabular SARSA(lambda), 10 loops x 50 episodes x 5 passes, at the regime the
lever experiments selected: `max_sensors=2`, tl=1, `grid` preset. Every arm is
identical except for what happens when the trajectory switches.

| Run | Arm | Kept across a switch | Mean acc | Sensors | Latency | Restarts | Caught | False |
|-----|-----|----------------------|---------:|--------:|--------:|---------:|-------:|------:|
| 811 | **no detector** | everything | **71.62%** | 1.88 | 7.5 | 0 | — | — |
| 812 | warm restart | pooled prior | 67.94% | 1.93 | 5.1 | 40 | 22/49 | 18 |
| 813 | library restart | best archive | 62.86% | 1.88 | 8.3 | 73 | 46/49 | 27 |
| 814 | **cold, ORACLE** | nothing | 59.60% | 1.84 | 9.2 | 49 | 49/49 | **0** |
| 810 | cold, GLR | nothing | 47.51% | 1.87 | 10.7 | 107 | 47/49 | 60 |

References in this regime: PPO (run 803) **31.53%**; pooled stationary tabular
**70.35%**; single-loop tabular lower bound **80.23%**.

## Headline: change detection with restart HURTS here

Accuracy is monotonically ordered by how much knowledge survives a restart —
keep everything 71.62%, pooled prior 67.94%, best archive 62.86%, wipe 47.51%.

## The oracle control decomposes the deficit almost exactly in half

Run 814 restarts at the 49 KNOWN switch points, so detector error is removed:

```
  no detector                     71.62%
  cold, PERFECT detection (814)   59.60%   cost of restarting at all      -12.02pp
  cold, GLR detection     (810)   47.51%   additional cost of false alarms -12.09pp
```

Half the cold arm's shortfall is the detector over-firing, which tuning could
recover. **The other half is the cost of restarting at all, and no detector can
recover it.** Restart-on-change is the wrong architecture for this problem.

## Why

The 10 loops share a transition structure — every loop is a path through
`IOBT_MAP` — which is exactly why run 241 covers all of them zero-shot after
training only on the topology's random walk. Knowledge of one loop therefore
transfers to the others, and a restart discards precisely that. The premise
behind restart-on-change is that segments are unrelated; here they are not.

Per-pass accuracy confirms it. Only the no-detector arm accumulates:

| Arm | pass 0 | 1 | 2 | 3 | 4 |
|-----|-------:|--:|--:|--:|--:|
| no detector | 67.0% | 68.9% | 70.3% | 76.6% | 75.3% |
| warm | 67.5% | 66.1% | 67.6% | 68.2% | 70.2% |
| library | 65.0% | 61.7% | 62.2% | 60.3% | 65.1% |
| cold (oracle) | 57.5% | 60.0% | 61.5% | 60.2% | 58.7% |
| cold (GLR) | 50.8% | 47.2% | 47.0% | 44.0% | 48.5% |

Cold restart with the GLR detector is flat-to-declining: 107 restarts across
2,500 episodes is one every ~23 episodes, so it never accumulates anything.

## Tabular TD vs PPO

71.62% against PPO's 31.53% in the identical regime — **2.3x**. With 21 states
and 55 actions there is nothing for function approximation to buy, and PPO's
clipped objective is pure drag. This is the clearest evidence in the project
that the Track-MDP state space is small enough to solve exactly.

## Caveats, stated plainly

- **Latency favours the wrong arms.** Warm shows the best latency (5.1) and cold
  the worst (10.7), but latency is measured against each segment's own plateau,
  so a learner that settles quickly into a worse plateau scores well. Read
  accuracy. The limitation is documented in `adaptation_latency`.
- **One detector tuning was tested** (delta 0.01, min_samples 30, warmup and
  cooldown 5 episodes). A sweep would move run 810 toward run 814's 59.60%, but
  814 bounds what any tuning can achieve, and it still trails the baseline.
- **Restart may yet win where segments do NOT share structure.** Nothing here
  argues against change detection in general — only against it on a loop family
  drawn from one graph. A schedule over genuinely unrelated topologies is the
  test that would show the opposite, and it is untried.

## What to do instead

Continual learning with no restart is the recommendation for this problem. If
adaptation speed to a switch matters more than steady-state accuracy, warm
restart is the compromise: it gave the best latency (5.1 episodes) at a 3.7pp
accuracy cost. Cold restart has nothing to recommend it here.
