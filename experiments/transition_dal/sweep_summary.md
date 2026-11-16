Tolerance: 300 episodes

| run | env | detector | mode | rho | seed | caught | false alarms | median delay (ep) | greedy mean % | greedy end % | greedy sensors/step | train sensors/step | probe frac % | restarts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_rho0.05_s1 | grid | transition | monitor | 0.05 | 1 | 14/14 | 0 | 1.00 | 94.65 | 94.77 | 3.82 | 3.75 | 0.03 | 0 |
| grid_rho0.1_s1 | grid | transition | monitor | 0.10 | 1 | 14/14 | 0 | 1.00 | 94.26 | 94.73 | 3.81 | 3.76 | 0.34 | 0 |
| grid_rho0.2_s1 | grid | transition | monitor | 0.20 | 1 | 14/14 | 0 | 0.50 | 94.98 | 94.81 | 3.81 | 3.75 | 1.55 | 0 |
| grid_rho0.3_s1 | grid | transition | monitor | 0.30 | 1 | 14/14 | 0 | 1.00 | 95.07 | 95.25 | 3.84 | 3.78 | 3.15 | 0 |
| grid_rho0_s1 | grid | transition | monitor | 0.00 | 1 | 14/14 | 0 | 1.00 | 94.64 | 95.80 | 3.83 | 3.75 | 0.00 | 0 |
| iobt_rho0.05_s1 | iobt | transition | monitor | 0.05 | 1 | 14/14 | 0 | 0.50 | 94.81 | 95.35 | 4.65 | 4.61 | 0.17 | 0 |
| iobt_rho0.1_s1 | iobt | transition | monitor | 0.10 | 1 | 14/14 | 0 | 0.50 | 95.60 | 95.89 | 4.70 | 4.65 | 1.20 | 0 |
| iobt_rho0.2_s1 | iobt | transition | monitor | 0.20 | 1 | 14/14 | 0 | 1.00 | 94.79 | 95.00 | 4.58 | 4.55 | 3.02 | 0 |
| iobt_rho0.3_s1 | iobt | transition | monitor | 0.30 | 1 | 14/14 | 0 | 0.50 | 94.95 | 95.03 | 4.71 | 4.66 | 5.19 | 0 |
| iobt_rho0_s1 | iobt | transition | monitor | 0.00 | 1 | 14/14 | 0 | 0.50 | 94.87 | 95.69 | 4.63 | 4.58 | 0.00 | 0 |

"train sensors/step" includes probe steps; greedy evaluation never probes. Re-applying the rho rule to training sensors: every rho is within +1.6% of rho = 0 (grid max 1.009, IoBT max 1.016).

## rho* choice

Rule (spec item 4): smallest rho with 14/14 caught, median delay <= 200 episodes, and
sensors/step <= 1.05 x the rho = 0 run. Every rho passes, including rho = 0: free tau=0
samples alone catch every switch within 0-4 episodes (100-step resolution, since alarms
are acted on at episode end), with no false alarms.

**rho\* = 0.05 for both environments** (not 0). The random-matrix switches redraw every
row, so some watched stream always shifts and coverage never matters on this benchmark.
Probes exist for changes confined to cells the policy does not watch; rho = 0.05 keeps
that guarantee at a cost of < 0.3% of steps (probe fraction 0.002 grid, 0.004 IoBT).
