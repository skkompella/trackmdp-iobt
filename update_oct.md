# Track-MDP IoBT — October Update: online learning, change detection, random environments

_Updated 2026-10-06. Covers the synthetic (simulated) work since update.md: one agent
across many movement patterns, an online (tabular) learner versus PPO, and DAL-style
change detection with restart. Every number is one seed unless stated._

---

## TL;DR — 5 random IoBT and 5 random 5×5 grid environments

**Setup.** Five random transition matrices on the Camp Buckner IoBT graph and five on a
5×5 grid. Each row is a random (Dirichlet(1)) distribution over "stay + each neighbour", so
the object moves one step at a time in an arbitrary drift pattern. One learner sees
R0 → R1 → R2 → R3 → R4, three passes (14 switches). Online learner: 1,000 episodes per
segment; PPO: 200 iterations (~1,024 episodes). Four arms: online, online + DAL, PPO,
PPO + DAL. DAL = GLR change detector on the per-step hit stream + **warm restart** to an
equal-probability base model. Budgets k = 4 and 5, sensor cost −0.16 and −0.04.
"Best possible" = a tracker that knows each matrix and watches its most likely cells.

**Greedy accuracy (mean over the run), cells switched on per tracked step in brackets:**

| Env | k | Cost | Best possible | Online | Online + DAL | PPO | PPO + DAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 4 | −0.16 | 94.6% | **78.0%** (2.8) | 75.8% (3.2) | 50.7% (3.9) | 76.1% (3.9) |
| grid | 4 | −0.04 | 96.0% | **92.2%** (3.5) | **92.3%** (3.8) | 45.5% (4.0) | 56.2% (4.0) |
| grid | 5 | −0.16 | 94.6% | 73.2% (2.7) | 68.3% (3.0) | 57.8% (4.7) | **97.8%** (4.4) |
| grid | 5 | −0.04 | 96.9% | 94.6% (3.7) | **95.6%** (4.0) | 56.0% (4.8) | 90.9% (4.6) |
| IoBT | 4 | −0.16 | 91.3% | 75.4% (3.4) | 73.6% (3.7) | 81.9% (3.6) | **83.5%** (3.8) |
| IoBT | 4 | −0.04 | 96.7% | 92.5% (3.9) | **93.6%** (4.0) | 79.8% (3.8) | 83.5% (3.8) |
| IoBT | 5 | −0.16 | 91.3% | 69.4% (3.8) | 68.3% (3.9) | 90.8% (3.8) | **95.8%** (4.0) |
| IoBT | 5 | −0.04 | 97.5% | 94.6% (4.6) | **97.6%** (4.7) | 88.9% (4.3) | 95.5% (4.0) |

**What it says**

1. **At cost −0.04 the online learner is the best or tied in every setting** (92–98%,
   within 0–4 points of best possible). At k=5 on IoBT, online + DAL (97.6%) matches the
   best possible reward-maximising tracker.
2. **At cost −0.16 PPO + DAL wins 3 of 4 settings** (up to 97.8%). The online learner
   switches on too few cells (2.7–3.8) and stalls at 69–78%. It cannot even learn the
   equal-probability matrix at that cost (its base models score 61–71%).
3. **PPO without DAL is weak on random movement** (45–58% on the grid): training straight
   through the switches never settles.
4. **DAL helps by exactly as much as its base model beats the learner alone**
   ("headroom"). Correlation of headroom vs DAL gain: 0.997 for PPO (8/8 signs agree),
   0.78 online, 0.95 overall. PPO + DAL ends up roughly where its base model is.
5. **PPO base-model quality is erratic** (60%–99% from the same 1,200-iteration recipe),
   and that decides PPO + DAL more than the detector does.
6. **The detector over-fires on random movement** (31–609 restarts for 14 switches).
   Harmless with a good base model, costly (−1 to −5 points) with a weak one.

Full table with restart counts and end-of-segment numbers:
`experiments/random_matrix/results.md`. Runs 1000–1031.

---

## 1. What changed in the codebase

| Piece | File | Purpose |
|---|---|---|
| Synthetic loops | `src/core/iobt_loops.py`, `multiloop_iobt_env.py` | 81 directed cycles on IOBT_MAP; object follows one loop per episode |
| Equal-probability IoBT walk | `src/core/equalprob_iobt_env.py` | Object stays or moves to any neighbour with equal probability (base model) |
| Matrix-driven IoBT env | `src/core/matrix_iobt_env.py` | Object samples a 10×10 transition matrix; switchable; PPO-compatible |
| Random matrices | `src/core/random_transition_matrices.py` | Dirichlet rows over stay + neighbours, grid and IoBT; `R<i>,C` specs |
| Switching grid | `src/core/switching_grid_env.py`, `grid_transition_matrices.py` | 5×5 grid with switchable matrices A (NE drift), B (SW), C (equal) |
| Online learner | `src/core/tabular_td.py` | Tabular SARSA(λ); **subset**, **factored** and **local** action modes |
| Change detection | `src/core/change_detection.py` | GLR detector (Bernoulli/Gaussian), ARL0 calibration, cold/warm/library restart |
| Detector calibration | `detection_signals.py`, `detector_eval.py`, `examples/calibrate_detector.py` | Offline replay of recorded streams at matched false-alarm rate |
| Shared rollout | `src/core/online_schedule.py` | Episode loop, **greedy evaluation**, cells-per-step histogram |
| Scripts | `examples/train_multiloop_online.py`, `train_switching_grid_online.py`, `train_switching_grid_ppo.py`, `random_matrix_oracle.py` | Online (IoBT / grid), PPO (grid / IoBT), best-possible reference |

Measurement fixes made along the way:

- **Greedy evaluation.** Online accuracy used to be scored while exploring (ε = 0.1–0.15),
  which understated the learner by 6–10 points. All runs from 854 onward also report
  greedy accuracy (exploration off), the same way PPO and the base models are scored.
- **Cells per tracked step.** Recorded separately from the energy-billed "sensors/step"
  (which charges a full rescan after losing the object at k sensors).
- **PPO warm restart.** `algo.restore()` crashed rollout workers mid-episode; RLlib
  restarted them, silently losing samples. Now the base model's weights are set in place.
  The older grid PPO warm runs (832, 835) used the old path.
- **Index-order clipping bias** in `evaluate_policy` (favoured low-index cells) — fixed
  with `clip_mode="random"`.

---

## 2. Two learners, and what DAL is

- **PPO** (RLlib 2.54): the existing Track-MDP agent. Updates once per 512-step batch with
  a clipped objective, so it adapts slowly by design.
- **Online learner**: tabular SARSA updated every step. Three ways to represent actions:
  - *subset*: one value per sensor set (IoBT, 10 nodes);
  - *factored*: one weight per cell (5×5 grid; 245k sets otherwise). **Shares credit
    across all active cells**, which caps it (see §5);
  - *local*: one value per subset of the cells the object can have reached
    (5 cells at delay 0, 13 at delay 1) — exact and small. Replaced factored on the grid.
- **DAL** (our implementation of the DAL-style detect-and-reset idea): a GLR change
  detector watches a per-step signal; when it fires, the learner resets — *cold* (wipe),
  *warm* (load a base model), or *library* (best earlier table).

---

## 3. IoBT synthetic loops (deterministic movement)

### 3a. PPO, random loop each episode (runs 800–807)

| Run | k | Sensor cost | Accuracy | Sensors/step | Worst loop |
|---|---:|---:|---:|---:|---:|
| 801 | 6 | −0.16 | 57.02% | 5.69 | 29.00% |
| 802 | 3 | −0.16 | 47.47% | 3.00 | 1.40% |
| 803 | 2 | −0.16 | 31.53% | 1.75 | 0.50% |
| 804 | 1 | −0.16 | 24.11% | 1.00 | 0.25% |
| 805–807 | 6 | −0.5 / −1 / −2 | 58.21 / 60.46 / 53.58% | 4.98 / 5.39 / 5.44 | — |
| **241** | 6 | −0.25 | **99.00% on every loop** (trained on the equal-probability walk) | 4.26 | 99.00% |

`max_sensors` controls activation exactly; scaling the sensor cost barely moves it
(the cost also appears in the missing-state penalty, so it cancels).

### 3b. Online learner, loops in sequence (10 loops × 5 passes)

| Runs | k | Segment | Arm | Accuracy | Notes |
|---|---:|---:|---|---:|---|
| 811 | 2 | 50 ep | none | 71.62% (exploring) | PPO in the same regime: 31.53% |
| 810 / 815 | 2 | 50 | DAL cold | 47.51% / 56.54% | 107 / 58 restarts |
| 812 / 816 | 2 | 50 | DAL warm (pooled-loop base) | 67.94% / 68.96% | base was worse than the learner (70.35%) |
| 814 | 2 | 50 | cold, at true switch times | 59.60% | restarting at all costs 12 points |
| 852 / 853 | 5 | 50 | none / DAL warm (equal-prob base) | 86.74% / 87.65% | |
| 850 / 851 | 6 | 50 | none / DAL warm | 90.52% / 90.62% | |
| 854 / 855 | 6 | 200 | none / DAL warm | **98.72%** / 98.59% greedy | 4.17 / 4.90 sensors |
| 856 / 857 | 5 | 200 | none / DAL warm | 97.76% / 96.88% greedy | |
| 858 / 859 | 6 | 1000 | none / DAL warm | 98.47% / 98.97% greedy | 3.33 / 4.90 sensors |
| 860 / 861 | 5 | 1000 | none / DAL warm | 98.24% / 98.42% greedy | 3.01 / 4.53 sensors |
| 862 / 863 | 3 / 4 | 1000 | none | 97.91% / 97.81% greedy | 2.22 / 2.70 sensors |

On deterministic loops the online learner reaches the 99% ceiling on its own once
segments are 200+ episodes. DAL warm removes the dip right after a switch (first check
after a switch: 95% → 99%) but keeps the agent on the base model's "watch every
neighbour" hedge, costing 0.7–1.5 extra sensors per step.

---

## 4. 5×5 grid, matrices A and B alternating (k=6)

| | No detector | DAL cold | DAL warm (matrix-C base) |
|---|---:|---:|---:|
| Online (factored) | 66.18% | 18.84% | 82.13% |
| PPO | 71.03% | 30.08% | **95.51%** (98.67% end, 3.83 sensors) |

The oracle-timing control (warm reset exactly at switches, run 824: 77.04%) scored *below*
the noisy detector (822: 82.13%): the gain is returning to a good model, not timing.

---

## 5. Why the grid learner was stuck near 90%, and the fix

- A matrix-aware tracker under matrix A at cost −0.16 maximises reward by watching only
  3 cells → **92.5% is the reward-optimal ceiling**; at −0.04 it rises to 98.3%.
- But lowering the cost did not help the factored learner (24-run sweep, runs 880–903:
  best 72–74%). Diagnosis: at cell 12 it valued unreachable cells as highly as the 40%
  destinations; only 28% of watched cells were correct. Shared per-cell credit.
- **Local mode** (exact table over reachable subsets), runs 910–917:

| Run | k | Cost | Greedy mean / end | Cells per tracked step |
|---|---:|---:|---|---:|
| 912 | 4 | −0.04 | **95.22% / 95.95%** | 3.42 |
| 916 | 5 | −0.04 | **94.74% / 95.42%** | 3.35 |
| 910 / 914 | 4 / 5 | −0.16 | 81–82% | 2.7–2.8 |

Online learning curves (no detector): IoBT saturates within 200–400 episodes at every k;
the grid factored learner was still rising at 1,000 episodes (runs 862–873).

---

## 6. Change-detector calibration

- Signals must be compared at the same false-alarm rate (ARL0), not the same δ.
- The best signal depends on whether the policy can **hedge** (cover every possible next
  cell): on the 5×5 grid at k=6 `hit` is blind (2/5 switches) and `transition_surprise`
  catches 5/5; on IoBT loops at k=2 `hit` catches 43/49 with a 27-step delay.
- No tuned threshold transfers between environments; the procedure does.
- Tuned detection improved every *cold*-restart arm (+6 to +28 points) and left *warm*
  arms flat. Detection quality and tracking accuracy are decoupled whenever the restart
  target is good.

---

## 7. The headroom law

Across every experiment: **DAL warm gain ≈ base-model accuracy − learner-alone accuracy.**

| Experiment | Correlation | Sign agreement |
|---|---:|---:|
| Loops + A/B grid (6 settings) | 0.90 | 6/6 |
| Random matrices, PPO (8) | 0.997 | 8/8 |
| Random matrices, online (8) | 0.78 | 6/8 |

The detector's job is only to trigger the reload. What matters is (a) having a base model
better than what the learner reaches alone, and (b) that base model being reliably
trained.

---

## 8. Open questions / next steps

1. Re-train PPO base models with several seeds and keep the best (finding 5 above).
2. Calibrate the detector threshold on random-matrix streams (it fires 2–40× too often).
3. Fix the online learner at high sensor cost (under-watching at −0.16).
4. Multiple seeds for the headline online results at cost −0.04.
5. Scale-up: 5 random 10×10 grids, with and without DAL (queued).
