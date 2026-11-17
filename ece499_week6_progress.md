# ECE 499 Senior Thesis — Week 6 Progress Report

**Srihari Kompella** | skk10@illinois.edu | Fall 2026
**Project:** Real-time reinforcement learning for multi-sensor target tracking (Track-MDP), under the IoBT CRA
**Advisor:** Prof. Venugopal V. Veeravalli | **Course director:** Prof. Erhan Kudeki

The week 3 report ended on an open problem: the tracker reached 98.35% on its training
session but only 87.35% on an unseen one, because the policy had learned that session's
particular movement pattern. Weeks 4–6 therefore focused on making the tracker
**movement-agnostic and adaptive**. One agent should track targets whose motion it has
never seen, notice when the motion pattern changes, and recover quickly, while staying
within a fixed sensor budget. All of this work was done in simulation, on the Camp Buckner
10-node graph and on 5×5 and 10×10 sensor grids, so that ground truth and the motion
model are fully controlled.

---

## 1. Research completed since week 3

- **Multi-pattern training environments.** Built simulated environments in which the
  target follows a different motion pattern each episode or segment:
  - 81 deterministic loops on the IoBT graph;
  - an equal-probability random walk;
  - randomly generated transition matrices (each move drawn at random over "stay or step
    to a neighbour") on the IoBT graph and on 5×5 and 10×10 grids.

  Patterns can be switched mid-run, which is how a change in the target's behaviour is
  simulated.
- **A truly online learner.** PPO updates once per 512-step batch, so it adapts slowly by
  design. I implemented a tabular SARSA(λ) learner that updates after every step. Its
  first version (one weight per sensor) shared credit across sensors and plateaued around
  72–74% on the grid. Restricting the action table to the cells the target can actually
  have reached fixed this, raising accuracy to about 95% at a 4–5 sensor budget with no
  model of the motion.
- **Change detection with restart (DAL-style).** Added a GLR change detector that watches
  a per-step signal. When it fires, the learner restarts, either *cold* (wiped), *warm*
  (loaded from a base model trained on the equal-probability walk), or *library* (best
  earlier table). This was evaluated with both the online learner and PPO.
- **Detector calibration.** Built a harness that records detector-signal streams once and
  replays them offline through thousands of detector settings. Signals are compared at a
  matched false-alarm rate (ARL0, the mean steps between false alarms) rather than at a
  matched threshold. Three candidate signals were compared: tracking hit, reward, and
  transition surprise.
- **Large-scale experiment campaign.** Ran more than 100 runs on the three environments.
  They cover sensor budgets k = 3–6, a sensor-cost sweep (−0.01 to −0.16), and four arms
  (online, online + DAL, PPO, PPO + DAL). Each setting is compared against an oracle
  tracker that knows the true transition matrix.
- **Engineering and measurement fixes.**
  - **Greedy evaluation.** Online accuracy had been scored with exploration on, which
    understated it by 6–10 points. It is now scored with exploration off, as PPO is.
  - **PPO warm-restart crash.** Loading a checkpoint mid-run crashed rollout workers and
    silently dropped samples. Restarts now set the base model's weights in place.
  - **Checkpoint overwrite bug.** The final checkpoint was overwriting the best one.
  - **Memory-aware job scheduler.** It runs up to six Ray/RLlib jobs concurrently
    without exhausting machine memory.
  - **Tests.** The test suite grew to 247 tests.

## 2. Results and their significance

**Main result: random motion, k = 5 sensors.** Five random transition matrices per
environment, visited in sequence three times (14 pattern switches). Values are mean greedy
accuracy over the whole run.

| Environment | Oracle (knows matrix) | Online + DAL (cost −0.04) | PPO + DAL (best cost) |
|---|---:|---:|---:|
| IoBT graph (10 nodes) | 97.5% | **97.6%** | 96.6% |
| 5×5 grid (25 cells) | 96.9% | **95.6%** | **97.8%** |
| 10×10 grid (100 cells) | 99.0% | **97.2%** | **98.8%** |

- **Near-oracle tracking of unseen, changing motion with no motion model.** Both learners
  come within 0–2 points of a tracker that is told the true transition matrix, while
  switching on only about 4–5 of 10–100 sensors per step. This is the movement-agnostic
  behaviour the week 3 plan called for.
- **What makes adaptation work: the "headroom law".** Across every experiment, the gain
  from DAL equals the base model's accuracy minus the learner's accuracy on its own.
  The correlation is 0.90–0.997 across four experiment families, and the sign of the gain
  was predicted correctly in 28 of 30 settings. The detector's role is only to trigger a
  reload. The benefit comes from having a good general-purpose base model to reload. One
  control showed this directly: restarting at the exact true switch times (77.0%) scored
  *below* restarting on noisy detections (82.1%).
- **Sensor cost: the two learners behave very differently.**

| Cost per sensor | −0.02 | −0.04 | −0.06 | −0.08 | −0.12 | −0.16 |
|---|---:|---:|---:|---:|---:|---:|
| Online + DAL, IoBT | 93.8 | **97.6** | 97.1 | 89.1 | 77.6 | 68.3 |
| Online + DAL, 10×10 | 95.0 | **97.2** | 95.6 | 82.7 | 55.7 | 45.1 |
| PPO + DAL, IoBT | 90.9 | 95.5 | — | **96.6** | 94.1 | 95.8 |
| PPO + DAL, 10×10 | 98.7 | 97.1 | — | **98.8** | 98.5 | 96.7 |

  The online learner has a clear optimum at cost −0.04 to −0.06 and collapses at higher
  cost, because it learns to switch on too few sensors. PPO + DAL is largely insensitive
  to cost. This gives a practical rule: the online learner suits deployments where
  accuracy matters more than energy, and PPO + DAL suits energy-constrained ones.
- **Sensor budget.** At k = 6 on the 10×10 grid, PPO + DAL reaches 98.5–98.8% with only
  1–3 restarts. At k = 4, no method exceeds 86% on the 10×10 grid. The budget, not the
  learner, is the limit there: the oracle itself caps at 93.7%.

## 3. Challenges and future work

- **The change detector is still poorly calibrated in the loop.** Calibrating offline to
  one false alarm per 150,000–500,000 steps picked the transition-surprise signal on all
  three environments. It caught 10–14 of 14 switches on recorded streams. In the live
  loop, however, the same detector fired far more often: 79 to 976 restarts, against 14
  real switches. Accuracy moved by only −1.9 to +0.6 points. My explanation is that each
  restart changes the policy, and with it the very stream the detector was calibrated on.
  Calibration on recorded data therefore does not carry over to the closed loop. Because
  a warm restart is cheap (the headroom law above), this costs little accuracy today, but
  it wastes energy and must be fixed before field use.
- **PPO base-model quality is erratic.** The same training recipe produced base models
  scoring between 60% and 99%, and this decides PPO + DAL performance more than the
  detector does. The planned fix is to train several seeds and keep the best.
- **Single-seed results.** Most headline numbers are from one seed. Multiple seeds are
  needed before these go in the thesis.
- **Back to real data.** These results are simulated. The next step is to apply the
  adaptive learner with a warm-restart base model to the recorded Camp Buckner sessions,
  to test whether it closes the 98.35% vs 87.35% in-domain/unseen gap from week 3.
- **Planned next steps:**
  1. multi-seed PPO base models and headline runs;
  2. a detector that accounts for its own restarts, for example a cool-down window or
     recalibration after a restart;
  3. cross-session evaluation of the online + DAL tracker on the field recordings;
  4. write the simulation chapter of the thesis draft from `update_oct.md`.

---

**Status vs. course objectives.** The movement-agnostic and adaptive-tracking research
proposed in week 3 has been carried out and documented, with a positive main result:
near-oracle accuracy on unseen random motion at k = 5. The remaining open problems
(detector calibration in the loop, seed variance, transfer back to field data) are
identified and scoped. I remain on track for the thesis draft and the Undergraduate
Research Symposium.
