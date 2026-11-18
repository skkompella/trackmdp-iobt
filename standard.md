# Experiment standard

Parameters every Track-MDP experiment in this repo keeps fixed, so results stay
comparable between runs and with the original Track-MDP
([Grimoid/Track-MDP](https://github.com/Grimoid/Track-MDP), whose
`src/core/environment.py` is identical to ours).

**Rule for agents:** use these settings unless the user explicitly asks for a
deviation. When a run deviates, say so in its notes/README and in the report to
the user. Do not silently change any of them.

---

## 1. Infinite horizon (episodes end only when the target exits)

As in the original Track-MDP, an episode has **no step limit**. It ends only when
the target leaves the grid, which happens with probability **0.005 at every step**
(the "terminal" transition). Episode length is therefore geometric, about
**200 steps on average**, and varies widely from episode to episode.

- Training: no truncation. `--max-ep-steps 0` in both
  `examples/training_grid_rect.py` and `examples/train_switching_grid_ppo.py`
  (the latter also needs `--exit-prob 0.005`).
- Training length: `--iterations N` is a **minimum**. With `--finish-episodes`
  (`training_grid_rect.py`), training continues after iteration N until every
  episode that was in progress at that point has ended (checked per rollout
  worker; usually 1–2 extra iterations, capped at 50), so the run never cuts an
  episode off. Use it for all runs under this standard.
- Evaluation: also runs until the target exits. The 1,000-step cap in the
  evaluators is only a safety net and is almost never reached.
- Do **not** use the old 100-step cap (`max_ep_steps=100`, the RLlib-port
  default) or the `exit_prob=0` "target circulates forever" setting of the
  earlier switching experiments (runs 8xx–14xx). Those runs are not comparable
  with runs under this standard.
- "Infinite horizon" refers to episode termination. PPO still discounts with
  γ = 0.99 (RLlib default).

## 2. Target movement: the 9-candidate / 4-slot scheme

This is the original Track-MDP movement model
(`grid_env.object_transition_matrix_random_3x3`, `src/core/environment.py:128`).
It is easy to get wrong, so in detail:

**Candidates (up to 9).** For each cell, the candidate destinations are its
3×3 neighbourhood restricted to the grid: the cell itself (stay) plus its up to
8 neighbours, **diagonals included**.

| Cell position | Candidates |
|---|---:|
| Interior | 9 |
| Edge (not corner) | 6 |
| Corner | 4 |

**Slots (exactly 4 per cell).** Each cell gets `num_trans = 4` slots. Each slot
is filled by drawing one candidate **uniformly at random, with replacement**.
The draw happens **once**, when the environment is built; after that the table
is fixed. One such table is one "movement model".

**Slot probabilities.** Every step, the target picks a slot using the cumulative
probabilities `[0.2817, 0.5633, 0.845, 0.995]`:

| Slot | Probability |
|---|---:|
| 1 | 0.2817 |
| 2 | 0.2817 |
| 3 | 0.2817 |
| 4 | 0.1500 |
| exit (terminal) | 0.0050 |

These come from `terminal_st_prob = 0.005`, `state_prob_run = 0.15`,
`num_trans = 4`: slots 1–3 share `1 − 0.005 − 0.15` equally and slot 4 gets the
extra 0.15.

**Consequences an agent must understand:**

- From any cell the target can reach **at most 4 distinct cells**, not 9.
  Because draws are with replacement, slots often repeat, and repeated slots
  merge their probability (e.g. two slots on the same cell give it 0.5633).
  Measured: 2–4 distinct next cells per cell, **about 3.3 on average** (run 198's
  saved table: 3.25).
- Which cells those are differs from cell to cell, and can include diagonal
  moves and staying put (stay is a possible move in roughly 40% of cells).
- Across the whole grid all 9 offsets appear, but each cell uses only its own 4
  slots. A tracker that knows the table therefore needs only about 3–4 sensors
  per step to reach the 99% ceiling; one that does not must learn the table.
- Example (run 198, cell 55 on 10×10): slots {65, 44, 64, 46} = down, up-left,
  down-left, up-right, each about 0.28 / 0.28 / 0.28 / 0.15. No stay.

**How to get it in this repo:**

| Script | Flag |
|---|---|
| `examples/training_grid_rect.py` | `--movement original` (`grid_env_rect._draw_original_dests`) |
| `examples/train_switching_grid_ppo.py` | `--matrices O0,O1,... --moore-moves --matrix-seed <seed>` (`random_transition_matrices.original_matrix`; `O<i>` uses seed `matrix_seed + i`) |
| `src/core/environment.py` | the original class; draws with the global numpy RNG |

**Not this scheme** (do not substitute): `--movement 4conn` (stay + up/down/left/
right, equal probability), `--movement moore` (all 9 candidates equally likely),
the Dirichlet `R<i>` matrices, and the unused 4-direction
`object_transition_matrix()` in `environment.py`. The equal-probability Moore
walk (`--movement moore`) is used deliberately for *general base models*
(runs 1510, 1520); that is a stated deviation for that purpose only.

**Reproducibility.** Every movement model must come from a recorded seed (see
§4). Multi-model experiments use seeds 1, 2, 3, 4, … for P0, P1, P2, P3, …
(§7). The original seeds the global RNG with 1, but other code also draws from it
first, so run 198's exact table is not reproducible from the seed alone (it
exists only in `runs/agent_run198_ppo/env_s1.pt`). Bit-exact reproduction of
upstream tables is not required; reproducing *our own* runs is.

## 3. Learning algorithm: PPO

The original Track-MDP trained with **A2C**. A2C is deprecated in Ray RLlib, so
this repo uses **PPO** (Ray RLlib 2.54, old API stack:
`enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False`).

PPO settings (`--ppo-preset port` in `train_switching_grid_ppo.py`; the defaults
of `training_grid_rect.py`):

| Setting | Value |
|---|---|
| Learning rate | 1e-4 (as the original's A2C) |
| Grad clip | 30 (as the original) |
| Rollout workers | 4 (as the original) |
| Train batch | 4,000 steps (RLlib default) |
| Minibatch / epochs | 128 / 30 (RLlib default) |
| Clip / γ / λ / entropy | 0.3 / 0.99 / 1.0 / 0.0 (RLlib default) |
| Network | RLlib default fully connected, 2 × 256 |
| `normalize_actions` | False (required for the MultiDiscrete action) |

The faster `--ppo-preset switching` settings (batch 512, lr 5e-4) belong to the
earlier switching experiments and are a deviation under this standard.

---

## 4. Seeded random number generators

Runs must be reproducible, so **every run is seeded and the seeds are recorded**
in its notes (and in `metrics_history.json`, which stores the run config).

- **Movement table:** fixed by a seed. `training_grid_rect.py --seed S`
  (the code default is unseeded, so always pass it); `train_switching_grid_ppo.py
  --matrix-seed S` (default 20261006; the O0–O3 runs use 20261114).
- **Environment randomness during training** (target start positions, moves,
  exits): seeded per rollout worker as `seed + 1000 * worker_index +
  vector_index` (`src/core/grid_wrapper_rect.py`). This is deliberately *not*
  what upstream does: upstream seeds the global RNG identically in every
  worker, so all four workers generate the same trajectories and PPO sees
  duplicated data. Offsetting by worker index keeps runs reproducible while
  keeping the workers' data independent.
- **Evaluation environment:** its own seed (`--eval-seed`, default 12345, in
  `train_switching_grid_ppo.py`; `--seed` in `training_grid_rect.py`).
- **Not covered:** PPO's network initialisation and action sampling are not
  seeded, so two runs with the same seeds still differ slightly. Compare runs
  by trends, not by exact numbers.

## 5. Accuracy metric (upstream Track-MDP definition)

Reported accuracy uses the **upstream metric**
(`src/evaluation/evaluator.py:evaluate_policy` in the original repo). Precisely:

1. Run E evaluation episodes (100 per evaluation in `training_grid_rect.py`)
   with exploration off (`compute_single_action(..., explore=False)`). Each
   episode starts with the target at a random cell and the tracker in the
   missing state, and runs until the target exits (§1; safety cap 1,000 steps).
2. At every step, count it as **found** if either
   - the tracker is **in the missing state**: all sensors are switched on for
     the rescan, so the step always counts as found; or
   - the tracker is tracking and the target's cell is one of the sensors the
     policy switched on.
3. Episode accuracy = found steps / steps in that episode.
4. **Accuracy = the mean of the per-episode accuracies.** Every episode weighs
   the same, however long it was (this is not total found / total steps).

Things to know when reading it:

- Because every rescan step counts as found, a policy that loses the target
  often still scores well, and the number is higher than "hits while
  tracking". It is the number comparable with the original Track-MDP.
- `training_grid_rect.py` also reports the stricter **tracked accuracy** (hits
  only while tracking, total hits / total steps; the earlier port metric) as a
  secondary column. `train_switching_grid_ppo.py`'s summaries still use the
  tracked metric as `tracking_accuracy`; the upstream one is available from the
  shared evaluator as `upstream_accuracy` (`examples/finetune_deterministic.py`).
  Always say which metric a number is.
- Best-checkpoint selection (§6) uses the upstream metric.

## 6. Checkpoints and metric history

`examples/training_grid_rect.py` does, at every evaluation (every 10 iterations):

- saves the **current** checkpoint to `<save_dir>/current/`;
- copies it to `<save_dir>/best/` when accuracy (§5) beats the best so far;
- rewrites the metric history: `metrics_history.csv` (one row per evaluation:
  iteration, training reward, accuracy, tracked accuracy, evaluation reward,
  episode length, sensors/step) and `metrics_history.json` (same rows, plus the
  run config and the best row).

At the end it reports the **best** accuracy, its iteration, its tracked
accuracy and sensors/step, and the paths of both checkpoints. Report the best
(with its iteration) and the final value to the user.

History is stored as JSON/CSV rather than the upstream `.pt` files: same
content, but readable without torch and without pickle/version issues.
Checkpoints themselves are RLlib checkpoints.

`examples/train_switching_grid_ppo.py` is different: it writes
`online_history.csv` (every iteration) and `online_summary.json` at the end,
and saves a resumable checkpoint every 25 iterations and at each segment end
(`--checkpoint-every`, `--resume`, `--stop-after`).

---

## 7. Thesis experiments

The core question: when the target's movement model changes without warning,
does **warm restart** (change detection + reload of a general base policy)
track better than **no restart** (one PPO policy that keeps learning through the
changes)? Cold restart is excluded: it was consistently the weakest arm in
earlier experiments.

All non-tabular settings of §1–§6 apply (infinite horizon, 9-candidate/4-slot
movement, PPO port settings, seeds recorded, upstream accuracy metric,
best/current checkpoints and metric history).

### Common design (both grids)

- **Movement models:** 4 models P0–P3, each a seeded 9-candidate/4-slot table
  (§2), **reused unchanged by every run** of that grid. **Seeds: P0, P1, P2, P3
  use seeds 1, 2, 3, 4** (continue 5, 6, … if more models are needed). With
  `O<i>` tokens (`O<i>` = seed `matrix_seed + i`) that is
  `--matrices O0,O1,O2,O3 --matrix-seed 1`. Exception: run 1550 (schedule v1)
  was launched before this rule with `--matrix-seed 20261117`.
- **Switch schedule:** `(P0, 0) → (P1, T0) → (P2, T1) → (P3, T2)`: P0 from
  iteration 0, then the target switches to P1 at T0, P2 at T1, P3 at T2 (3
  switches). The times may be arbitrary (segments need not be equal), but they
  are **fixed in advance and identical for every run** of that grid, so the arms
  see exactly the same changes. T is in **PPO iterations** (1 iteration = 4,000
  training steps ≈ 20 episodes at ~200 steps per episode); switches happen
  between iterations. Record the schedule and total length per grid (below).
- **The two runs**, identical except for restarts:

| Run | Policy at start | On a detected change |
|---|---|---|
| 1. No restart | cold (random initial policy) | nothing; PPO keeps training through every switch |
| 2. Warm restart | cold, as run 1 (to confirm) | change detector fires → reload the general base's weights, keep training |

- **General base:** a PPO policy trained on the equal-probability Moore walk
  (`--movement moore`). This is not arbitrary: averaged over the random slot
  draws, the 4-slot scheme gives every in-bounds 3×3 candidate the same
  probability (0.995 / #candidates, plus the 0.005 exit), so the
  equal-probability walk is exactly the *expected* movement model — the best
  thing to assume when the current model is unknown. Same grid and settings as
  the switching runs. Report its accuracy; warm restart can only help when the
  base beats what the learner reaches alone (the "headroom" finding).
- **Change detector (run 2): the transition-stream detector**
  (`src/core/transition_detector.py`; design
  `docs/superpowers/specs/2026-11-15-transition-dal-detector-design.md`; validated
  with the tabular learner in `experiments/transition_dal/results.md`: 0 false
  alarms in 6 null runs of 1.5M steps, 14/14 switches caught).
  - *Samples.* At every step where the tracker has just seen the target at cell i
    (τ = 0, not the missing state), each switched-on cell j of the 3×3 block
    around i gives one sample: did the target move to j (1) or not (0)? The
    sensors are chosen before the target moves, so under a fixed movement model
    each stream is i.i.d. Bernoulli(P(i, j)) whatever the policy does. Steps with
    τ > 0 and rescans give no samples.
  - *Streams.* One Bernoulli GLR per (i, j) with j in the 3×3 block of i (grid
    topology only, never the movement table): 169 streams on 5×5, 784 on 10×10.
  - *Test.* Threshold log(n^1.5 / δ_s) with δ_s = δ_F / (number of streams)
    (union bound), δ_F = 0.01; window 2,000 samples per stream; minimum 30
    samples; no warm-up or cooldown; all streams cleared after an alarm and on a
    restart. No probes (ρ = 0): overriding PPO's action would corrupt its update.
  - *PPO.* Samples come from the rollout workers' training steps (all of them, not
    an evaluation episode), drained after every iteration; alarms are resolved
    once per iteration. Status: plumbing being built
    (`docs/superpowers/plans/2026-11-16-ppo-transition-plumbing.md`).
  - *Report* alarms, switches caught (alarm within tolerance after a switch),
    detection delay in iterations, missed switches and false alarms.
  - *Superseded:* the earlier GLR on the per-step hit stream (δ = 0.01, window
    500, warm-up/cooldown; `src/core/change_detection.py`, `--delta`). Its samples
    are autocorrelated (a hit puts the tracker at τ = 0, which makes the next hit
    likely), so it false-alarms: replayed in order it fired about every 3,700
    steps, shuffled never. Do not use it for new runs.
- **Report:** accuracy (upstream metric) per segment and overall, sensors/step,
  recovery time after each switch, restarts, and the oracle reference.
- **Oracle:** `examples/random_matrix_oracle.py --standard` — a tracker that
  knows the active model, run under this standard (infinite horizon, upstream
  and tracked metrics). It decides **per cell**: switch on every cell whose
  predicted probability is ≥ θ (one step ahead at delay 0, two steps at delay
  1), with θ swept to find the reward-optimal policy; θ = 0 watches every
  possible next cell. On the 4-slot models the reward-optimal oracle is full
  coverage: **100% (upstream) / 99.47% (tracked) at about 3 cells per step**
  (1550's O0–O3: 2.85–3.26 cells; 1530's table: 3.26). Compare PPO with this,
  not with the older uniform-m oracle (same m at every cell, 100-step episodes,
  tracked metric only), which understated what is achievable.

### 5×5 grid

As above on 5×5. **General base: none yet.** Run 1520 (equal-probability Moore
walk, 1,000 iterations, 2026-11-14) is **no longer recognised as the general
base**: it predates this standard and was evaluated differently (tracked-only
accuracy, evaluation clipped to k = 9 by lowest cell index, unseeded, one
overwritten checkpoint, history only in the log). Its checkpoint is kept
(`experiments/port_switching/general_base_5x5/agent_run1520_ppo`; paused run 1623
depends on it) but must not be used as a base for new runs. The 5×5 general base
under this standard is **run 1560** (`experiments/thesis_5x5/general_base/`):
equal-probability Moore walk, seed 1.

**General-base training length:** max(300 iterations, 40 iterations after
accuracy has plateaued), then finish open episodes (§1). Plateau = no rise of
≥ 0.5 points in the rolling mean of the last 3 evaluations (evaluations every 10
iterations). `training_grid_rect.py --iterations 300 --plateau-patience 40
--plateau-min-delta 0.005 --max-iterations 2000 --finish-episodes`.

**Schedule v1 (3 switches, equal intervals):**

| Segment | Model | Starts at iteration |
|---|---|---:|
| 1 | P0 = `O0` | 0 |
| 2 | P1 = `O1` | T0 = 150 |
| 3 | P2 = `O2` | T1 = 300 |
| 4 | P3 = `O3` | T2 = 450 |
| end | | ≥ 600 (then until open episodes end) |

`--matrices O0,O1,O2,O3 --matrix-seed 20261117 --switch-at 150,300,450`
(`examples/training_grid_rect.py`, which switches every worker env and the
evaluation env after the listed iterations and evaluates right before and
right after each switch). Run 1 (no restart) = run 1550, **cold start** (random
initial policy), `--finish-episodes`,
`experiments/thesis_5x5/cmd_no_restart.sh`. Oracle (`--standard`,
`experiments/thesis_5x5/oracle_standard.json`): 100% / 99.47% tracked on every
model at 2.85–3.26 cells per step. Equal intervals: the policy
cannot anticipate the switches either way, and equal segments make the
per-segment numbers directly comparable. 150 iterations ≈ 3,000 episodes ≈
600k steps per model; for scale, run 1530 (one model, from scratch) reached 90%
at iteration 70 and 98% at 140. Both runs use exactly this schedule.

**Evaluation:** 100 greedy episodes every 10 iterations (as §5/§6), plus one at
iteration 0, so the drop at each switch is visible. Per segment, report: the
accuracy just before and just after the switch, and the iterations from the
post-switch accuracy to 90 / 95 / 97 / 98 / 99% (first reached and stays above,
as in the single-switch test `experiments/robustness/`, runs 1540–1541), plus
sensors/step and the oracle for that model.

### 10×10 grid

The same design on 10×10. General base: run 1510 reached only 57% after 1,000
iterations — below what the learner reaches alone — so it needs more training
before it is a useful restart target.

### Still to settle before these runs

- **Switch times:** 5×5 set (v1 above); 10×10 to set.
- ~~Sensor budget~~: decided, no hard budget (see "Sensor budget k" below).
- **Script support** (`train_switching_grid_ppo.py`): schedule v1 is expressible
  today (`--iterations-per-segment 150 --passes 1`); arbitrary times still need
  `--switch-at`. Still missing for these runs: the upstream metric in its
  summaries, the §6-style evaluation (100 episodes every 10 iterations; it
  currently evaluates 1 episode every iteration), and the transition-stream
  detector hook for run 2 (in progress, another chat).

---

## 8. Tabular (online) models

The online learner is tabular SARSA(λ), `TabularTDAgent` in
`src/core/tabular_td.py`, updated after every step. These are the settings of
the current tabular experiments (random-matrix runs 1000–1302; scripts
`examples/train_switching_grid_online.py` for grids and
`examples/train_multiloop_online.py` for the IoBT graph). **Pass them
explicitly**: several script defaults differ (e.g. α 0.02 / 0.1, γ 0.95,
λ 0.8, ε 0.15, optimistic init 0).

| Setting | Value | Flag |
|---|---|---|
| Step size α | 0.05 | `--alpha 0.05` |
| Discount γ | 0.9 | `--gamma 0.9` |
| Trace λ | 0.0 (plain SARSA) | `--lam 0.0` |
| Exploration ε | 0.1 (ε-greedy) | `--epsilon 0.1` |
| Optimistic initial value | 5.0 | `--optimistic-init 5.0` |
| Action representation, grids | `local`: an exact table over subsets of the cells the target can have reached (Manhattan distance ≤ delay + 1) | `--action-mode local` |
| Action representation, IoBT | `subset`: an exact table over sensor subsets | (default in `train_multiloop_online.py`) |
| Sensor budget k | 4 or 5 (k = 5 is the main setting). Built into the action set: the table only contains subsets of at most k cells, so k is enforced in training and evaluation | `--max-sensors` |
| Sensor cost | −0.04 per sensor (best for online + DAL at k = 5 in the cost sweep, runs 1200–1225) | `--sensor-rew -0.04` |
| Time limit | 1 | `--time-limit 1` |
| Episodes per movement model | 1,000, 3 passes | `--episodes-per-segment 1000 --passes 3` |
| Evaluation | greedy (ε = 0), every 100 episodes, 10 episodes each; report greedy accuracy | `--greedy-eval-every 100 --greedy-eval-episodes 10` |
| Change detector (DAL) | GLR on the per-step hit stream, δ = 1e-4, window 500, minimum 30 samples, 5-episode warm-up and cooldown. **Superseded** (see §7): new tabular runs use the transition-stream detector, `examples/train_online_dal.py --detector transition` (δ_F 0.01, ρ 0.05; `experiments/transition_dal/`) | `--delta 1e-4` (others default) |
| Restart | warm: reload a base table trained 6,000 episodes on the equal-probability matrix C | `--restart-strategy warm --prior-path <table.npy>` |

**These experiments predate this standard and deviate from §1, §2 and §5:**
episodes are capped at 100 steps with no exit (`--max-ep-steps 100`, target
never leaves), movement is the 4-neighbour Dirichlet `R<i>` matrices (or IoBT
graph edges), and accuracy is hits while tracking (port metric). New tabular
experiments that must be comparable with the PPO runs need exits, the
9-candidate/4-slot movement and the upstream metric; the tabular scripts do not
support those yet.

---

## Not yet standardised (decision pending)

### Sensor budget k (decided: no hard budget)

**Facts.** In upstream Track-MDP, k (`max_sensors`) has **no effect** on RL
training or evaluation: it is only read by `generate_combination_lists()` /
`generate_combination_lists_new()` (`environment.py:191–232`), which build the
action set of the old tabular learner and are never called. The only control on
sensor use is the −0.16-per-sensor reward. The old port behaviour (k applied
only at evaluation, keeping the lowest-index cells) is retired: it scored a
policy on a pruned version of what it learned, understating accuracy.

**The two options** (`examples/training_grid_rect.py --sensor-budget`):

| | Option 1: no hard budget (`none`) | Option 2: budget enforced in the environment (`env`) |
|---|---|---|
| What happens | k never applied; energy controlled only by the per-sensor cost (optionally an extra penalty per sensor above k) | at most k tracking sensors per step; if the policy turns on more, the environment keeps k at random and bills only those; same rule in training and evaluation; the rescan is not budgeted |
| Formulation | upstream Track-MDP as it is | a hard constraint added to Track-MDP |
| Accuracy means | accuracy at whatever sensor use the cost settles on (always report sensors/step) | accuracy at budget k |
| General base (warm restart) | can watch all 9 equally likely candidates after a switch | covers at most k of 9 candidates (k = 6: about two-thirds while tracking), so a weaker restart target and less headroom |
| Adapted policy | the cost pushes it down to the ~3–4 cells a 4-slot model needs (run 1500: 4.1 sensors/step) | k = 6 ≥ the ≤4 next cells of a 4-slot model, so little loss expected |

**Test:** runs 1530 (option 1) and 1531 (option 2), 5×5, k = 6, one movement
model (seed 1), otherwise per this standard (`experiments/budget_test/`),
stopped at iteration 470 once both had converged:

| | 1530: no budget | 1531: k = 6 in the env |
|---|---|---|
| Iteration 10 | 78.8% at 7.1 sensors/step | 59.6% at 10.5 sensors/step |
| Reached 97% | iteration 110 | iteration 130 |
| Best (upstream metric) | 99.85% at 3.52 sensors/step (iteration 460) | 99.94% at 3.80 (iteration 290) |
| Last 10 evaluations | 99.47% (tracked 99.10%), 3.61 sensors/step | 99.37% (tracked 99.00%), 3.76 |

The budget **slowed early learning and did not change where the runs ended**.
Early on, an untrained policy turns on ~12 window cells; the budget keeps a
random 6, often dropping the target's cell, and the random truncation makes
credit assignment noisy until the policy learns to ask for ≤ k. Once adapted,
a policy needs only ~4 cells (≈3.3 possible next cells per cell under this
model), so k = 6 never binds. It would bind for a policy that must hedge over
more than k cells, i.e. the general base right after a switch.

**Decision: option 1 — no hard sensor budget** (`--sensor-budget none`, the
default). Energy is controlled by the per-sensor cost, as in upstream; always
report sensors/step next to accuracy. Option 2 remains available for
experiments that specifically need "accuracy at budget k".

### Other open items

| | Upstream Track-MDP | Run 1500 / `training_grid_rect.py` |
|---|---|---|
| Eval observation | action history forced to all ones (wrong after a miss) | real action history |
| Sensors/step at eval | unclipped; rescans counted as 9 although 100 are billed | was clipped to k; now follows the chosen budget option |
