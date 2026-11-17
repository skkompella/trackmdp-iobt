# Track-MDP for IoBT

Reinforcement learning for **target tracking with controlled sensing**, taken from
simulation onto a real Internet-of-Battlefield-Things (IoBT) sensor field. An agent
decides which sensor nodes to switch on at each timestep, trading tracking accuracy
against sensing energy.

Builds on *Track-MDP: Reinforcement Learning for Target Tracking with Controlled
Sensing* (Subramaniam, Gerogiannis, Hare, Veeravalli — ICASSP 2025,
[paper](https://ieeexplore.ieee.org/document/10890122)).

> Status: research code, work in progress.

---

## Field work: the IoBT testbed

The tracker was built and evaluated against live sensor nodes at the IoBT testing
facility (Camp Buckner, West Point), not only in simulation. Each node is an edge
computer (Jetson Orin on the 2026 nodes) carrying a **ReSpeaker microphone array** and, on the
10-node IoBT-MAX deployment, a **ZED stereo camera running YOLO**. A GPS tracker on
the moving target (person or vehicle) gives ground truth.

### Deployments recorded

| Data | Nodes | Sensors | Location in repo |
|---|---|---|---|
| IoBT-MAX sessions (2024, 2025) | 10 (`node1`–`node10`) | ReSpeaker audio (FLAC), ZED camera + YOLO detections, GT tracks | `iobt_data_10/<session>/` |
| ReSpeaker sessions (2026) | 6 (`dvpg_gq_orin_11`–`16`) | 6-channel ReSpeaker audio (FLAC), GPS CSV | `iobt_data/` |

Field data is not tracked in git. Scripts auto-discover sessions dropped into these
folders (see `CLAUDE.md` for the naming conventions).

### Real-time collection over MQTT

The nodes publish their sensor outputs to an MQTT broker on the field network. Two
collectors subscribe to every node live:

- **`collection/mqtt_collector.py`** — subscribes to `/<node>/respeaker/power` on each
  ReSpeaker node, downsamples to one reading per speaker per second, and writes a
  time-aligned `(steps × nodes)` power array with Unix timestamps, per-step JSONL
  records and metadata.
- **`collection/iobt_collector.py`** — subscribes to `/<node>/analytics/yolo/bbox` on
  all ten nodes and writes every YOLO detection message (class, confidence, box,
  depth, world coordinates, capture time) to a per-node JSONL file with a
  receiver-side timestamp, flushing cleanly on shutdown.

### From asynchronous streams to Track-MDP input

The nodes report independently and at different rates, so the raw streams are
turned into one synchronized timeline before the tracker sees them:

- **`iobt_step_builder.py`** — buckets each node's detections into fixed 0.5 s steps
  by capture time and produces an aligned `(T × 10)` detection array plus timestamps,
  filtered by class and confidence.
- **`flac_to_trackmdp.py`** — audio path: 16 kHz 6-channel FLAC → mic channels 1–4 →
  DC removal → dB power per window → rolling mean → Gaussian path-loss Bayesian
  posterior over source positions → most likely node per step.
- **`gps_to_sequence.py`** — maps the GPS track to the nearest node per step, giving
  the ground-truth sequence.
- **Per-node acoustic classifiers** (`train_node_classifiers*.py`,
  `train_classifier_multisession.py`) — each node decides "target present" from its own
  signal only, so a node's detection never depends on another node being awake. A
  pooled LightGBM variant shares the "target present" signal across sparse nodes.
- **Camera/audio fusion** (`examples/finetune_deterministic.py --soft-reward
  --fusion-mode ...`) — the YOLO camera score and the calibrated audio probability are
  fused per node (e.g. `or_max`) into a soft detection probability used as the
  tracker's reward. Audio fills in steps where the camera cannot see the target.

### Evaluating on field data

- **`collection/iobt_field_eval.py`** replays a recorded session through a trained
  policy: the recorded detections drive the target's position, and a step counts as
  tracked only when the policy had the right node switched on.
- **`--gps-eval`** scores accuracy against GPS ground truth rather than against the
  classifiers' own labels.

The best GPS-scored policy tracks the target on nearly every step while activating
about 3 of 10 nodes. That result is in-domain (trained and evaluated on the same
session). Generalising to unseen sessions is the main open problem on the field side.

---

## How Track-MDP works

- **State** = `(last node the object was seen at, time since it was seen)`, encoded as
  `node * (time_limit + 1) + time_delay`, plus a `missing_state` when the track is lost.
  The missing state triggers a full rescan (every sensor on).
- **Action** = a binary vector over nodes (`MultiDiscrete([2]*n)`), clipped to
  `max_sensors`.
- **Reward** = a bonus for detecting the object minus a per-sensor energy cost.
- **Metrics** = tracking accuracy (fraction of steps the object is under an active
  sensor) and sensors activated per step.

Agents are trained with PPO (Ray RLlib 2.54, old API stack) or, for the online
experiments, with a tabular SARSA learner.

---

## Simulation: adapting when the target's movement changes

Track-MDP assumes the target's motion is stationary. The simulation work drops that
assumption: one agent tracks a target whose movement pattern switches among several
random transition matrices (5×5 and 10×10 grids, and the IoBT node graph).
A GLR change detector watches the tracker's hit/miss stream and, when it fires,
restarts the learner from a base model trained on an equal-probability random walk
(DAL-style detect-and-restart).

What the experiments show, in short:

- Restarting to a good base model is what helps. The gain tracks how much better the
  base model is than the learner on its own; detector timing matters little.
- The hit/miss signal over-fires and barely separates real switches from noise; tuning
  the threshold δ does not fix that (`experiments/random_matrix/delta_tuning/`).
- Differences from the original Track-MDP simulator: no target exit (fixed 100-step
  episodes) and 4-neighbour rather than 3×3 movement.

Full write-ups: `update_oct.md`, `experiments/random_matrix/results.md` and the other
folders under `experiments/`.

---

## Repository layout

| Path | Contents |
|------|----------|
| `collection/` | Field pipeline: MQTT collectors, step builder, audio processing, GPS alignment, classifier training, field replay evaluation |
| `src/core/` | Environments, learners and change-detection code |
| `examples/` | Training, fine-tuning and evaluation scripts |
| `tests/` | pytest suite |
| `experiments/` | Per-run notes and result write-ups |
| `CLAUDE.md` | Operational guide: data conventions, classifier pipeline, fine-tuning flags, checkpoint inventory |

### Key modules in `src/core/`

| Module | Purpose |
|--------|---------|
| `environment.py`, `grid_env_rect.py`, `gym_wrapper*.py` | Original grid Track-MDP environment and Gymnasium wrappers |
| `iobt_environment.py`, `iobt_6node_env.py`, `respeaker_env.py` | IoBT environments driven by field recordings |
| `iobt_loops.py`, `multiloop_iobt_env.py`, `equalprob_iobt_env.py` | IoBT node graph, synthetic loop movement, equal-probability random walk |
| `matrix_iobt_env.py`, `random_transition_matrices.py` | IoBT graph and grids with switchable random transition matrices |
| `switching_grid_env.py`, `grid_transition_matrices.py` | Grids with switchable transition matrices |
| `tabular_td.py` | Online tabular SARSA(λ) agent (subset, factored and local action modes) |
| `change_detection.py`, `detection_signals.py`, `detector_eval.py` | GLR change detector, restart controller, candidate signals, offline calibration |
| `online_schedule.py` | Shared episode rollout, greedy evaluation and metrics |

---

## Setup

```bash
python -m venv track_mdp_env
./track_mdp_env/bin/pip install -r requirements.txt
./track_mdp_env/bin/pip install -e .
./track_mdp_env/bin/python -m pytest -q tests
```

Field data (`iobt_data/`, `iobt_data_10/`), trained classifiers (`results/`) and RLlib
checkpoints (`runs/`) are not tracked in git.

---

## Common entry points

```bash
# Live collection on the field network
./track_mdp_env/bin/python collection/mqtt_collector.py --broker <broker-ip>
./track_mdp_env/bin/python collection/iobt_collector.py --broker <broker-ip>

# Build aligned 0.5 s step arrays from collected YOLO detections
./track_mdp_env/bin/python collection/iobt_step_builder.py

# Fine-tune PPO on a recorded session, scored against GPS (see CLAUDE.md for flags)
./track_mdp_env/bin/python examples/finetune_deterministic.py --gps-eval ...

# Replay a recorded session through a trained policy
./track_mdp_env/bin/python collection/iobt_field_eval.py --detections ... --timestamps ... --checkpoint ...

# Simulation: switching movement patterns with change detection + restart
./track_mdp_env/bin/python examples/train_switching_grid_ppo.py ...
./track_mdp_env/bin/python examples/train_switching_grid_online.py ...
./track_mdp_env/bin/python examples/train_multiloop_online.py ...
```

---

## License

MIT — see [LICENSE](LICENSE).
