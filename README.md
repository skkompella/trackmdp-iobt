# Track-MDP for IoBT

Reinforcement learning for **target tracking with controlled sensing**, applied to an
Internet-of-Battlefield-Things (IoBT) acoustic sensor network. An agent decides which
sensor nodes to switch on at each timestep, trading detection accuracy against energy.

Builds on *Track-MDP: Reinforcement Learning for Target Tracking with Controlled
Sensing* (Subramaniam, Gerogiannis, Hare, Veeravalli — ICASSP 2025,
[paper](https://ieeexplore.ieee.org/document/10890122)).

> Status: research code, work in progress.

---

## How Track-MDP works

- **State** = `(last node the object was seen at, time since it was seen)`, encoded as
  `node * (time_limit + 1) + time_delay`, plus a `missing_state` when the track is lost
  (which triggers a full rescan).
- **Action** = a binary vector over nodes (`MultiDiscrete([2]*n)`), clipped to
  `max_sensors`.
- **Reward** = a bonus for detecting the object minus a per-sensor energy cost.
- **Metrics** = tracking accuracy (fraction of steps the object is under an active
  sensor) and sensors activated per step. Episodes start in the missing state, so the
  accuracy ceiling is `(max_ep_steps - 1) / max_ep_steps` (99% at 100 steps).

Agents are trained with PPO (Ray RLlib 2.54, old API stack) or, for the online
experiments, with a tabular SARSA learner.

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/core/` | Environments, learners and change-detection code (see below) |
| `examples/` | Training, fine-tuning and evaluation scripts |
| `collection/` | Field-data processing: FLAC/GPS ingestion, per-node and 6-class acoustic classifiers |
| `tests/` | pytest suite |
| `experiments/` | Per-run notes and result write-ups (`index.md`, `synthetic_multiloop/`, `switching_grid/`, `detector_calibration/`) |
| `CLAUDE.md` | Operational guide: data conventions, classifier pipeline, fine-tuning flags, checkpoint inventory |

### Key modules in `src/core/`

| Module | Purpose |
|--------|---------|
| `environment.py`, `grid_env_rect.py`, `gym_wrapper*.py` | Original grid Track-MDP environment and Gymnasium wrappers |
| `iobt_environment.py`, `iobt_6node_env.py`, `respeaker_env.py` | IoBT environments driven by field recordings (ReSpeaker nodes, GPS ground truth) |
| `iobt_loops.py` | IoBT node graph (`IOBT_MAP`), cycle enumeration, seeded loop sampling |
| `multiloop_iobt_env.py` | Synthetic IoBT env: the object follows one of N loops, chosen per episode |
| `equalprob_iobt_env.py` | The object random-walks the IoBT graph (stay or any neighbour, equal probability) |
| `grid_transition_matrices.py`, `switching_grid_env.py` | 5×5 grid with switchable object transition matrices (A, B, C) |
| `tabular_td.py` | Online tabular SARSA(λ) agent (subset or factored action representation) |
| `change_detection.py` | GLR change detector (Bernoulli / Gaussian), ARL0 calibration, restart controller (cold / warm / library) |
| `detection_signals.py`, `detector_eval.py` | Candidate detection signals, offline replay, operating-curve evaluation |
| `online_schedule.py` | Shared episode rollout and online metrics |

---

## Setup

```bash
python -m venv track_mdp_env
./track_mdp_env/bin/pip install -r requirements.txt
./track_mdp_env/bin/pip install -e .
./track_mdp_env/bin/python -m pytest -q tests
```

Field data (`iobt_data/`), trained classifiers (`results/`) and RLlib checkpoints
(`runs/`) are not tracked in git.

---

## Common entry points

```bash
# Fine-tune a PPO checkpoint on a recorded IoBT session (see CLAUDE.md for all flags)
./track_mdp_env/bin/python examples/finetune_deterministic.py --run 210 --no-moore --gps-eval ...

# Train one PPO agent across 10 synthetic IoBT loops
./track_mdp_env/bin/python examples/train_multiloop_synthetic.py

# Online learning across a sequence of loops, with optional change detection + restart
./track_mdp_env/bin/python examples/train_multiloop_online.py

# 5x5 switching-grid experiments (tabular and PPO)
./track_mdp_env/bin/python examples/train_switching_grid_online.py
./track_mdp_env/bin/python examples/train_switching_grid_ppo.py

# Change-detector calibration: record streams once, sweep offline
./track_mdp_env/bin/python examples/record_detection_streams.py --env grid --out ...
./track_mdp_env/bin/python examples/calibrate_detector.py --switching ... --stationary ... --out ...
```

---

## Current findings

Full write-ups live in `experiments/`.

- **Synthetic multi-loop IoBT.** PPO run 241, trained on an equal-probability random
  walk, reaches the 99% accuracy ceiling on all ten loops without seeing them, at
  `max_sensors=6`. `max_sensors` is the effective lever on sensors per step; scaling
  the sensor reward is not.
- **Hedging.** With a large enough sensor budget, the equal-probability model watches
  nearly every neighbour of the object's last position (≈5 nodes at k=6), so it
  generalises across loops (93.8% mean for the tabular version). A per-loop
  specialist uses fewer sensors and scores 99% on its own loop, but only 49.7% across
  all loops.
- **Change detection + restart.** Resetting to a good base model ("warm restart") is
  what helps; detector timing barely matters. The gain from warm restart tracks
  *headroom* (base-model accuracy minus what the learner reaches alone), correlation
  0.90 across six settings: +24 pp on the 5×5 grid (PPO), ≈+1 pp on IoBT, and
  negative when the base model is worse than the learner.
- **Detector calibration.** Signals must be compared at a matched false-alarm rate
  (ARL0), not a matched δ. The best signal depends on whether the policy can hedge:
  `hit` is best on IoBT at k=2, `transition_surprise` on the 5×5 grid at k=6. No tuned
  constant transfers between environments; the calibration procedure does.

---

## License

MIT — see [LICENSE](LICENSE).
