# CLAUDE.md — Track-MDP IoBT

## Project overview

Track-MDP is a reinforcement learning system for tracking a moving object (vehicle/person)
across a network of acoustic sensor nodes. The RL agent decides which nodes to activate
at each timestep to maximize detections while minimizing sensor energy usage.

The IoBT variant uses 6 ReSpeaker microphone nodes (IDs 11–16) deployed at Camp Buckner.
Raw audio is recorded as FLAC files; GPS ground truth is collected simultaneously.

---

## Python environment

Always use the project's own venv:
```bash
./track_mdp_env/bin/python <script>
```
Never use system Python or conda — `soundfile`, `sklearn`, `ray[rllib]` are only in `track_mdp_env`.

---

## Key directories

| Path | Contents |
|------|----------|
| `iobt_data/` | FLAC files + GPS CSVs for each recording session |
| `results/multisession/` | Trained classifiers (pkl) + LOSO reports |
| `results/{session}/` | Per-session classifier outputs |
| `runs/` | RLlib fine-tuned checkpoints |
| `examples/agent_run{N}_ppo/` | Pretrained base checkpoints |
| `src/core/` | Core RL environment classes |
| `collection/` | Data processing + classifier training scripts |
| `examples/` | Training, fine-tuning, evaluation scripts |

---

## Data conventions

A **valid session** requires all of the following in `iobt_data/`:
- 6 FLAC files: `{session}_dvpg_gq_orin_{11..16}_respeaker.flac`
- 1 GPS CSV: `{session}_gps2_gps.csv`

Session ID format: `YYYYMMDD_HHMMSS` (first 15 chars of filename).

Currently valid sessions: `20260416_154037`, `20260417_100634`, `20260417_103802`

Adding a new session: drop the 7 files into `iobt_data/` — all scripts auto-discover.

---

## Classifier pipeline

Two types of classifiers are trained, serving different roles:

### 1. 6-class movement classifier  (`--clf-pkl`)
Predicts *which of the 6 nodes* the person is currently at.
Used as the movement model driving RL training rewards.

**Features**: 27 per timestep — 6 rolling means + 6 rolling stds + 15 pairwise diffs
(all nodes' signals jointly). Per-session z-score normalised.

**Train (multi-session)**:
```bash
# NOTE: This trains the OLD 6-class version — currently train_classifier_multisession.py
# produces per-node binary classifiers (see below). To retrain 6-class, use:
./track_mdp_env/bin/python collection/supervised_transition.py \
    --flac-dir  iobt_data \
    --nodes-txt iobt_data/node_positions.txt \
    --gps-csv   iobt_data/{session}_gps2_gps.csv \
    --session   {session} \
    --out-dir   results/{session} \
    --model rf
```

Latest 3-session 6-class classifier:
`results/multisession/classifier_multisession_20260509_195151.pkl`

---

### 2. Per-node binary classifiers  (`--node-clfs-dir`)
6 separate classifiers, one per node. Each answers "is the person at *this* node?"
Used as a binary detection gate: a node must be both activated by the RL policy
AND trigger its own classifier for a detection to count.

**Features**: 2 per node — rolling mean + rolling std of z-scored dB power (isolated,
no cross-node information).

**Train (multi-session, auto-discovers all sessions)**:
```bash
./track_mdp_env/bin/python collection/train_classifier_multisession.py \
    --flac-dir  iobt_data \
    --nodes-txt iobt_data/node_positions.txt \
    --window 5 \
    --out-dir results/multisession
```

Output: `results/multisession/node_clf_N{11..16}_multisession_{TIMESTAMP}.pkl`

Latest: `results/multisession/node_clf_N*_multisession_20260509_201606.pkl`

**Train (single-session)**:
```bash
./track_mdp_env/bin/python collection/train_node_classifiers.py \
    --flac-dir  iobt_data \
    --gps-csv   iobt_data/20260417_100634_gps2_gps.csv \
    --nodes-txt iobt_data/node_positions.txt \
    --session   20260417_100634 \
    --out-dir   results/20260417_100634
```

---

## Fine-tuning the RL agent

Main script: `examples/finetune_deterministic.py`

### Standard fine-tune with GPS-honest evaluation
```bash
./track_mdp_env/bin/python examples/finetune_deterministic.py \
    --clf-pkl        results/multisession/classifier_multisession_20260509_195151.pkl \
    --node-clfs-dir  results/multisession \
    --flac-dir       iobt_data \
    --nodes-txt      iobt_data/node_positions.txt \
    --gps-csv        iobt_data/20260417_100634_gps2_gps.csv \
    --session        20260417_100634 \
    --gps-eval \
    --run 210 \
    --no-moore
```

### Key flags

| Flag | Purpose |
|------|---------|
| `--run N` | Load checkpoint from `agent_run{N}_ppo/` as starting point |
| `--no-moore` | Must match the checkpoint's action space (run 210 was trained with this) |
| `--clf-pkl` | 6-class sklearn pkl — drives movement model for training rewards |
| `--node-clfs-dir` | Directory of `node_clf_N{nid}_*.pkl` — binary detection gate |
| `--gps-eval` | Evaluate accuracy vs GPS ground truth instead of classifier predictions |
| `--gps-csv` | GPS CSV required when `--gps-eval` is used |
| `--loso-only` | LOSO eval only, skip saving final classifiers |
| `--iterations N` | Number of PPO fine-tuning iterations (default: 200) |

### Observation/action space notes
- `--no-moore`: action space = `MultiDiscrete([2]*6)`, obs = 33-dim
- default (Moore): action space = `MultiDiscrete([2]*25)`, obs = 77-dim
- **Checkpoints are not compatible across these two modes.**
- Run 210 (`examples/agent_run210_ppo/`) uses `--no-moore`.

---

## Checkpoint inventory

| Run | Location | Notes |
|-----|----------|-------|
| 200 | `examples/agent_run200_ppo/` | Base rect 2×3, Moore |
| 201 | `examples/agent_run201_ppo/` | Fine-tuned from 200 |
| 210 | *(find with `find . -name "agent_run210*" -maxdepth 3`)* | no-moore, 6-node |
| 300 | `agent_run300_ppo/` | — |
| 301 | `examples/agent_run301_ppo/` | — |

---

## Important architecture notes

### Energy isolation
Per-node classifiers (`--node-clfs-dir`) are trained on each node's own signal
only — fully isolated. At inference time, the binary detection matrix is precomputed
for all nodes upfront (offline approximation). True hardware isolation would require
on-demand per-node inference only for activated nodes.

### GPS correspondence
The `evaluate_full_sequence()` function in `train_node_classifiers.py` measures
true classifier recall against GPS. Single-session recall is typically 95–100%
(classifier sees its own training data). LOSO cross-session recall is much lower
(node 11: ~50%, nodes 13–16: <20%) due to session-to-session acoustic variance.

### Missing state
When the RL tracker loses the object (`missing_state`), binary detection gating
is **bypassed** — the full rescan always succeeds. This prevents unbounded
`time_delay` overflow.

---

## Useful one-liners

```bash
# Check which sessions are valid (auto-discovery test)
./track_mdp_env/bin/python -c "
from pathlib import Path
import sys; sys.path.insert(0, '.')
from collection.train_classifier_multisession import discover_valid_sessions
print(discover_valid_sessions(Path('iobt_data')))
"

# Inspect a checkpoint's obs/action spaces
./track_mdp_env/bin/python -c "
import pickle
with open('examples/agent_run210_ppo/algorithm_state.pkl', 'rb') as f:
    s = pickle.load(f)
print(s)
" 2>&1 | head -20

# List all multisession classifier pkls
ls results/multisession/*.pkl
```
