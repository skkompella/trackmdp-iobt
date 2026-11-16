#!/usr/bin/env python
"""
train_online_dal.py — CLI for src/core/online_dal.py: online tabular SARSA on a
switching-matrix schedule with the transition-stream change detector.

Example (5x5 grid, detector only, no restarts):
    ./track_mdp_env/bin/python examples/train_online_dal.py --env grid \\
        --detector transition --detector-mode monitor --probe-rate 0.2 \\
        --max-sensors 5 --sensor-rew -0.04 --out-dir experiments/transition_dal/demo

Pause: touch <out-dir>/PAUSE (or send SIGTERM); continue with the same
arguments plus --resume.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.online_dal import build_parser, run_online  # noqa: E402


def main():
    run_online(build_parser().parse_args())


if __name__ == "__main__":
    main()
