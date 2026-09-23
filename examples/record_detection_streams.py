#!/usr/bin/env python
"""
record_detection_streams.py — capture candidate change-detection signals.

Detector evaluation currently costs a full RL run. It need not: record the
streams once, then replay them through thousands of configurations offline.
That is what makes a real sweep affordable, and it removes RL seed noise from
the comparison because every configuration sees identical data.

Four conditions, and the split matters:

                   switching schedule        stationary (no switches)
  frozen policy    pure detectability        clean false-alarm reference
  learning policy  the realistic case        false-alarm reference with drift

The STATIONARY condition is what does not exist today and without which a
false-alarm rate cannot be measured at all — every alarm on it is by definition
false. The FROZEN/LEARNING split separates "this switch is undetectable" from
"this switch is masked by the learner's own improvement", which are currently
confounded in a single number.

Everything recorded is derived from what the agent legitimately observes. A hit
is read off the returned time_delay (0 means detected), and the observed cell
from next_state, which the environment sets to the object's true cell on every
successful detection and which the agent already uses as its own state. The
object's position during a MISS is never touched, and the active regime index
is stored only as ground truth for scoring, never as a signal.

Example
-------
    ./track_mdp_env/bin/python examples/record_detection_streams.py \\
        --env grid --condition learning --schedule switching
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.detection_signals import SignalRecorder          # noqa: E402
from src.core.grid_transition_matrices import (                # noqa: E402
    make_matrix_A, make_matrix_B, make_matrix_C,
)
from src.core.iobt_loops import sample_loops                   # noqa: E402
from src.core.multiloop_iobt_env import MultiLoopIoBTEnv       # noqa: E402
from src.core.switching_grid_env import SwitchingTransitionGridEnv  # noqa: E402
from src.core.tabular_td import TabularTDAgent                 # noqa: E402

MATRIX_MAKERS = {"A": make_matrix_A, "B": make_matrix_B, "C": make_matrix_C}


def record_episode(agent, env, recorder, cfg, learn: bool, explore: bool):
    """
    One episode, recording every candidate signal.

    Mirrors src/core/online_schedule.run_episode exactly in its state machine,
    but derives `hit` from the returned time_delay rather than reading
    env.object_pos. Same value; no privileged read.
    """
    time_limit    = cfg["time_limit"]
    missing_state = cfg["missing_state"]
    max_sensors   = cfg["max_sensors"]

    env.reset_object_state()
    current_state = missing_state
    time_delay    = 0
    prev = None

    for _ in range(cfg["max_ep_steps"]):
        was_missing = (current_state == missing_state)
        state_pos = agent.n_nodes if was_missing else current_state // (time_limit + 1)
        state_time = 0 if was_missing else current_state % (time_limit + 1)

        s = agent.obs_to_state((state_pos, state_time, None))
        action = agent.select_action(s, explore=explore)
        action_sensors = agent.action_vector(action)

        reward, next_state, _term, new_delay = env.get_reward_next_state(
            current_state, action_sensors, time_delay)

        # Honest derivation: the tracker re-acquired iff staleness reset to 0.
        detected = (new_delay == 0)
        cell = int(next_state // (time_limit + 1)) if detected else None
        # `hit` keeps the existing convention (a hit only counts while tracking)
        # so this stream is comparable with what the live detector consumed.
        hit = int(detected and not was_missing)
        sensors_on = max_sensors if was_missing else int(action_sensors.sum())

        recorder.step(hit=hit, cell=cell, delay=time_delay, reward=reward,
                      sensors=sensors_on)

        if learn:
            if prev is not None:
                agent.observe(prev[0], prev[1], prev[2], s, action, done=False)
            prev = (s, action, reward)

        current_state = next_state
        time_delay = new_delay

    if learn and prev is not None:
        agent.observe(prev[0], prev[1], prev[2], prev[0], prev[1], done=True)
        agent.end_episode()


def build_grid(args):
    n_cells = args.nrows * args.ncols
    tl = args.time_limit
    cfg = {"time_limit": tl, "missing_state": n_cells * (tl + 1) + 1,
           "max_ep_steps": args.max_ep_steps, "max_sensors": args.max_sensors,
           "n_cells": n_cells}
    names = [n.strip().upper() for n in args.matrices.split(",") if n.strip()]
    regimes = [MATRIX_MAKERS[n](args.nrows, args.ncols) for n in names]
    env = SwitchingTransitionGridEnv(
        args.nrows, args.ncols, args.max_sensors, args.max_sensors,
        cfg["missing_state"], tl, regimes, seed=args.seed,
        no_moore_constraint=True)
    return cfg, env, names, env.force_matrix


def build_iobt(args):
    tl = args.time_limit
    n_grid = 16
    cfg = {"time_limit": tl, "missing_state": n_grid * (tl + 1) + 1,
           "max_ep_steps": args.max_ep_steps, "max_sensors": args.max_sensors,
           "n_cells": 10}
    loops = sample_loops(args.num_loops, seed=args.loop_seed)
    env = MultiLoopIoBTEnv(args.max_sensors, args.max_sensors,
                           cfg["missing_state"], tl, loops, seed=args.seed,
                           no_moore_constraint=True, reward_preset="grid")
    return cfg, env, [str(i) for i in range(len(loops))], env.force_loop


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["grid", "iobt"], default="grid")
    p.add_argument("--condition", choices=["learning", "frozen"],
                   default="learning")
    p.add_argument("--schedule", choices=["switching", "stationary"],
                   default="switching")
    # grid
    p.add_argument("--nrows", type=int, default=5)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--matrices", type=str, default="A,B")
    # iobt
    p.add_argument("--num-loops", type=int, default=10)
    p.add_argument("--loop-seed", type=int, default=20260921)
    # shared
    p.add_argument("--time-limit", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=6)
    p.add_argument("--max-ep-steps", type=int, default=100)
    p.add_argument("--episodes-per-segment", type=int, default=1024)
    p.add_argument("--passes", type=int, default=3)
    # learner
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--optimistic-init", type=float, default=5.0)
    p.add_argument("--frozen-q", type=str, default=None,
                   help="Q-weights .npy for --condition frozen")
    p.add_argument("--surprise-decay", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    cfg, env, names, force = (build_grid(args) if args.env == "grid"
                              else build_iobt(args))

    action_mode = "factored" if args.env == "grid" else "subset"
    agent = TabularTDAgent(
        time_limit=cfg["time_limit"], max_sensors=args.max_sensors,
        missing_state=cfg["missing_state"], alpha=args.alpha, gamma=args.gamma,
        lam=args.lam, epsilon=args.epsilon,
        optimistic_init=args.optimistic_init, seed=args.seed,
        n_nodes=cfg["n_cells"], grid_width=cfg["n_cells"] if args.env == "grid" else 16,
        action_mode=action_mode)

    if args.condition == "frozen":
        if not args.frozen_q:
            raise SystemExit("--condition frozen requires --frozen-q")
        agent.restore(np.load(args.frozen_q))

    learn = args.condition == "learning"
    recorder = SignalRecorder(n_cells=cfg["n_cells"],
                              decay=args.surprise_decay)

    # A stationary schedule pins ONE regime for the whole run, so every alarm
    # it produces is false by construction. That is the false-alarm reference.
    segments = ([(p_, i) for p_ in range(args.passes)
                 for i in range(len(names))]
                if args.schedule == "switching"
                else [(p_, 0) for p_ in range(args.passes * len(names))])

    print("=" * 74)
    print(f"  RECORDING DETECTION STREAMS — {args.env}, {args.condition}, "
          f"{args.schedule}")
    print("=" * 74)
    print(f"  regimes           : {names}")
    print(f"  segments          : {len(segments)} x "
          f"{args.episodes_per_segment} episodes x {args.max_ep_steps} steps")
    print(f"  learning          : {learn}")
    print(f"  surprise decay    : {args.surprise_decay}\n")

    for seg_i, (pass_i, regime) in enumerate(segments):
        force(regime)
        if seg_i > 0 and args.schedule == "switching":
            recorder.mark_switch()
        for _ in range(args.episodes_per_segment):
            record_episode(agent, env, recorder, cfg, learn=learn,
                           explore=learn)
        done = recorder.total_steps
        hits = recorder.streams()["hit"]["values"][-args.episodes_per_segment
                                                  * args.max_ep_steps:]
        print(f"  segment {seg_i} (regime {names[regime]}): "
              f"{done:,} steps, hit rate {np.mean(hits)*100:5.2f}%")

    meta = {"env": args.env, "condition": args.condition,
            "schedule": args.schedule, "regimes": names,
            "max_sensors": args.max_sensors, "time_limit": args.time_limit,
            "episodes_per_segment": args.episodes_per_segment,
            "max_ep_steps": args.max_ep_steps,
            "surprise_decay": args.surprise_decay, "seed": args.seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    recorder.save(args.out, meta=meta)

    st = recorder.streams()
    print(f"\n  total steps       : {recorder.total_steps:,}")
    print(f"  switch steps      : {recorder.switch_steps}")
    print("  stream sample counts (differing rates are the point):")
    for name, s in st.items():
        print(f"    {name:>20}: {len(s['values']):>8,}")
    print(f"\n  Wrote {args.out}")


if __name__ == "__main__":
    main()
