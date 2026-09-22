#!/usr/bin/env python
"""
train_multiloop_online.py — online tabular TD on a SEQUENTIAL loop schedule.

The object walks loop 0 for N episodes, then loop 1 for N episodes, and so on
through every loop, repeating the cycle for several passes.  A tabular
SARSA(lambda) learner updates every step, and a GLR change detector watches the
per-step tracking-success stream; when it fires, the learner is restarted
according to --restart-strategy.

This is the contrast with PPO.  PPO's unit of adaptation is a 512-step batch and
its clipped objective exists specifically to bound how far the policy moves, so
it cannot respond to a trajectory switch in a few steps.  A table updated per
step, with no trust region, can.

What gets measured
------------------
Adaptation latency: episodes after a switch until per-episode accuracy returns
within --latency-epsilon of that segment's steady state.  Reported per switch,
per pass, and as a distribution — plus detector precision/recall against the
KNOWN switch times, which we can compute exactly because we set the schedule.

Example
-------
    ./track_mdp_env/bin/python examples/train_multiloop_online.py \\
        --max-sensors 2 --reward-preset grid --episodes-per-loop 50 --passes 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.change_detection import (                       # noqa: E402
    GLRChangeDetector, RestartController,
)
from src.core.iobt_loops import describe_loops, sample_loops  # noqa: E402
from src.core.multiloop_iobt_env import (                     # noqa: E402
    IOBT_N, REWARD_PRESETS, MultiLoopIoBTEnv,
)
from src.core.online_schedule import (                        # noqa: E402
    adaptation_latency, detector_scores, run_episode,
)
from src.core.tabular_td import N_GRID, TabularTDAgent        # noqa: E402


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # environment / loop family
    p.add_argument("--num-loops", type=int, default=10)
    p.add_argument("--loop-seed", type=int, default=20260921)
    p.add_argument("--time-limit", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=2)
    p.add_argument("--reward-preset", type=str, default="grid",
                   choices=sorted(REWARD_PRESETS))
    p.add_argument("--sensor-rew", type=float, default=None)
    # schedule
    p.add_argument("--episodes-per-loop", type=int, default=50)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--max-ep-steps", type=int, default=100)
    # learner
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--optimistic-init", type=float, default=0.0)
    # detector
    p.add_argument("--no-detector", action="store_true",
                   help="Baseline: pure continual learning, never restart")
    p.add_argument("--oracle-restart", action="store_true",
                   help="Control: restart at the KNOWN switch points instead of "
                        "using the detector. Separates 'restarting is harmful' "
                        "from 'the detector misfires' - with a perfect detector "
                        "this is the best any restart policy could do")
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--detector-warmup-episodes", type=int, default=5,
                   help="Do not feed the detector until the learner has had "
                        "this many episodes to stabilise")
    p.add_argument("--detector-cooldown-episodes", type=int, default=5,
                   help="Refractory period after a restart. Without it the "
                        "learner's own relearning looks like a change, trips "
                        "the detector, and restarts it again - a feedback loop")
    p.add_argument("--restart-strategy", type=str, default="cold",
                   choices=list(RestartController.STRATEGIES))
    p.add_argument("--prior-path", type=str, default=None,
                   help="Q-table .npy for --restart-strategy warm")
    p.add_argument("--save-q", type=str, default=None,
                   help="Write the final Q-table here (to build a warm prior)")
    # eval / bookkeeping
    p.add_argument("--latency-epsilon", type=float, default=0.05)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--new-run", type=int, default=810)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    loops = sample_loops(args.num_loops, seed=args.loop_seed)
    tl = args.time_limit
    cfg = {
        "time_limit":    tl,
        "missing_state": IOBT_N * IOBT_N * (tl + 1) + 1,
        "max_ep_steps":  args.max_ep_steps,
        "max_sensors":   args.max_sensors,
    }

    env = MultiLoopIoBTEnv(
        args.max_sensors, args.max_sensors, cfg["missing_state"], tl,
        loops, seed=args.train_seed, no_moore_constraint=True,
        reward_preset=args.reward_preset, sensor_rew=args.sensor_rew,
    )
    agent = TabularTDAgent(
        time_limit=tl, max_sensors=args.max_sensors,
        missing_state=cfg["missing_state"], alpha=args.alpha, gamma=args.gamma,
        lam=args.lam, epsilon=args.epsilon,
        optimistic_init=args.optimistic_init, seed=args.train_seed,
    )

    prior = np.load(args.prior_path) if args.prior_path else None
    controller = RestartController(agent, strategy=args.restart_strategy,
                                   prior=prior)
    detector = None if (args.no_detector or args.oracle_restart) else \
        GLRChangeDetector(delta=args.delta, min_samples=args.min_samples)

    out_dir = args.out_dir or os.path.join(
        project_root, "experiments", "synthetic_multiloop", f"run{args.new_run}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print("  TRACK-MDP — ONLINE TABULAR TD ON A SEQUENTIAL LOOP SCHEDULE")
    print("=" * 74)
    print(describe_loops(loops))
    print(f"\n  Q-table           : {agent.n_states} states x "
          f"{agent.n_actions} actions = {agent.q.size:,} entries")
    print(f"  time_limit        : {tl}")
    print(f"  max_sensors       : {args.max_sensors} of 10")
    print(f"  reward preset     : {args.reward_preset} "
          f"(track={env.tracking_rew}, sensor={env.sensor_rew})")
    print(f"  SARSA(lambda)     : alpha={args.alpha} gamma={args.gamma} "
          f"lambda={args.lam} eps={args.epsilon}")
    if args.oracle_restart:
        _det_label = "ORACLE (restart at known switches, no detector)"
    elif detector is None:
        _det_label = "DISABLED (continual-learning baseline)"
    else:
        _det_label = f"GLR delta={args.delta}, min_samples={args.min_samples}"
    print(f"  Detector          : {_det_label}")
    print(f"  Restart strategy  : {args.restart_strategy}")
    print(f"  Schedule          : {args.passes} passes x {args.num_loops} loops "
          f"x {args.episodes_per_loop} episodes\n")

    # ── the schedule ───────────────────────────────────────────────────────
    history, segments = [], []
    fired_episodes, switch_episodes = [], []
    ep = 0
    suppress_until = 0   # episode index until which detection is muted

    print(f"  {'pass':>4}  {'loop':>4}  {'acc':>7}  {'sensors':>8}  "
          f"{'latency':>8}  {'restarts':>8}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    for pass_idx in range(args.passes):
        for loop_idx in range(args.num_loops):
            env.force_loop(loop_idx)
            if ep > 0:
                switch_episodes.append(ep)
                if args.oracle_restart:
                    # Perfect information: restart precisely at the switch.
                    controller.restart(segment_reward=float(
                        np.mean(segments[-1]["mean_accuracy"]) if segments else 0.0))
                    fired_episodes.append(ep)
            seg_acc, seg_sens, seg_reward = [], [], 0.0

            for _ in range(args.episodes_per_loop):
                res = run_episode(agent, env, cfg, learn=True, explore=True)
                seg_acc.append(res["accuracy"])
                seg_sens.append(res["sensors"])
                seg_reward += res["reward"]
                history.append({"episode": ep, "pass": pass_idx,
                                "loop": loop_idx, "accuracy": res["accuracy"],
                                "sensors": res["sensors"],
                                "reward": res["reward"]})

                # Suppress detection during warmup and for a refractory window
                # after each restart: while the learner is still climbing, its
                # own success rate is non-stationary, and feeding that to the
                # detector produces restart -> relearn -> detect loops rather
                # than genuine changepoints.
                quiet = (ep < args.detector_warmup_episodes
                         or ep < suppress_until)
                if detector is not None and not quiet:
                    for d in res["detections"]:
                        if detector.update(d):
                            fired_episodes.append(ep)
                            controller.restart(
                                segment_reward=float(np.mean(seg_acc)))
                            suppress_until = ep + 1 + args.detector_cooldown_episodes
                            break
                elif detector is not None:
                    detector.reset()
                ep += 1

            lat = adaptation_latency(seg_acc, args.latency_epsilon)
            segments.append({
                "pass": pass_idx, "loop": loop_idx,
                "start_episode": ep - args.episodes_per_loop,
                "mean_accuracy": float(np.mean(seg_acc)),
                "final_accuracy": float(np.mean(seg_acc[-10:])),
                "mean_sensors": float(np.mean(seg_sens)),
                "adaptation_latency": lat,
                "segment_reward": seg_reward,
            })
            print(f"  {pass_idx:>4}  {loop_idx:>4}  "
                  f"{np.mean(seg_acc)*100:6.2f}%  {np.mean(seg_sens):8.2f}  "
                  f"{('n/a' if lat is None else lat):>8}  "
                  f"{controller.n_restarts:>8}")

    # ── summary ────────────────────────────────────────────────────────────
    lats = [s["adaptation_latency"] for s in segments
            if s["adaptation_latency"] is not None]
    first_pass = [s for s in segments if s["pass"] == 0]
    later      = [s for s in segments if s["pass"] > 0]

    def _mean_lat(rows):
        v = [r["adaptation_latency"] for r in rows
             if r["adaptation_latency"] is not None]
        return float(np.mean(v)) if v else None

    summary = {
        "run": args.new_run,
        "max_sensors": args.max_sensors,
        "time_limit": tl,
        "reward_preset": args.reward_preset,
        "sensor_rew": env.sensor_rew,
        "detector_enabled": detector is not None,
        "oracle_restart": bool(args.oracle_restart),
        "restart_strategy": args.restart_strategy,
        "n_restarts": controller.n_restarts,
        "episodes_per_loop": args.episodes_per_loop,
        "passes": args.passes,
        "q_entries": int(agent.q.size),
        "mean_accuracy": float(np.mean([s["mean_accuracy"] for s in segments])),
        "mean_sensors": float(np.mean([s["mean_sensors"] for s in segments])),
        "mean_latency_episodes": float(np.mean(lats)) if lats else None,
        "median_latency_episodes": float(np.median(lats)) if lats else None,
        "segments_never_recovered": len(segments) - len(lats),
        "mean_latency_first_pass": _mean_lat(first_pass),
        "mean_latency_later_passes": _mean_lat(later),
        "ceiling": (args.max_ep_steps - 1) / args.max_ep_steps,
    }
    if detector is not None or args.oracle_restart:
        summary["detector"] = detector_scores(
            fired_episodes, switch_episodes,
            tolerance=max(2, args.episodes_per_loop // 5))

    print("\n" + "=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print(f"  Mean accuracy        : {summary['mean_accuracy']*100:.2f}%  "
          f"(ceiling {summary['ceiling']*100:.2f}%)")
    print(f"  Mean sensors/step    : {summary['mean_sensors']:.2f}")
    if lats:
        print(f"  Adaptation latency   : mean {summary['mean_latency_episodes']:.1f}, "
              f"median {summary['median_latency_episodes']:.1f} episodes")
        print(f"    first pass         : {summary['mean_latency_first_pass']}")
        print(f"    later passes       : {summary['mean_latency_later_passes']}")
    print(f"  Segments not recovered: {summary['segments_never_recovered']} "
          f"of {len(segments)}")
    print(f"  Restarts             : {controller.n_restarts}")
    if "detector" in summary:
        d = summary["detector"]
        print(f"  Detector             : {d['true_positive']}/{d['n_switches']} "
              f"switches caught, {d['false_alarms']} false alarms")

    with open(os.path.join(out_dir, "online_summary.json"), "w") as fh:
        json.dump({"summary": summary, "segments": segments, "loops": loops},
                  fh, indent=2)
    with open(os.path.join(out_dir, "online_history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    if args.save_q:
        np.save(args.save_q, agent.q)
        print(f"  Q-table saved        : {args.save_q}")

    print(f"\n  Wrote {out_dir}/online_summary.json and online_history.csv")


if __name__ == "__main__":
    main()
