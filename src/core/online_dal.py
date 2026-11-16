"""
online_dal.py — online tabular SARSA on a switching-matrix schedule, with the
transition-stream change detector, coverage probes, and resumable checkpoints.

Detector arms (``--detector``):
  none        never restart (continual-learning baseline)
  oracle      restart exactly at the known switches
  transition  per-(cell, next cell) GLR streams fed from tau=0 steps, with
              coverage probes (src/core/transition_detector.py, probing.py)

``--detector-mode monitor`` logs alarms (and clears the detector) without
touching the learner: that measures the detector alone.  ``restart`` restarts
the learner (cold or warm) at the end of the episode in which an alarm fired.

Runs checkpoint every ``--checkpoint-every`` episodes and when paused
(SIGTERM / SIGINT / SIGUSR1, or a PAUSE file in the out dir); ``--resume``
continues bit-for-bit.  See experiments/transition_dal/README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .change_detection import RestartController
from .matrix_iobt_env import MatrixIoBTEnv
from .online_schedule import adaptation_latency, detector_scores, greedy_eval, run_episode
from .probing import CoverageProber
from .random_transition_matrices import parse_matrix_spec
from .run_checkpoint import PauseRequest, has_checkpoint, load_checkpoint, save_checkpoint
from .switching_grid_env import SwitchingTransitionGridEnv
from .tabular_td import TabularTDAgent
from .transition_detector import TransitionStreamDetector, row_supports

IOBT_BACKING_N = 4          # MatrixIoBTEnv observations live on a 4x4 grid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # environment
    p.add_argument("--env", choices=["grid", "iobt"], default="grid")
    p.add_argument("--nrows", type=int, default=5)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=5)
    p.add_argument("--sensor-rew", type=float, default=None)
    p.add_argument("--reward-preset", type=str, default="grid",
                   help="IoBT reward preset (grid env ignores it)")
    p.add_argument("--matrices", type=str, default="R0,R1,R2,R3,R4")
    p.add_argument("--matrix-seed", type=int, default=20261006)
    p.add_argument("--concentration", type=float, default=1.0)
    p.add_argument("--action-mode", choices=["factored", "local"], default="local",
                   help="grid only; IoBT uses the full subset table")
    # schedule
    p.add_argument("--episodes-per-segment", type=int, default=1000)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--max-ep-steps", type=int, default=100)
    # learner
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--optimistic-init", type=float, default=5.0)
    # detector
    p.add_argument("--detector", choices=["none", "transition", "oracle"],
                   default="transition")
    p.add_argument("--detector-mode", choices=["restart", "monitor"],
                   default="restart")
    p.add_argument("--delta-F", type=float, default=0.01,
                   help="False-alarm budget for the whole detector")
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--max-window", type=int, default=2000)
    p.add_argument("--probe-rate", type=float, default=0.2,
                   help="rho: every stream keeps >= floor(rho * visits) samples")
    p.add_argument("--restart-strategy", choices=["cold", "warm"], default="cold")
    p.add_argument("--prior-path", type=str, default=None)
    # evaluation
    p.add_argument("--greedy-eval-every", type=int, default=100)
    p.add_argument("--greedy-eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=99)
    # bookkeeping
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--save-streams", action="store_true",
                   help="Write every stream's samples to <out-dir>/streams.npz")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-episodes", type=int, default=None,
                   help="Checkpoint and exit (as if paused) after N episodes")
    return p


class DetectionHook:
    """run_episode hook: probes, the detector, and (optionally) stream recording."""

    def __init__(self, supports, detector, prober, record_streams):
        self.supports = supports
        self.detector = detector
        self.prober = prober
        self.streams = ({(i, j): [] for i, s in supports.items() for j in s}
                        if record_streams else None)

    def probe(self, cell):
        if self.prober is not None and self.prober.due(cell):
            return self.prober.probe_cells(cell)
        return None

    def observe(self, cell, watched, found, probed):
        if self.prober is not None:
            self.prober.record(cell, watched)
        if self.streams is not None and (found is None or found in self.supports[cell]):
            for j in set(watched).intersection(self.supports[cell]):
                self.streams[(cell, j)].append(int(j == found))
        if self.detector is not None:
            self.detector.observe(cell, watched, found)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _build(args):
    """Fresh run state: envs, learner, controller, detector, prober, hook."""
    tl = args.time_limit
    if args.env == "grid":
        names, mats = parse_matrix_spec(args.matrices, "grid", args.matrix_seed,
                                        args.concentration, args.nrows, args.ncols)
        n_cells = args.nrows * args.ncols
        cfg = {"time_limit": tl, "missing_state": n_cells * (tl + 1) + 1,
               "max_ep_steps": args.max_ep_steps, "max_sensors": args.max_sensors}

        def make_env(seed):
            return SwitchingTransitionGridEnv(
                args.nrows, args.ncols, args.max_sensors, args.max_sensors,
                cfg["missing_state"], tl, mats, seed=seed,
                no_moore_constraint=True, sensor_rew=args.sensor_rew)
        agent = TabularTDAgent(
            time_limit=tl, max_sensors=args.max_sensors,
            missing_state=cfg["missing_state"], alpha=args.alpha,
            gamma=args.gamma, lam=args.lam, epsilon=args.epsilon,
            optimistic_init=args.optimistic_init, seed=args.train_seed,
            n_nodes=n_cells, grid_width=n_cells, action_mode=args.action_mode,
            grid_shape=(args.nrows, args.ncols))
        supports = row_supports("grid", args.nrows, args.ncols)
    else:
        names, mats = parse_matrix_spec(args.matrices, "iobt", args.matrix_seed,
                                        args.concentration)
        cfg = {"time_limit": tl,
               "missing_state": IOBT_BACKING_N * IOBT_BACKING_N * (tl + 1) + 1,
               "max_ep_steps": args.max_ep_steps, "max_sensors": args.max_sensors}

        def make_env(seed):
            return MatrixIoBTEnv(
                args.max_sensors, args.max_sensors, cfg["missing_state"], tl,
                mats, seed=seed, no_moore_constraint=True,
                reward_preset=args.reward_preset, sensor_rew=args.sensor_rew)
        agent = TabularTDAgent(
            time_limit=tl, max_sensors=args.max_sensors,
            missing_state=cfg["missing_state"], alpha=args.alpha,
            gamma=args.gamma, lam=args.lam, epsilon=args.epsilon,
            optimistic_init=args.optimistic_init, seed=args.train_seed)
        supports = row_supports("iobt")

    prior = np.load(args.prior_path) if args.prior_path else None
    controller = RestartController(agent, strategy=args.restart_strategy,
                                   prior=prior)
    detector = prober = hook = None
    if args.detector == "transition":
        detector = TransitionStreamDetector(
            supports, delta_F=args.delta_F, min_samples=args.min_samples,
            max_window=args.max_window)
        prober = CoverageProber(supports, args.max_sensors, args.probe_rate)
        hook = DetectionHook(supports, detector, prober, args.save_streams)

    return {
        "names": names, "cfg": cfg, "agent": agent, "controller": controller,
        "env": make_env(args.train_seed),
        "eval_env": make_env(args.eval_seed) if args.greedy_eval_every > 0 else None,
        "detector": detector, "prober": prober, "hook": hook,
        "history": [], "segments": [], "alarms": [], "switches": [],
        "episode": 0, "step": 0, "n_restarts": 0, "n_probes": 0,
        "seg": 0, "seg_ep": 0, "acc": {},
    }


def _new_accumulators():
    return {"acc": [], "sens": [], "g_acc": [], "g_sens": [], "g_hist": []}


def _segment_record(st, args, pass_idx, m_name):
    a = st["acc"]
    rec = {"pass": pass_idx, "matrix": m_name,
           "start_episode": st["episode"] - args.episodes_per_segment,
           "mean_accuracy": float(np.mean(a["acc"])),
           "final_accuracy": float(np.mean(a["acc"][-100:])),
           "mean_sensors": float(np.mean(a["sens"])),
           "adaptation_latency": adaptation_latency(a["acc"], 0.05)}
    if a["g_acc"]:
        rec.update({
            "greedy_mean_accuracy": float(np.mean(a["g_acc"])),
            "greedy_final_accuracy": a["g_acc"][-1],
            "greedy_mean_sensors": float(np.mean(a["g_sens"])),
            "greedy_curve": a["g_acc"], "greedy_sensor_curve": a["g_sens"],
            "greedy_active_hist": a["g_hist"],
            "greedy_cells_per_tracked_step": [
                float(np.dot(h, np.arange(len(h))) / max(1, sum(h)))
                for h in a["g_hist"]]})
    return rec


def _summary(st, args):
    segs = st["segments"]
    summary = {
        "learner": "tabular", "env": args.env, "matrices": st["names"],
        "detector": args.detector, "detector_mode": args.detector_mode,
        "delta_F": args.delta_F, "probe_rate": args.probe_rate,
        "restart_strategy": args.restart_strategy, "train_seed": args.train_seed,
        "max_sensors": args.max_sensors, "sensor_rew": st["env"].sensor_rew,
        "episodes_per_segment": args.episodes_per_segment, "passes": args.passes,
        "env_steps": st["step"], "n_alarms": len(st["alarms"]),
        "n_restarts": st["n_restarts"], "n_probes": st["n_probes"],
        "probe_fraction": st["n_probes"] / max(1, st["step"]),
        "mean_accuracy": float(np.mean([s["mean_accuracy"] for s in segs])),
        "final_accuracy": float(np.mean([s["final_accuracy"] for s in segs])),
        "mean_sensors": float(np.mean([s["mean_sensors"] for s in segs])),
        "ceiling": (args.max_ep_steps - 1) / args.max_ep_steps,
        "detection": detector_scores(
            [a["episode"] for a in st["alarms"]],
            [s["episode"] for s in st["switches"]],
            tolerance=min(300, args.episodes_per_segment)),
    }
    if st["detector"] is not None:
        summary["n_streams"] = st["detector"].n_streams
        summary["n_off_support"] = st["detector"].n_off_support
    if segs and "greedy_mean_accuracy" in segs[0]:
        summary.update({
            "greedy_mean_accuracy": float(np.mean([s["greedy_mean_accuracy"] for s in segs])),
            "greedy_final_accuracy": float(np.mean([s["greedy_final_accuracy"] for s in segs])),
            "greedy_mean_sensors": float(np.mean([s["greedy_mean_sensors"] for s in segs])),
            "greedy_end_cells_per_tracked_step": float(np.mean(
                [s["greedy_cells_per_tracked_step"][-1] for s in segs]))})
    return summary


def _finish(st, args, out):
    summary = _summary(st, args)
    with open(os.path.join(out, "online_history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(st["history"][0].keys()))
        w.writeheader()
        w.writerows(st["history"])
    _write_json(os.path.join(out, "alarms.json"),
                {"alarms": st["alarms"], "switches": st["switches"]})
    hook = st["hook"]
    if hook is not None and hook.streams is not None:
        np.savez_compressed(os.path.join(out, "streams.npz"), **{
            f"{i}_{j}": np.asarray(v, dtype=np.int8)
            for (i, j), v in hook.streams.items()})
    # online_summary.json is the "done" marker (resume and the scheduler skip a
    # run that has it), so it is written only after every other output.
    _write_json(os.path.join(out, "online_summary.json"),
                {"summary": summary, "segments": st["segments"]})
    _write_json(os.path.join(out, "status.json"),
                {"state": "done", "episode": st["episode"]})
    return summary


def _after_episode(st, args, res):
    """Account one training episode; handle an alarm at its end."""
    ep = st["episode"]
    st["step"] += len(res["detections"])
    st["n_probes"] += res["n_probes"]
    st["acc"]["acc"].append(res["accuracy"])
    st["acc"]["sens"].append(res["sensors"])
    det = st["detector"]
    if det is not None and det.alarmed:
        st["alarms"].append(dict(det.last_alarm, episode=ep, step=st["step"]))
        if args.detector_mode == "restart":
            st["controller"].restart(segment_reward=float(np.mean(st["acc"]["acc"])))
            st["n_restarts"] += 1
        det.reset()
        st["prober"].reset()
    st["episode"] = ep + 1


def run_online(args) -> dict:
    out = os.path.abspath(args.out_dir)
    ckpt_dir = os.path.join(out, "ckpt")
    summary_path = os.path.join(out, "online_summary.json")
    os.makedirs(out, exist_ok=True)
    if args.restart_strategy == "warm" and not args.prior_path:
        raise SystemExit("--restart-strategy warm requires --prior-path")

    args_dict = {k: v for k, v in vars(args).items() if k != "out_dir"}
    if args.resume:
        if os.path.exists(summary_path):
            print(f"  {out}: already done")
            with open(summary_path) as fh:
                return json.load(fh)["summary"]
        st = load_checkpoint(ckpt_dir, args_dict)
    else:
        if has_checkpoint(ckpt_dir) or os.path.exists(summary_path):
            raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
        st = _build(args)
        st["acc"] = _new_accumulators()

    names = st["names"]
    schedule = [(p, m) for p in range(args.passes) for m in range(len(names))]
    pause = PauseRequest(os.path.join(out, "PAUSE"))
    pause.clear()
    try:
        while st["seg"] < len(schedule):
            pass_idx, m_idx = schedule[st["seg"]]
            st["env"].force_matrix(m_idx)
            if st["seg_ep"] == 0 and st["seg"] > 0:
                prev = schedule[st["seg"] - 1][1]
                if prev != m_idx:
                    st["switches"].append({"episode": st["episode"], "step": st["step"],
                                           "from": names[prev], "to": names[m_idx]})
                    if args.detector == "oracle":
                        st["controller"].restart()
                        st["n_restarts"] += 1

            res = run_episode(st["agent"], st["env"], st["cfg"], learn=True,
                              explore=True, hook=st["hook"])
            st["history"].append({
                "episode": st["episode"], "pass": pass_idx, "matrix": names[m_idx],
                "accuracy": res["accuracy"], "sensors": res["sensors"],
                "reward": res["reward"], "n_probes": res["n_probes"]})
            _after_episode(st, args, res)
            st["seg_ep"] += 1

            seg_done = st["seg_ep"] == args.episodes_per_segment
            if st["eval_env"] is not None and (
                    st["seg_ep"] % args.greedy_eval_every == 0 or seg_done):
                st["eval_env"].force_matrix(m_idx)
                ga, gs, gh = greedy_eval(st["agent"], st["eval_env"], st["cfg"],
                                         args.greedy_eval_episodes)
                st["acc"]["g_acc"].append(ga)
                st["acc"]["g_sens"].append(gs)
                st["acc"]["g_hist"].append(gh)
            if seg_done:
                st["segments"].append(_segment_record(st, args, pass_idx, names[m_idx]))
                a = st["segments"][-1]
                print(f"  pass {pass_idx} {names[m_idx]:>4}  acc {a['mean_accuracy']*100:6.2f}%"
                      f"  alarms {len(st['alarms'])}  restarts {st['n_restarts']}", flush=True)
                st["acc"] = _new_accumulators()
                st["seg"] += 1
                st["seg_ep"] = 0

            ep = st["episode"]
            stop = args.stop_after_episodes is not None and ep >= args.stop_after_episodes
            paused = pause.requested() or stop
            finished = st["seg"] >= len(schedule)
            # Always checkpoint the final state too, so a crash while writing
            # the outputs can be resumed straight into _finish.
            if finished or paused or ep % args.checkpoint_every == 0:
                save_checkpoint(ckpt_dir, st, args_dict)
            if paused and not finished:
                _write_json(os.path.join(out, "status.json"),
                            {"state": "paused", "episode": ep})
                print(f"  paused at episode {ep}; continue with --resume", flush=True)
                return _summary(st, args) if st["segments"] else {}
            _write_json(os.path.join(out, "status.json"),
                        {"state": "running", "episode": ep})
        return _finish(st, args, out)
    finally:
        pause.uninstall()


__all__ = ["build_parser", "run_online", "DetectionHook"]
