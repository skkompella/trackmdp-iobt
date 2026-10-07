#!/usr/bin/env python
"""
calibration_pipeline.py — per-environment DAL detector tuning (online learner, k=5).

Runs after the k=5 cost sweep (cost_sweep_scheduler.sh). For each of the 5x5 grid,
IoBT and 10x10 grid random-matrix environments:

  1. pick the sensor cost with the best online + DAL greedy accuracy at k=5
  2. record detector-signal streams with the online learner (no detector):
       switching   R0..R4 x 3 passes x 1000 episodes (the real schedule)
       stationary  R0 for the same length -- every alarm on it is false
  3. sweep delta x signal {hit, reward, transition_surprise} offline at a target
     false-alarm rate of ARL0 >= 100,000 steps (at most ~1 per 1000-episode segment)
  4. re-run online + DAL warm with the tuned signal/delta (runs 1300-1302) and
     compare with the untuned run at the same cost

Everything is the same design as results.md; only the detector changes.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PY = "./track_mdp_env/bin/python"
OUT = "experiments/random_matrix/calibration"
PRIORS = "experiments/random_matrix/priors"
LOGS = "/tmp/track_mdp_logs"
K = 5
TARGET_ARL0 = 100_000
HP = ["--alpha", "0.05", "--gamma", "0.9", "--lam", "0.0", "--epsilon", "0.1",
      "--optimistic-init", "5.0"]
MATRICES = "R0,R1,R2,R3,R4"
SIGNALS = "hit,reward,transition_surprise"
ENV = {
    # env: (results dir, existing online+DAL k=5 runs incl. the sweep, new run)
    "grid":   ("grid",   [1026, 1027] + list(range(1200, 1206)), 1300),
    "iobt":   ("iobt",   [1030, 1031] + list(range(1210, 1216)), 1301),
    "grid10": ("grid10", [1114, 1115] + list(range(1220, 1226)), 1302),
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd, logfile):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with open(logfile, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        raise RuntimeError(f"rc={rc}: {' '.join(cmd)} (log {logfile})")


def best_cost(env):
    d, runs, _ = ENV[env]
    rows = []
    for r in runs:
        f = f"experiments/random_matrix/{d}/run{r}/online_summary.json"
        if os.path.exists(f):
            s = json.load(open(f))["summary"]
            rows.append((s["greedy_mean_accuracy"], s["sensor_rew"], r))
    rows.sort(reverse=True)
    log(f"{env}: online+DAL k=5 by cost: " +
        ", ".join(f"{c:+.2f}->{a*100:.2f}% (run {r})" for a, c, r in rows))
    acc, cost, run_id = rows[0]
    return cost, acc, run_id


def geometry(env):
    if env == "grid":
        return ["--env", "grid", "--action-mode", "local", "--matrices", MATRICES]
    if env == "grid10":
        return ["--env", "grid", "--nrows", "10", "--ncols", "10",
                "--action-mode", "local", "--matrices", MATRICES]
    return ["--env", "iobt", "--iobt-matrices", MATRICES]


def online_cmd(env):
    if env == "iobt":
        return [PY, "examples/train_multiloop_online.py", "--matrices", MATRICES,
                "--episodes-per-loop", "1000"]
    g = ["--nrows", "10", "--ncols", "10"] if env == "grid10" else []
    return [PY, "examples/train_switching_grid_online.py", *g, "--action-mode",
            "local", "--matrices", MATRICES, "--episodes-per-segment", "1000"]


def pipeline(env):
    os.makedirs(OUT, exist_ok=True)
    cost, acc, untuned = best_cost(env)
    c = f"{abs(cost):.2f}"
    common = [*geometry(env), "--max-sensors", str(K), "--sensor-rew", str(cost),
              "--episodes-per-segment", "1000", "--passes", "3", *HP[:2],
              "--gamma", "0.9", "--lam", "0.0", "--epsilon", "0.1",
              "--optimistic-init", "5.0"]
    streams = {}
    for sched in ("switching", "stationary"):
        out = f"{OUT}/{env}_k5_c{c}_{sched}.npz"
        if not os.path.exists(out):
            log(f"{env}: recording {sched} stream (cost {cost})")
            run([PY, "examples/record_detection_streams.py", *common,
                 "--condition", "learning", "--schedule", sched, "--out", out],
                f"{LOGS}/calib_rec_{env}_{sched}.log")
        streams[sched] = out

    sweep = f"{OUT}/{env}_k5_c{c}_sweep.json"
    if not os.path.exists(sweep):
        log(f"{env}: offline sweep")
        run([PY, "examples/calibrate_detector.py",
             "--switching", streams["switching"], "--stationary", streams["stationary"],
             "--out", sweep, "--signals", SIGNALS, "--n-deltas", "14",
             "--delta-lo", "1e-26", "--delta-hi", "0.4", "--max-window", "200",
             "--stationary-limit", "1500000", "--target-arl0", str(TARGET_ARL0)],
            f"{LOGS}/calib_sweep_{env}.log")
    summ = json.load(open(sweep))["summary"]
    ok = [s for s in summ if s.get("detectable") and s.get("met_target_arl0")]
    if ok:
        ok.sort(key=lambda s: (s["n_switches"] - s["n_caught"], s["mean_delay"]))
        pick = ok[0]
    else:
        cand = [s for s in summ if s.get("detectable")]
        pick = max(cand, key=lambda s: s["arl0"])
        log(f"{env}: no signal reached ARL0 {TARGET_ARL0}; using highest ARL0")
    log(f"{env}: tuned -> {pick['signal']} delta {pick['delta']:.2e} "
        f"(ARL0 {pick['arl0']:.0f}, caught {pick['n_caught']}/{pick['n_switches']}, "
        f"delay {pick['mean_delay']:.0f})")

    d, _, new_run = ENV[env]
    out_dir = f"experiments/random_matrix/{d}/run{new_run}"
    if not os.path.exists(f"{out_dir}/online_summary.json"):
        log(f"{env}: validation run {new_run}")
        tag = f"tab_{env}_C_k{K}_c{c}"
        run([*online_cmd(env), "--passes", "3", "--max-sensors", str(K),
             "--sensor-rew", str(cost), "--restart-strategy", "warm",
             "--prior-path", f"{PRIORS}/{tag}.npy", "--detector-signal", pick["signal"],
             "--delta", f"{pick['delta']:.6e}", "--max-window", "200", *HP,
             "--greedy-eval-every", "100", "--greedy-eval-episodes", "10",
             "--new-run", str(new_run), "--out-dir", out_dir],
            f"{LOGS}/calib_run{new_run}.log")
    tuned = json.load(open(f"{out_dir}/online_summary.json"))["summary"]
    base = json.load(open(f"experiments/random_matrix/{d}/run{untuned}/online_summary.json"))["summary"]
    res = {"env": env, "k": K, "cost": cost, "pick": pick,
           "untuned_run": untuned, "tuned_run": new_run,
           "untuned": {"acc": base["greedy_mean_accuracy"], "restarts": base["n_restarts"],
                       "detector": base.get("detector")},
           "tuned": {"acc": tuned["greedy_mean_accuracy"], "restarts": tuned["n_restarts"],
                     "detector": tuned.get("detector"),
                     "cells": tuned.get("greedy_end_cells_per_tracked_step")}}
    log(f"{env}: untuned {res['untuned']['acc']*100:.2f}% ({res['untuned']['restarts']} "
        f"restarts) -> tuned {res['tuned']['acc']*100:.2f}% ({res['tuned']['restarts']} restarts)")
    return res


def main():
    status = "experiments/random_matrix/cost_sweep_scheduler.status"
    while not (os.path.exists(status) and "ALL COST SWEEP DONE" in open(status).read()):
        time.sleep(120)
    log("cost sweep done; calibrating")
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(pipeline, ["grid", "iobt", "grid10"]))
    json.dump(results, open(f"{OUT}/calibration_results.json", "w"), indent=1)
    log("ALL CALIBRATION DONE")


if __name__ == "__main__":
    sys.exit(main())
