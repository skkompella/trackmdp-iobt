"""Tests for examples/replay_streams.py and examples/summarize_dal_validation.py."""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from replay_streams import glr_alarm_steps, replay  # noqa: E402
from src.core.change_detection import GLRChangeDetector  # noqa: E402
from summarize_dal_validation import detection_metrics, summarize_runs  # noqa: E402


def _iid(n_streams=10, n=5_000, p=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return {f"s{k}": (rng.random(n) < p).astype(np.int8) for k in range(n_streams)}


def _markov(n_streams=10, n=5_000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n_streams):
        x = np.empty(n, dtype=np.int8)
        x[0] = 1
        u = rng.random(n)
        for t in range(1, n):
            x[t] = u[t] < (0.9 if x[t - 1] else 0.1)
        out[f"s{k}"] = x
    return out


def test_replay_iid_ordered_matches_shuffled():
    s = _iid()
    ordered = replay(s, 0.5, None)
    shuffled = np.mean([replay(s, 0.5, seed) for seed in range(3)])
    assert abs(ordered - shuffled) <= 3


def test_replay_detects_clustering():
    s = _markov()
    ordered = replay(s, 0.5, None)
    shuffled = np.mean([replay(s, 0.5, seed) for seed in range(3)])
    assert ordered > 5 * (shuffled + 1)


def test_detection_metrics():
    sw = [{"episode": 1000}, {"episode": 2000}]
    al = [{"episode": 1050}, {"episode": 1500}, {"episode": 2400}]
    m = detection_metrics(al, sw, 300)
    assert m["caught"] == 1 and m["n_switches"] == 2
    assert m["false_alarms"] == 2 and m["delays"] == [50]


def test_summarize_runs_table(tmp_path):
    run = tmp_path / "grid_transition_s1"
    run.mkdir()
    (run / "online_summary.json").write_text(json.dumps({"summary": {
        "env": "grid", "detector": "transition", "detector_mode": "monitor",
        "probe_rate": 0.2, "train_seed": 1, "greedy_mean_accuracy": 0.9,
        "greedy_final_accuracy": 0.95, "greedy_mean_sensors": 4.1,
        "probe_fraction": 0.05, "n_restarts": 0}, "segments": []}))
    (run / "alarms.json").write_text(json.dumps({
        "alarms": [{"episode": 1050}], "switches": [{"episode": 1000}]}))
    rows, md = summarize_runs([str(run)], 300)
    assert rows[0]["caught"] == 1 and rows[0]["false_alarms"] == 0
    assert "grid_transition_s1" in md and "1/1" in md


def test_fast_glr_matches_reference_detector():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.random(3000) < 0.2, rng.random(3000) < 0.7,
                        rng.random(3000) < 0.1]).astype(float)
    ref = GLRChangeDetector(delta=1e-3, min_samples=30, max_window=500)
    want = [t for t, v in enumerate(x) if ref.update(v)]
    assert len(want) >= 2
    assert glr_alarm_steps(x, 1e-3, min_samples=30, max_window=500) == want


def test_summary_reports_training_sensors_including_probes(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "online_summary.json").write_text(json.dumps({"summary": {
        "env": "grid", "detector": "transition", "mean_sensors": 4.25,
        "greedy_mean_sensors": 4.0}, "segments": []}))
    (run / "alarms.json").write_text(json.dumps({"alarms": [], "switches": []}))
    rows, md = summarize_runs([str(run)], 300)
    assert rows[0]["train_sensors"] == 4.25
    assert "train sensors/step" in md and "4.25" in md


def test_summary_warns_on_run_without_alarms(tmp_path, capsys):
    run = tmp_path / "r"
    run.mkdir()
    (run / "online_summary.json").write_text(json.dumps({"summary": {}, "segments": []}))
    rows, _ = summarize_runs([str(run)], 300)
    assert rows == [] and "missing alarms.json" in capsys.readouterr().err
