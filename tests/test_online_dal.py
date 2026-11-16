"""
Tests for src/core/online_dal.py — the resumable online runner with
transition-stream detection, coverage probes and oracle restarts.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.core.online_dal import build_parser, run_online  # noqa: E402

PY = os.path.join(ROOT, "track_mdp_env", "bin", "python")
TINY = ["--env", "grid", "--nrows", "3", "--ncols", "3",
        "--matrices", "R0,R1", "--passes", "1",
        "--episodes-per-segment", "6", "--max-ep-steps", "20",
        "--max-sensors", "3", "--action-mode", "local",
        "--greedy-eval-every", "3", "--greedy-eval-episodes", "2"]


def _run(out, *extra):
    return run_online(build_parser().parse_args(
        TINY + ["--out-dir", str(out)] + list(extra)))


def _read(out, name):
    with open(os.path.join(out, name)) as fh:
        return fh.read()


def _status(out):
    return json.loads(_read(out, "status.json"))


def test_smoke_grid(tmp_path):
    summary = _run(tmp_path)
    for f in ("online_summary.json", "online_history.csv", "alarms.json"):
        assert os.path.exists(tmp_path / f)
    assert _status(tmp_path)["state"] == "done"
    alarms = json.loads(_read(tmp_path, "alarms.json"))
    assert len(alarms["switches"]) == 1
    assert alarms["switches"][0]["episode"] == 6
    assert summary["env_steps"] == 12 * 20


def test_smoke_iobt(tmp_path):
    args = ["--env", "iobt", "--matrices", "R0,R1", "--passes", "1",
            "--episodes-per-segment", "4", "--max-ep-steps", "20",
            "--max-sensors", "3", "--out-dir", str(tmp_path)]
    run_online(build_parser().parse_args(args))
    assert _status(tmp_path)["state"] == "done"


def test_save_streams(tmp_path):
    import numpy as np
    _run(tmp_path, "--save-streams")
    z = np.load(tmp_path / "streams.npz")
    assert "4_4" in z.files and z["4_4"].dtype == np.int8


def test_resume_is_bit_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(a)
    _run(b, "--stop-after-episodes", "5")
    assert _status(b)["state"] == "paused"
    _run(b, "--resume")
    assert _read(a, "online_history.csv") == _read(b, "online_history.csv")
    assert _read(a, "alarms.json") == _read(b, "alarms.json")


def test_fresh_run_refuses_existing_checkpoint(tmp_path):
    _run(tmp_path, "--stop-after-episodes", "2")
    with pytest.raises(SystemExit, match="--resume"):
        _run(tmp_path)


def test_resume_refuses_changed_args(tmp_path):
    _run(tmp_path, "--stop-after-episodes", "2")
    with pytest.raises(SystemExit, match="arguments differ"):
        _run(tmp_path, "--resume", "--probe-rate", "0.5")


def test_resume_of_finished_run_is_noop(tmp_path):
    first = _run(tmp_path)
    history = _read(tmp_path, "online_history.csv")
    again = _run(tmp_path, "--resume")
    assert again == first
    assert _read(tmp_path, "online_history.csv") == history


def test_identical_matrices_are_not_switches(tmp_path):
    out = tmp_path
    run_online(build_parser().parse_args(
        [a if a != "R0,R1" else "R0" for a in TINY]
        + ["--passes", "3", "--out-dir", str(out)]))
    assert json.loads(_read(out, "alarms.json"))["switches"] == []


def test_oracle_restarts_once_per_switch(tmp_path):
    summary = _run(tmp_path, "--detector", "oracle")
    assert summary["n_restarts"] == 1 and summary["n_alarms"] == 0


def test_alarms_at_most_one_per_episode_and_restart_each(tmp_path):
    summary = _run(tmp_path, "--episodes-per-segment", "30",
                   "--delta-F", "0.5", "--min-samples", "5")
    alarms = json.loads(_read(tmp_path, "alarms.json"))["alarms"]
    assert len(alarms) > 0
    assert len({a["episode"] for a in alarms}) == len(alarms)
    assert summary["n_restarts"] == len(alarms) == summary["n_alarms"]


def test_monitor_mode_never_restarts(tmp_path):
    summary = _run(tmp_path, "--episodes-per-segment", "30",
                   "--delta-F", "0.5", "--min-samples", "5",
                   "--detector-mode", "monitor")
    assert summary["n_alarms"] > 0 and summary["n_restarts"] == 0


def test_sigterm_pauses_and_resumes(tmp_path):
    straight, paused = tmp_path / "s", tmp_path / "p"
    longer = [a if a != "6" else "60" for a in TINY]
    run_online(build_parser().parse_args(longer + ["--out-dir", str(straight)]))
    cmd = [PY, os.path.join(ROOT, "examples", "train_online_dal.py")] + longer \
        + ["--out-dir", str(paused)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if _status(paused)["episode"] >= 3:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 0, proc.stderr.read().decode()
    st = _status(paused)
    assert st["state"] == "paused" and st["episode"] < 120
    subprocess.run(cmd + ["--resume"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    assert _read(straight, "online_history.csv") == \
        _read(paused, "online_history.csv")


def test_crash_while_finishing_is_not_reported_done(tmp_path, monkeypatch):
    import numpy as np
    straight, crashed = tmp_path / "s", tmp_path / "c"
    _run(straight, "--save-streams")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError):
        _run(crashed, "--save-streams")
    monkeypatch.undo()
    assert not os.path.exists(crashed / "online_summary.json")
    _run(crashed, "--save-streams", "--resume", "--checkpoint-every", "200")
    assert _read(straight, "online_history.csv") == _read(crashed, "online_history.csv")
    assert _read(straight, "alarms.json") == _read(crashed, "alarms.json")
