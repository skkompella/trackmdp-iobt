"""Tests for src/core/run_checkpoint.py — atomic run checkpoints and pausing."""
import os
import pickle
import random
import signal
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.run_checkpoint import (  # noqa: E402
    PauseRequest,
    has_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_roundtrip_and_rng_restore(tmp_path):
    np.random.seed(3)
    random.seed(3)
    save_checkpoint(str(tmp_path), {"episode": 5, "x": [1, 2]}, {"a": 1})
    want = (np.random.rand(), random.random())
    st = load_checkpoint(str(tmp_path), {"a": 1})
    assert st["x"] == [1, 2] and st["episode"] == 5
    assert (np.random.rand(), random.random()) == want


def test_has_checkpoint(tmp_path):
    assert not has_checkpoint(str(tmp_path / "none"))
    save_checkpoint(str(tmp_path), {"episode": 1}, {})
    assert has_checkpoint(str(tmp_path))


def test_only_latest_step_kept(tmp_path):
    save_checkpoint(str(tmp_path), {"episode": 1}, {})
    save_checkpoint(str(tmp_path), {"episode": 2}, {})
    steps = [f for f in os.listdir(tmp_path) if f.startswith("step_")]
    assert steps == ["step_0000002.pkl"]
    assert load_checkpoint(str(tmp_path), {})["episode"] == 2


def test_mismatched_args_refuse(tmp_path):
    save_checkpoint(str(tmp_path), {"episode": 1}, {"a": 1, "b": 2})
    with pytest.raises(SystemExit, match="arguments differ.*'a'"):
        load_checkpoint(str(tmp_path), {"a": 9, "b": 2})


def test_ignored_args_allowed(tmp_path):
    save_checkpoint(str(tmp_path), {"episode": 1},
                    {"a": 1, "resume": False, "stop_after_episodes": 5})
    st = load_checkpoint(str(tmp_path),
                         {"a": 1, "resume": True, "stop_after_episodes": None})
    assert st["episode"] == 1


def test_episode_required(tmp_path):
    with pytest.raises(KeyError):
        save_checkpoint(str(tmp_path), {"x": 1}, {})


def test_crash_during_save_keeps_previous(tmp_path, monkeypatch):
    save_checkpoint(str(tmp_path), {"episode": 1, "tag": "first"}, {})

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(pickle, "dump", boom)
    with pytest.raises(RuntimeError):
        save_checkpoint(str(tmp_path), {"episode": 2, "tag": "second"}, {})
    monkeypatch.undo()
    assert load_checkpoint(str(tmp_path), {})["tag"] == "first"


def test_pause_file_and_signal(tmp_path):
    pr = PauseRequest(str(tmp_path / "PAUSE"))
    try:
        assert not pr.requested()
        (tmp_path / "PAUSE").touch()
        assert pr.requested()
        pr.clear()
        assert not pr.requested() and not (tmp_path / "PAUSE").exists()
        os.kill(os.getpid(), signal.SIGUSR1)
        assert pr.requested()
    finally:
        pr.uninstall()
