"""
run_checkpoint.py — atomic checkpoints and pause requests for long online runs.

A run pickles its whole state (learner, envs, detector, histories, loop
position) plus both global RNG states, so a resumed run continues bit-for-bit.
Writes go to a temporary file that is renamed into place, and LATEST is
updated the same way, so a kill mid-save leaves the previous checkpoint intact.

PauseRequest turns SIGTERM / SIGINT / SIGUSR1 or the presence of a PAUSE file
into a flag the run checks at episode boundaries: it then checkpoints and
exits cleanly, and ``--resume`` picks it up later.
"""
from __future__ import annotations

import os
import pickle
import random
import signal

import numpy as np

DEFAULT_IGNORE = ("resume", "stop_after_episodes", "checkpoint_every")
_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_checkpoint(ckpt_dir: str, state: dict, args_dict: dict) -> str:
    """Write state + args + RNG states; keep only this step. Returns the path."""
    episode = int(state["episode"])
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"step_{episode:07d}.pkl"
    path = os.path.join(ckpt_dir, name)
    payload = {"state": state, "args": dict(args_dict),
               "rng": {"np_global": np.random.get_state(),
                       "py": random.getstate()}}
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _write_atomic(os.path.join(ckpt_dir, "LATEST"), name.encode())
    for f in os.listdir(ckpt_dir):
        if f.startswith("step_") and f != name:
            os.remove(os.path.join(ckpt_dir, f))
    return path


def has_checkpoint(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, "LATEST"))


def load_checkpoint(ckpt_dir: str, args_dict: dict,
                    ignore=DEFAULT_IGNORE) -> dict:
    """Load the latest checkpoint, refuse on changed args, restore RNGs."""
    if not has_checkpoint(ckpt_dir):
        raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
    with open(os.path.join(ckpt_dir, "LATEST")) as fh:
        name = fh.read().strip()
    with open(os.path.join(ckpt_dir, name), "rb") as fh:
        payload = pickle.load(fh)
    saved = payload["args"]
    keys = (set(saved) | set(args_dict)) - set(ignore)
    diff = {k: (saved.get(k), args_dict.get(k)) for k in sorted(keys)
            if saved.get(k) != args_dict.get(k)}
    if diff:
        raise SystemExit(f"--resume: arguments differ from the original run: {diff}")
    np.random.set_state(payload["rng"]["np_global"])
    random.setstate(payload["rng"]["py"])
    return payload["state"]


class PauseRequest:
    """Flag set by SIGTERM / SIGINT / SIGUSR1 or by a PAUSE file existing."""

    def __init__(self, pause_file: str):
        self.pause_file = pause_file
        self._flag = False
        self._previous = {}
        for sig in _SIGNALS:
            self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum, frame):
        self._flag = True

    def requested(self) -> bool:
        return self._flag or os.path.exists(self.pause_file)

    def clear(self) -> None:
        self._flag = False
        if os.path.exists(self.pause_file):
            os.remove(self.pause_file)

    def uninstall(self) -> None:
        """Restore the signal handlers that were active before."""
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}


__all__ = ["save_checkpoint", "load_checkpoint", "has_checkpoint",
           "PauseRequest"]
