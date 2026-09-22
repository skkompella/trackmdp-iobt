"""
online_schedule.py — rollout and metrics shared by the online experiments.

Extracted from examples/train_multiloop_online.py so the IoBT loop experiment
and the switching-grid experiment measure themselves the same way.  Nothing
here is specific to either environment: it needs only ``reset_object_state()``,
``object_move()``, ``get_reward_next_state()`` and ``object_pos``, which both
MultiLoopIoBTEnv and SwitchingTransitionGridEnv provide.
"""
from __future__ import annotations

import numpy as np


def run_episode(agent, env, cfg, learn=True, explore=True):
    """
    One episode on whichever regime the env is currently pinned to.

    Mirrors the state machine in evaluate_policy() so training and evaluation
    agree: episodes start in the missing state, and that first step counts in
    the denominator but can never count as a detection — which is why the
    ceiling is (max_ep_steps-1)/max_ep_steps rather than 1.0.
    """
    time_limit    = cfg["time_limit"]
    missing_state = cfg["missing_state"]
    max_sensors   = cfg["max_sensors"]
    # Any value >= agent.n_nodes selects the agent's missing-state row.
    missing_pos   = cfg.get("missing_obs_pos", agent.n_nodes)

    env.reset_object_state()
    current_state = missing_state
    time_delay    = 0

    found = steps = sensors_on = 0
    total_reward = 0.0
    detections: list[int] = []

    prev = None   # (state, action, reward) pending SARSA update

    for _ in range(cfg["max_ep_steps"]):
        was_missing = (current_state == missing_state)
        if was_missing:
            state_pos, state_time = missing_pos, 0
        else:
            state_pos  = current_state // (time_limit + 1)
            state_time = current_state % (time_limit + 1)

        s      = agent.obs_to_state((state_pos, state_time, None))
        action = agent.select_action(s, explore=explore)
        action_sensors = agent.action_vector(action)

        # Accuracy accounting, matching evaluate_policy: a hit only counts
        # while the tracker is actually tracking.
        hit = int(not was_missing and action_sensors[env.object_pos] == 1)
        found += hit
        steps += 1
        detections.append(hit)

        reward, next_state, _terminal, time_delay = env.get_reward_next_state(
            current_state, action_sensors, time_delay)
        total_reward += reward

        # Charge what evaluate_policy charges, or these numbers would not be
        # comparable with PPO's.  It applies its max_sensors clip AFTER the
        # missing-state branch, so a full rescan is billed at max_sensors even
        # though the env internally scans everything to guarantee re-acquisition.
        sensors_on += max_sensors if was_missing else int(action_sensors.sum())

        if learn:
            if prev is not None:
                agent.observe(prev[0], prev[1], prev[2], s, action, done=False)
            prev = (s, action, reward)

        current_state = next_state

    if learn and prev is not None:
        agent.observe(prev[0], prev[1], prev[2], prev[0], prev[1], done=True)
        agent.end_episode()

    return {
        "accuracy":   found / max(steps, 1),
        "sensors":    sensors_on / max(steps, 1),
        "reward":     total_reward,
        "detections": detections,
    }


def adaptation_latency(accuracies, epsilon, tail_frac=0.3, window=3):
    """
    Episodes until accuracy recovers to within ``epsilon`` of steady state.

    Steady state is the mean over the last ``tail_frac`` of the segment.

    LIMITATION, stated plainly: because steady state is the segment's own tail,
    this measures how fast the learner reached *its own* plateau, not how good
    that plateau is.  A segment that degrades and stays bad scores latency 0.
    Always read it next to mean accuracy; latency alone can flatter a learner
    that gave up quickly.  Returns None only when the segment is too short to
    fit the averaging window.
    """
    acc = np.asarray(accuracies, dtype=float)
    n = len(acc)
    if n < window + 1:
        return None
    tail = max(1, int(n * tail_frac))
    steady = float(acc[-tail:].mean())
    for i in range(n - window + 1):
        if acc[i:i + window].mean() >= steady - epsilon:
            return i
    return None


def detector_scores(fired_episodes, switch_episodes, tolerance):
    """Precision/recall of detections against the known switch points."""
    fired  = list(fired_episodes)
    truth  = list(switch_episodes)
    matched_truth, matched_fire = set(), set()
    for ti, t in enumerate(truth):
        for fi, f in enumerate(fired):
            if fi in matched_fire:
                continue
            if 0 <= f - t <= tolerance:
                matched_truth.add(ti)
                matched_fire.add(fi)
                break
    tp = len(matched_truth)
    return {
        "n_detections":  len(fired),
        "n_switches":    len(truth),
        "true_positive": tp,
        "false_alarms":  len(fired) - len(matched_fire),
        "precision":     tp / len(fired) if fired else None,
        "recall":        tp / len(truth) if truth else None,
        "tolerance_episodes": tolerance,
    }


__all__ = ["run_episode", "adaptation_latency", "detector_scores"]
