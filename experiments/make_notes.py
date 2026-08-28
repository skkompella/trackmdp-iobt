#!/usr/bin/env python3
"""
experiments/make_notes.py
Parse a /tmp/track_mdp_logs/runN.log file, extract training metrics, write
experiments/runN/notes.json, and regenerate experiments/index.md.

This is the objective-push-track-876cb3 worktree's own copy (runs 600+), kept
separate from experiments_ref (read-only reference log for runs 227-242) and
the sibling worktree's copy.

Usage:
    python experiments/make_notes.py --run 600
    python experiments/make_notes.py --run 600 --log /tmp/track_mdp_logs/run600.log
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Per-run metadata that can't be parsed from the log
# ---------------------------------------------------------------------------
HYPOTHESES = {
    600: "Reproduce run 236 baseline (soft-scale=0.8, camera fusion, from 227) in this worktree to confirm setup is correct before trying anything new.",
    601: "RESUME.md next-step #1: sweep --soft-scale below 0.8 (from run 227 base, camera fusion) since the "
         "1.2->0.8 trend was reported monotonically positive and not bottomed out. Tried soft-scale=0.5 first.",
    602: "RESUME.md next-step #2: multi-session training. Combine session 20250812_165739 (130 GPS-covered "
         "steps) with 20250812_091600 (1736 GPS-covered steps, camera P_cam available despite 3 missing "
         "audio FLACs and 3 malformed YOLO json files) into one concatenated training set (1866 steps total), "
         "same soft-scale=0.8/threshold=0.4 camera-fusion recipe as run 600/236 from the run 227 base. "
         "Required a code change to examples/finetune_deterministic.py (--iobt-extra-sessions flag) to build "
         "and concatenate P_fused/gt_seq across sessions for TRAINING while keeping EVAL on session "
         "20250812_165739 alone (via a separate eval-only RealIoBTEnv), so the reported accuracy number stays "
         "comparable to single-session runs 600/236 rather than being diluted across a much larger second "
         "session.",
    603: "RESUME.md next-step #3: --advanced-hparams slow-convergence PPO (lr=1e-4, train_batch_size=4000, "
         "num_sgd_iter=10, sgd_minibatch_size=128, clip_param=0.2, entropy_coeff=0.01, grad_clip=30, "
         "num_workers=4, rollout_fragment_length=1000) on the same run-227-base camera-fusion recipe as "
         "runs 600/236 (soft-scale=0.8, soft-threshold=0.4, session 20250812_165739, 200 iterations), to test "
         "whether a slower, more conservative PPO update schedule avoids the early-peak-then-decay pattern "
         "seen in every fast-convergence run so far and lets accuracy keep climbing past ~90%.",
    605: "RESUME.md next-step #5: use the run 241 base (500-iter topo prior, 99% raw eval accuracy) as the "
         "source for a full soft-scale sweep instead of just the single scale=1.2 pass done in run 242. This "
         "run tests scale=0.8 (the best scale found from the run 227 base in runs 600-604) from the stronger "
         "run 241 base, to see whether the better-pretrained base checkpoint raises the ceiling once combined "
         "with the best-known soft-reward config.",
    604: "RESUME.md next-step #4: combine --node-clfs-dir binary gating with --soft-reward camera fusion. "
         "Required a code change to examples/finetune_deterministic.py: a new build_binary_detection_matrix_10() "
         "function (using the same 20-feature amplitude+spectral pipeline as train_node_classifiers_10.py) plus "
         "a new --node-clfs-dir-10 flag, since the existing --node-clfs-dir/build_binary_detection_matrix() was "
         "hardcoded to the 6-node ReSpeaker CLAUDE.md pipeline (nodes 11-16, different feature set) and cannot "
         "be reused for the 10-node soft-reward system. Used the existing pre-trained 10-node per-node binary "
         "classifiers at results/v3_10node/node_clf_N{1..10}_multisession_v3_20260520_173424.pkl (trained "
         "in-session on 20250812_165739 — no other 10-node session has full binary-classifier training data). "
         "Gate is applied as P_fused[t,k] *= B[t,k] before GPS-mask alignment: zeroes soft-reward at (t,node) "
         "cells the binary classifier does not confirm, testing RESUME.md's hypothesis that this cuts false "
         "positives capping accuracy below 95%. Otherwise identical to run 600/236: source=227, camera fusion, "
         "soft-scale=0.8, soft-threshold=0.4, session 20250812_165739, 200 iterations, --gps-eval.",
    606: "RESUME.md next-step #6: re-run the cam-fallback-thresh sweep with an intermediate threshold (0.3) "
         "not tried in the original runs 232 (thresh=0.0) / 238 (thresh=0.65). Fusion mode cam_fallback: use "
         "camera P_cam when it exceeds cam-fallback-thresh, else fall back to audio P_audio. Otherwise same "
         "recipe as run 600/236: source=227, soft-scale=0.8, soft-threshold=0.4, session 20250812_165739, "
         "200 iterations, --gps-eval. IMPORTANT: this run required --calibrators results/pooled/"
         "pooled_calibrators_20260521_164149.pkl to be passed explicitly alongside --pooled-clf — a first "
         "launch attempt omitted it (RESUME.md's own example commands never show --calibrators, easy to miss) "
         "and produced a spurious baseline of 95.15% purely because the raw uncalibrated pooled audio "
         "classifier outputs mean=0.80/min=0.365 across all 10 nodes (i.e. it's not node-discriminative at "
         "all, everything scores 'high'), vs the properly calibrated mean=0.1985/max=0.449 that matches run "
         "232/238's historical P_fused stats. That first attempt was killed before completion and its "
         "checkpoint/log discarded — it was never a real result and is not being reported.",
    607: "RESUME.md next-step #7 (final ranked item): establish a --gt oracle upper bound that drives P_fused "
         "directly from GPS ground truth, bypassing the audio/camera classifiers entirely, to check whether "
         "95% is even achievable given the current sensor/classifier noise floor. Required a code change to "
         "examples/finetune_deterministic.py: a new fusion_mode='gt' branch in _build_session_fused() that "
         "builds P_fused as a perfect one-hot (P_fused[t, gt_seq[t]] = 1.0, zero elsewhere) directly from the "
         "GPS-derived ground-truth node sequence, skipping the pooled audio classifier and YOLO camera "
         "detector builds entirely. Otherwise identical recipe to run 600/236: source=227, soft-scale=0.8, "
         "soft-threshold=0.4, session 20250812_165739, 200 iterations, --gps-eval.",
}

FINDINGS = {
    600: "Reproduced within ~1pt of run 236: best 89.95% at iter 34 (vs run 236's 91.00% at iter 54/58) — "
         "run-to-run variance since neither run pins a training RNG seed. Confirms this worktree's setup, "
         "checkpoints, and pooled classifier are all correct. Accuracy peaks early (iter 30-40) then degrades "
         "over the remaining ~160 iterations, same oscillate-then-decay shape documented for run 236. "
         "Also fixed a real infra bug hit during this run: ray.init(_node_ip_address='127.0.0.1') from the "
         "previous iteration was a no-op — Ray's services.resolve_ip_for_localhost() special-cases the exact "
         "string '127.0.0.1' (and 'localhost'/'::1') and silently rewrites it back to the auto-detected "
         "external IP, which this sandbox blocks for self-connections, so GCS startup timed out every time. "
         "Fixed by pinning to '127.0.0.2' instead (still loopback on Linux, not special-cased by Ray).",
    601: "REJECTS the monotonic-trend hypothesis: soft-scale=0.5 best accuracy is only 83.35% (vs 89.95% for "
         "scale=0.8 in run 600, and 91.00% documented for run 236) — clearly worse, not better. Accuracy "
         "oscillates in the 79-83% band across all 200 iterations with no upward trend, unlike scale=0.8's "
         "peak-then-decay shape reaching ~90%. The 1.2->0.8 trend does NOT continue below 0.8 — 0.8 sits near "
         "a local optimum, not partway down a monotonic slope. Likely cause: at scale=0.5 the reward signal "
         "becomes too weak/flat for PPO's advantage estimation to distinguish good from bad actions, undoing "
         "the benefit that softening provided going from 1.2->0.8. Next-step #1 (soft-scale sweep below 0.8) "
         "is now considered explored and unpromising — the drop from 0.8->0.5 is steep enough that 0.6/0.7 are "
         "very unlikely to beat 0.8. Moving to next-steps #2+ in RESUME.md's ranked list rather than spending "
         "more runs narrowing this direction.",
    602: "Best accuracy 90.05% at iter 110 (4.93 nodes/step), baseline 88.65% — essentially FLAT vs run 600's "
         "single-session 89.95% (and still below run 236's documented 91.00%), well within run-to-run noise "
         "given the eval set is only 130 GPS-covered steps on session 20250812_165739. Adding ~14x more "
         "training data from a second session did NOT raise the accuracy ceiling on the primary session. "
         "Same oscillate-after-early-peak shape as every other camera-fusion run (best at iter 110, then "
         "decays/oscillates in the 85-90% band for the remaining 90 iterations) — the failure mode looks "
         "identical to the single-session case, so more training data from one additional, very different "
         "session isn't fixing whatever is capping accuracy around 90%. RESUME.md next-step #2 (multi-session "
         "training) is now considered explored and NOT promising enough to pursue further with more session "
         "combinations — the bottleneck is more likely in the reward/PPO dynamics (oscillate-then-decay after "
         "an early peak, same as every single-session run) than in training-set diversity or overfitting to "
         "one trajectory. Moving to next-steps #3+ (--advanced-hparams, node-clf gating, run-241 base sweep, "
         "cam-fallback-thresh, --gt oracle).",
    603: "Best accuracy 89.90% at iter 14 (5.20 nodes/step), baseline 89.20% — essentially FLAT vs run 600's "
         "89.95% and still below run 236's documented 91.00%. The slow-convergence hyperparameters do NOT "
         "avoid the early-peak-then-decay pattern; if anything the decay is worse and monotonic: after peaking "
         "at iter 14, accuracy drifts down through the 87-89% band (iters 15-100), then keeps sliding to the "
         "83-86% band (iters 100-160), and finishes in the 81-87% band (iters 160-200, final iter 200 = "
         "87.30%) while sensors/step steadily collapses from 5.2 down to ~3.2-3.7 — the lower entropy_coeff "
         "schedule interacting with kl_coeff appears to be driving the policy toward using fewer and fewer "
         "sensors over training, which look like a slow, steady collapse toward a degenerate low-sensor-usage "
         "policy rather than continued improvement. This run took ~90 min wall-clock (vs run 600's ~38 min) "
         "for the same 200 iterations due to the 8x larger train_batch_size (4000 vs default 512) — expensive "
         "for no benefit. RESUME.md next-step #3 (--advanced-hparams) is now considered explored and NOT "
         "promising: neither the fast-convergence defaults nor the slow-convergence advanced hparams beat the "
         "~90% ceiling from the run-227 base recipe. Moving to next-steps #4+ (node-clf binary gating combined "
         "with soft-reward camera fusion, run-241 base sweep, finer cam-fallback-thresh sweep, --gt oracle).",
    604: "Best accuracy 90.05% at iter 111 (5.59 nodes/step), baseline 89.20% — essentially FLAT vs run 600's "
         "89.95%/run 236's documented 91.00%, and NOT an improvement over the ~90% ceiling despite the gate "
         "zeroing 90.0% of all P_fused cells (binary classifier overall positive rate only 10.0% across the "
         "10 nodes x 132 steps, in-session on 20250812_165739). Same oscillate-after-early-peak-then-decay "
         "shape as every other camera-fusion run: after the iter-111 peak, accuracy drifts down into the "
         "82-86% band by iter 190-200 (final iter 200 = 84.85%), sensors/step drifting up from 5.3 to ~5.8 "
         "(more sensors activated for less reward, unlike run 603's collapse toward fewer sensors — opposite "
         "failure mode, same net effect of accuracy decay after the peak). The binary gate being this aggressive "
         "(90% of cells zeroed) yet leaving best-accuracy essentially unchanged suggests the RL policy is "
         "already learning to activate sensors close to the true GT node (where P_cam is naturally highest AND "
         "the in-session-overfit binary classifier is naturally most likely to fire), so gating out low-P_fused "
         "cells elsewhere doesn't meaningfully change which action the policy prefers — it mostly removes reward "
         "noise the policy was already learning to ignore. RESUME.md next-step #4 (node-clf binary gating) is "
         "now considered explored and NOT promising: it neither breaks through the ~90% ceiling nor materially "
         "changes the training dynamics (same peak-iter-~100-150 then decay pattern seen in runs 600/601/602). "
         "Caveat: only one gating classifier set exists (in-session on 165739, LOSO recall near-0 on other nodes "
         "per results/v3_10node/node_clf_v3_report.txt), so this is a best-case (overfit) gate, not a realistic "
         "cross-session one; a stricter/looser gate on more sessions was not tested given time budget. "
         "Moving to next-steps #5+ (run-241 base full soft-scale sweep, finer cam-fallback-thresh sweep, --gt "
         "oracle upper bound).",
    605: "Best accuracy 90.20% at iter 146 (4.32 nodes/step), baseline 89.10% — marginally above the ~90% "
         "ceiling seen in runs 600/602/604 (89.95-90.05%) but still below run 236's documented 91.00% and far "
         "below the 95% target, and within normal run-to-run noise given the 130-step eval set (~0.77pp "
         "granularity per step). Despite run 241's much stronger raw base accuracy (99% per RESUME.md), fine-"
         "tuning it with the same scale=0.8 camera-fusion recipe produces the same oscillate-after-early-peak "
         "training shape as every prior run from the weaker run 227 base: peaks at iter 146, i.e. even later "
         "than run 600's iter 30 peak, but still decays afterward (iter 198 dropped back to 88.65%). This "
         "confirms next-step #5's premise (241 base helps slightly) but the gain is marginal (+0.15-0.25pp over "
         "the 227-base ceiling), nowhere near closing the 4-5pp gap to 95%. Sensors/step at best (4.32) is "
         "notably lower than every 227-base run's best (4.9-5.6), suggesting the stronger base's better raw "
         "movement-prediction lets the policy use fewer, more targeted sensor activations for similar or "
         "slightly better accuracy — but this doesn't translate into materially higher accuracy. RESUME.md "
         "next-step #5 is now considered explored and NOT sufficient on its own to reach 95%: the bottleneck "
         "is not primarily base-checkpoint quality, reinforcing the pattern from runs 600-604 that something "
         "structural in the soft-reward PPO training dynamics (or the underlying camera/audio classifier noise "
         "floor) caps this recipe family around 90%. Next: RESUME.md next-step #6 (finer cam-fallback-thresh "
         "sweep) and #7 (--gt oracle upper bound) remain untried.",
    606: "Best accuracy 90.00% at iter 43 (5.26 nodes/step), baseline 89.20% — matches run 232's historical "
         "cam_fallback-thresh=0.0 result almost exactly (P_fused mean=0.349 vs 232's 0.349, baseline=89.20% "
         "vs 232's 89.20%, best 90.00% vs 232's 89.95%), confirming thresh=0.3 behaves essentially "
         "identically to thresh=0.0 for this session: P_cam is either exactly 0 or a high-confidence YOLO "
         "score (typically ≫0.3) with almost no probability mass in (0, 0.3], so raising the threshold from "
         "0.0 to 0.3 barely changes which cells fall back to audio (cam_coverage=0.263 in both cases). Same "
         "oscillate-after-early-peak-then-decay training shape as every other camera/cam_fallback run: peaks "
         "at iter 43, then decays (iter 135 dropped to 86.00%). RESUME.md next-step #6 (finer cam-fallback-"
         "thresh sweep) does NOT beat the ~90% ceiling and does not meaningfully differ from the already-"
         "tried thresh=0.0 case — the P_cam distribution's bimodal (zero-or-high) shape means intermediate "
         "thresholds between 0 and ~0.5 are unlikely to ever behave differently from thresh=0, so this "
         "direction is now considered fully explored. Also surfaced and fixed a real methodology bug: a "
         "first launch of this run omitted --calibrators (RESUME.md's example commands don't show this flag "
         "for the soft-reward path, easy to miss), which lets the pooled audio classifier's raw uncalibrated "
         "output (mean=0.80, min=0.365 across all 10 nodes at every timestep — i.e. NOT node-discriminative) "
         "leak into the reward's audio-fallback channel, producing a spurious 95.15% baseline / apparent "
         "95%+ accuracy purely from an artifact (the tracker's internal 'tracked' state persists far more "
         "often when almost every activated node scores >0.4 threshold regardless of correctness), not from "
         "any real detection improvement. That run was killed before completion, its checkpoint deleted, and "
         "its number (606) reused for the corrected, properly-calibrated run reported here. Any future run "
         "touching P_audio (fusion modes audio/or_max/weighted/cam_fallback) MUST pass --calibrators "
         "results/pooled/pooled_calibrators_20260521_164149.pkl — only pure camera-fusion mode is safe to "
         "run without it (P_cam doesn't go through the pooled audio classifier at all). Moving to RESUME.md "
         "next-step #7 (--gt oracle upper bound), the last untried ranked next-step.",
    607: "Best accuracy 99.00% at iter 191 (3.85 nodes/step), baseline 96.60% (before any fine-tuning at all, "
         "purely from the run-227 base checkpoint evaluated against a perfect one-hot reward) — DECISIVELY "
         "answers RESUME.md next-step #7's question: 95% IS achievable given the current environment/PPO/"
         "policy architecture, with a wide margin to spare (99.00%, 4pp above target), when the reward signal "
         "is clean. This is the highest accuracy of any run in this worktree's log by a large margin (vs the "
         "~90-91% ceiling for every one of runs 600-606, all of which used real audio/camera classifier "
         "signal). Critically, the TRAINING SHAPE is qualitatively different from every real-classifier run: "
         "instead of the oscillate-after-early-peak-then-decay pattern seen in 100% of runs 600-606, this run "
         "climbs steadily and near-monotonically for all 200 iterations (96.6% -> 97.0-97.5% by iter ~20-130 "
         "-> 98.3-98.4% by iter ~180-190 -> 99.0% at iter 191, then holds flat at 98.2-99.0% through iter 200 "
         "with NO decay), while sensors/step falls smoothly from 5.18 to ~3.85-3.92 (a cleaner, more efficient "
         "policy, not a collapsing one). This strongly confirms the hypothesis that has been building across "
         "runs 600-606: the ~90% ceiling on the real system is NOT caused by PPO training dynamics, RL "
         "hyperparameters, base-checkpoint quality, training-data volume, or reward-shaping choices (soft-"
         "scale/threshold/fusion-mode) — every one of those was independently ruled out by next-steps #1-6. "
         "It is caused by noise in the audio/camera classifier signal itself: real P_fused values are noisy, "
         "sometimes wrong, and sometimes near-uniform across nodes (as seen with the uncalibrated-audio bug in "
         "run 606), which both caps the achievable policy quality AND destabilizes PPO's advantage estimation "
         "enough to produce the oscillate-then-decay pattern absent here. CONCLUSION for the ranked next-steps "
         "list: all 7 items are now tried. None of the 6 concrete, classifier-driven directions (soft-scale "
         "sweep, multi-session training, advanced-hparams, node-clf gating, run-241 base, cam-fallback-thresh) "
         "broke the ~90-91% ceiling — best real-classifier result remains run 236's documented 91.00%. The "
         "--gt oracle (this run) measured 99.00%, confirming 95% is achievable in principle and the bottleneck "
         "is the audio/camera classifier noise floor on this session's sensor data, not the RL pipeline. "
         "Reaching 95% on the REAL system would require improving the underlying audio/camera classifiers "
         "themselves (e.g. better YOLO detections, less noisy audio features, more/better-labeled training "
         "sessions for the pooled+camera classifiers) rather than further RL/PPO tuning — a different "
         "workstream outside this task's ranked next-steps list.",
}


# ---------------------------------------------------------------------------
# Log parsers
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> dict:
    """Parse a finetune_deterministic.py log and return extracted fields."""
    text = log_path.read_text(errors="replace")
    fields = {}

    def bget(pattern, cast=str, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    fields["run"]            = bget(r"Output run\s*:\s*(\d+)", int)
    fields["source_run"]     = bget(r"Source run\s*:\s*(\d+)", int)
    fields["session"]        = bget(r"Session\s*:\s*(\S+)")
    fields["fusion_mode"]    = bget(r"Fusion mode\s*:\s*(\S+)", default="audio")
    fields["soft_scale"]     = bget(r"Soft scale\s*:\s*([0-9.]+)", float)
    fields["soft_threshold"] = bget(r"Soft threshold\s*:\s*([0-9.]+)", float)
    fields["iterations"]     = bget(r"Iterations\s*:\s*(\d+)", int)

    m = re.search(r"\[fused\]\s+P_fused:\s+mean=([0-9.]+)\s+max=([0-9.]+)", text)
    if m:
        fields["p_fused_mean"] = float(m.group(1))
        fields["p_fused_max"]  = float(m.group(2))
    else:
        fields["p_fused_mean"] = None
        fields["p_fused_max"]  = None

    m = re.search(r"\[cam_fallback\]\s+cam coverage:\s+([0-9.]+)\s+audio fill-in:\s+([0-9.]+)", text)
    if m:
        fields["cam_coverage"]    = float(m.group(1))
        fields["audio_fill_rate"] = float(m.group(2))

    m = re.search(r"Accuracy\s*:\s*([0-9.]+)\s+\(", text)
    if m:
        fields["baseline_acc"] = float(m.group(1))

    m = re.search(r"Baseline accuracy\s*:\s*([0-9.]+)", text)
    if m:
        fields["baseline_acc"] = float(m.group(1))

    m = re.search(r"Baseline sensors\s*:\s*([0-9.]+)", text)
    if m:
        fields["baseline_sensors"] = float(m.group(1))

    m = re.search(r"Best accuracy\s*:\s*([0-9.]+)", text)
    if m:
        fields["best_acc"] = float(m.group(1))

    m = re.search(r"Best sensors\s*:\s*([0-9.]+)", text)
    if m:
        fields["sensors_at_best"] = float(m.group(1))

    m = re.search(r"Improvement\s*:\s*([+-][0-9.]+)", text)
    if m:
        fields["improvement"] = float(m.group(1))

    star_matches = list(re.finditer(r"★ New best ([0-9.]+) — saved", text))
    if star_matches:
        last_star = star_matches[-1]
        preceding = text[max(0, last_star.start()-300):last_star.start()]
        iter_m = re.search(
            r"^\s*(\d+)\s+[+-]?(?:nan|[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.]+\s*$",
            preceding, re.MULTILINE
        )
        if iter_m:
            fields["best_iter"] = int(iter_m.group(1))
            if "sensors_at_best" not in fields:
                fields["sensors_at_best"] = float(iter_m.group(3))

    return fields


def write_notes(run: int, fields: dict, log_path: Path) -> Path:
    run_dir = PROJECT_ROOT / "experiments" / f"run{run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    notes = {
        "run":              run,
        "date":             datetime.now().strftime("%Y-%m-%d"),
        "source_run":       fields.get("source_run"),
        "checkpoint_path":  f"runs/agent_run{run}_ppo",
        "session":          fields.get("session"),
        "fusion_mode":      fields.get("fusion_mode", "audio"),
        "soft_scale":       fields.get("soft_scale"),
        "soft_threshold":   fields.get("soft_threshold"),
        "iterations":       fields.get("iterations"),
        "p_fused_mean":     fields.get("p_fused_mean"),
        "p_fused_max":      fields.get("p_fused_max"),
        "cam_coverage":     fields.get("cam_coverage"),
        "audio_fill_rate":  fields.get("audio_fill_rate"),
        "baseline_acc":      fields.get("baseline_acc"),
        "baseline_sensors":  fields.get("baseline_sensors"),
        "best_acc":          fields.get("best_acc"),
        "best_iter":         fields.get("best_iter"),
        "sensors_at_best":   fields.get("sensors_at_best"),
        "improvement":       fields.get("improvement"),
        "log_path":         str(log_path),
        "hypothesis":       HYPOTHESES.get(run, ""),
        "findings":         FINDINGS.get(run, ""),
    }

    notes_path = run_dir / "notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))
    print(f"  Wrote {notes_path}")
    return notes_path


def regenerate_index():
    exp_dir = PROJECT_ROOT / "experiments"
    all_notes = []
    for d in sorted(exp_dir.iterdir()):
        nf = d / "notes.json"
        if nf.exists():
            try:
                all_notes.append(json.loads(nf.read_text()))
            except Exception:
                pass

    all_notes.sort(key=lambda n: n.get("run", 0))

    lines = [
        "# Experiment Index — Track-MDP IoBT (objective-push-track-876cb3 worktree)\n",
        f"_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n",
        "Continues from the main checkout's `experiments_ref/` log (runs 227-242, best "
        "91.00% at run 236). This worktree's own runs start at 600 to avoid colliding "
        "with the main checkout (300, 500) or the sibling worktree (301, 302). "
        "Model checkpoints live in `runs/agent_runN_ppo/`. "
        "This file is auto-generated by `experiments/make_notes.py`.\n\n",
        "## Results Table\n\n",
        "| Run | Date | Source | Session | Fusion | Scale | Thresh | Iters | "
        "Base acc | Base nodes/step | Best acc | Best iter | Best nodes/step | Δ |\n",
        "|-----|------|--------|---------|--------|-------|--------|-------|"
        "---------|-----------------|----------|-----------|-----------------|---|\n",
    ]

    for n in all_notes:
        run          = n.get("run", "?")
        date         = n.get("date", "")[:10]
        src          = n.get("source_run", "—")
        session      = (n.get("session") or "—")[-6:] if n.get("session") else "—"
        fusion       = n.get("fusion_mode") or "—"
        scale        = n.get("soft_scale") or "—"
        thresh       = n.get("soft_threshold") or "—"
        iters        = n.get("iterations") or "—"
        base         = f"{n['baseline_acc']:.4f}" if n.get("baseline_acc") else "—"
        base_sensors = f"{n['baseline_sensors']:.2f}" if n.get("baseline_sensors") else "—"
        best         = f"**{n['best_acc']:.4f}**" if n.get("best_acc") else "—"
        b_iter       = n.get("best_iter") or "—"
        sensors      = f"{n['sensors_at_best']:.2f}" if n.get("sensors_at_best") else "—"
        delta        = f"{n['improvement']:+.4f}" if n.get("improvement") else "—"
        lines.append(
            f"| {run} | {date} | {src} | {session} | {fusion} | {scale} | "
            f"{thresh} | {iters} | {base} | {base_sensors} | {best} | {b_iter} | {sensors} | {delta} |\n"
        )

    lines.append("\n## Run Notes\n\n")
    for n in all_notes:
        run = n.get("run", "?")
        lines.append(f"### Run {run} — {n.get('fusion_mode', '?')} "
                     f"(src: {n.get('source_run', '?')})\n")
        lines.append(f"**Checkpoint:** `{n.get('checkpoint_path', '?')}`  \n")
        lines.append(f"**Session:** `{n.get('session', '?')}`  \n")
        hyp = n.get("hypothesis", "")
        if hyp:
            lines.append(f"**Hypothesis:** {hyp}  \n")
        findings = n.get("findings", "")
        if findings:
            lines.append(f"**Findings:** {findings}  \n")
        if n.get("best_acc"):
            base_s = (f"{n['baseline_sensors']:.2f}" if n.get("baseline_sensors")
                      else "—")
            best_s = (f"{n['sensors_at_best']:.2f}" if n.get("sensors_at_best")
                      else str(n.get("sensors_at_best", "?")))
            lines.append(
                f"**Result:** baseline={n.get('baseline_acc'):.4f} ({base_s} nodes/step)  "
                f"best={n.get('best_acc'):.4f} (iter {n.get('best_iter', '?')}, "
                f"{best_s} nodes/step)  \n"
            )
        lines.append("\n")

    index_path = exp_dir / "index.md"
    index_path.write_text("".join(lines))
    print(f"  Regenerated {index_path}")


def main():
    ap = argparse.ArgumentParser(description="Write per-run notes.json and regenerate index.md")
    ap.add_argument("--run", type=int, required=True, help="Run number")
    ap.add_argument("--log", type=str, default=None,
                    help="Path to log file (default: /tmp/track_mdp_logs/runN.log)")
    args = ap.parse_args()

    run      = args.run
    log_path = Path(args.log) if args.log else Path(f"/tmp/track_mdp_logs/run{run}.log")

    if not log_path.exists():
        print(f"  WARNING: log file not found: {log_path}")
        fields = {}
    else:
        print(f"  Parsing {log_path} ...")
        fields = parse_log(log_path)
        if fields.get("run") and fields["run"] != run:
            print(f"  WARNING: log says run={fields['run']} but --run={run}")
        fields["run"] = run  # authoritative

    write_notes(run, fields, log_path)
    regenerate_index()
    print(f"  Done — run {run} documented.")


if __name__ == "__main__":
    main()
