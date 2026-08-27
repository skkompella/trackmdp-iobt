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
}

FINDINGS = {}


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
