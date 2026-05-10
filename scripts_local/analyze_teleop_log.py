#!/usr/bin/env python3
"""Analyze SARM teleop JSONL log: plot the LATEST episode only, overlay GT.

JSONL is append-only across runs/episodes. Episode boundary detected via
step-counter reset (step[i] < step[i-1]). Default behavior: keep only the
LAST episode that contains GT stage entries; fall back to the longest one.

Usage:
    uv run python scripts_local/analyze_teleop_log.py outputs/teleop_logs/sarm_teleop.jsonl
    # Or pick a specific episode by 0-based index:
    uv run python scripts_local/analyze_teleop_log.py log.jsonl --episode 1
    # Or plot all episodes side-by-side:
    uv run python scripts_local/analyze_teleop_log.py log.jsonl --all
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def split_episodes(rows):
    """Split rows on step-counter resets. Returns list of episode rows."""
    eps = []
    cur = []
    for r in rows:
        if cur and r["step"] < cur[-1]["step"]:
            eps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        eps.append(cur)
    return eps


def select_episode(eps, mode: str, idx: int | None):
    """Pick which episode to plot."""
    if idx is not None:
        if idx < 0:
            idx += len(eps)
        return [eps[idx]] if 0 <= idx < len(eps) else []
    if mode == "all":
        return eps
    # Default: prefer the LAST episode that has any GT stage entries; else longest.
    def has_gt(e):
        return any(r.get("gt_stage_idx") not in (None, 0) for r in e)
    with_gt = [e for e in eps if has_gt(e)]
    if with_gt:
        return [with_gt[-1]]
    return [max(eps, key=len)] if eps else []


def plot_one(ep, ax_progress, ax_stage, ax_conf, title_prefix=""):
    steps = [r["step"] for r in ep]
    progress = [r["progress"] for r in ep]
    stage_idx = [r["stage_idx"] for r in ep]
    stage_conf = [r["stage_conf"] for r in ep]
    gt_stage_idx = [r.get("gt_stage_idx") for r in ep]
    # Detect actual stage transitions (first occurrence of each gt_stage_idx)
    # — gt_stage_started_this_frame may stick across frames due to upstream
    # info-dict reuse, so we dedupe by (first row where gt_stage_idx changes).
    gt_starts = []
    last_gt = None
    for r in ep:
        gi = r.get("gt_stage_idx")
        if gi is not None and gi != last_gt:
            gt_starts.append((r["step"], gi, r.get("gt_stage_name")))
            last_gt = gi

    max_so_far = np.maximum.accumulate(stage_idx)
    regress_mask = np.array(stage_idx) < max_so_far
    n_regress = int(regress_mask.sum())

    ax_progress.plot(steps, progress, "-", color="C0", lw=1.1)
    ax_progress.set_ylabel("progress")
    ax_progress.set_ylim(-0.05, 1.05)
    ax_progress.set_title(f"{title_prefix}SARM trace  steps={steps[0]}..{steps[-1]}  regressions={n_regress}")
    ax_progress.grid(alpha=0.3)

    ax_stage.plot(steps, stage_idx, "-", color="C1", lw=1.4, label="SARM pred")
    if any(g is not None for g in gt_stage_idx):
        gx = [s for s, g in zip(steps, gt_stage_idx) if g is not None]
        gy = [g for g in gt_stage_idx if g is not None]
        ax_stage.step(gx, gy, where="post", color="green", lw=2.2, label="GT (user)")
    ax_stage.plot(steps, max_so_far, "--", color="gray", lw=0.7, label="max_so_far")
    ax_stage.fill_between(steps, 0, 6, where=regress_mask, color="red", alpha=0.10, label="regressions")
    for gs, gi, gn in gt_starts:
        ax_stage.axvline(gs, color="green", lw=0.6, alpha=0.4)
    ax_stage.set_ylabel("stage_idx")
    ax_stage.set_yticks(range(7))
    ax_stage.set_ylim(-0.5, 6.5)
    ax_stage.legend(loc="upper left", fontsize=8)
    ax_stage.grid(alpha=0.3)

    ax_conf.plot(steps, stage_conf, "-", color="C2", lw=1)
    ax_conf.set_ylabel("stage_conf")
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.set_xlabel("env step")
    ax_conf.grid(alpha=0.3)
    return gt_starts, stage_idx, steps, n_regress


def summarize(ep, gt_starts, stage_idx, steps):
    print(f"records         : {len(ep)}")
    print(f"steps           : {steps[0]}..{steps[-1]}")
    print(f"delta_indices   : {ep[0].get('delta_indices')}")
    print(f"SARM stage dist : {dict(Counter(r['stage_idx'] for r in ep))}")
    if any(r.get('gt_stage_idx') is not None for r in ep):
        print(f"GT stage dist   : {dict(Counter(r.get('gt_stage_idx') for r in ep))}")
    print(f"unique progress : {len(set(r['progress'] for r in ep))}")
    if gt_starts:
        print(f"\nGT entries → SARM catch-up:")
        for gs, gi, gn in gt_starts:
            i_gs = next((i for i, s in enumerate(steps) if s >= gs), None)
            if i_gs is None:
                continue
            j = next((i for i in range(i_gs, len(steps)) if stage_idx[i] >= gi), None)
            if j is not None:
                print(f"  gt stage {gi} ({gn}): gt_step={gs} → SARM step={steps[j]}  lag={steps[j]-gs}")
            else:
                print(f"  gt stage {gi} ({gn}): gt_step={gs} → SARM never caught up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--episode", type=int, default=None,
                    help="0-based ep index; negative counts from end. Default: last ep with GT.")
    ap.add_argument("--all", action="store_true", help="Plot every episode in stacked subplots.")
    args = ap.parse_args()

    p = Path(args.path)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        print("empty log"); return
    eps = split_episodes(rows)
    print(f"episodes in log : {len(eps)} (lengths={[len(e) for e in eps]})")
    selected = select_episode(eps, mode="all" if args.all else "default", idx=args.episode)
    if not selected:
        print("no episode selected"); return

    n = len(selected)
    fig, axes = plt.subplots(3 * n, 1, figsize=(11, 4 + 2.5 * n), sharex=False)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    else:
        axes = np.array(axes).reshape(3 * n, 1)

    for i, ep in enumerate(selected):
        prefix = f"ep{i}: " if n > 1 else ""
        ax_p = axes[3 * i, 0]
        ax_s = axes[3 * i + 1, 0]
        ax_c = axes[3 * i + 2, 0]
        gt, sidx, steps, nreg = plot_one(ep, ax_p, ax_s, ax_c, title_prefix=prefix)
        if n == 1 or i == n - 1:
            print()
        if n == 1:
            summarize(ep, gt, sidx, steps)
        else:
            print(f"\n--- ep{i} ({len(ep)} samples) ---")
            summarize(ep, gt, sidx, steps)

    plt.tight_layout()
    out = Path(args.out) if args.out else p.with_suffix(".png")
    plt.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
