#!/usr/bin/env python3
"""Analyze SARM teleop JSONL log: plot stage trajectory + flag regressions.

Usage:
    uv run python scripts_local/analyze_teleop_log.py outputs/teleop_logs/sarm_teleop.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="JSONL log path")
    ap.add_argument("--out", type=str, default=None, help="output PNG (default: <path>.png)")
    args = ap.parse_args()

    p = Path(args.path)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        print("empty log"); return

    steps = [r["step"] for r in rows]
    progress = [r["progress"] for r in rows]
    stage_idx = [r["stage_idx"] for r in rows]
    stage_conf = [r["stage_conf"] for r in rows]
    stage_name = [r.get("stage_name", "?") for r in rows]
    bufs = [r.get("buffer_len", 0) for r in rows]
    deltas = rows[0].get("delta_indices")
    gt_stage_idx = [r.get("gt_stage_idx") for r in rows]
    gt_starts = [(r["step"], r["gt_stage_idx"], r.get("gt_stage_name"))
                 for r in rows if r.get("gt_stage_started_this_frame") is not None]

    # Detect regressions: stage_idx[i] < max(stage_idx[:i])
    max_so_far = np.maximum.accumulate(stage_idx)
    regress_mask = np.array(stage_idx) < max_so_far
    regression_steps = [int(steps[i]) for i in np.where(regress_mask)[0]]
    print(f"records       : {len(rows)}")
    print(f"steps         : {steps[0]}..{steps[-1]}")
    print(f"delta_indices : {deltas}")
    print(f"unique stages : {dict(Counter(stage_name))}")
    print(f"unique progress values: {len(set(progress))} ({sorted(set(progress))[:8]}{'...' if len(set(progress))>8 else ''})")
    print(f"regressions   : {len(regression_steps)} (stage drops back)")
    if gt_starts:
        print(f"gt stage entries (step, idx, name):")
        for s, gi, gn in gt_starts:
            print(f"  step={s} → stage {gi} ({gn})")
    # Compute lag per gt entry
    if gt_starts and any(g is not None for g in gt_stage_idx):
        for gs, gi, gn in gt_starts:
            i_gs = next((i for i, s in enumerate(steps) if s >= gs), None)
            if i_gs is None: continue
            j = next((i for i in range(i_gs, len(steps)) if stage_idx[i] >= gi), None)
            if j is not None:
                print(f"  gt stage {gi} ({gn}) entered step={gs}; SARM caught up at step={steps[j]} (lag={steps[j]-gs})")
    if regression_steps:
        runs = []
        cur = [regression_steps[0]]
        for s in regression_steps[1:]:
            if s - cur[-1] <= 4:
                cur.append(s)
            else:
                runs.append((cur[0], cur[-1]))
                cur = [s]
        runs.append((cur[0], cur[-1]))
        print(f"regression runs (start..end steps):")
        for s, e in runs[:20]:
            print(f"  {s} .. {e}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(steps, progress, "-", color="C0", lw=1)
    axes[0].set_ylabel("progress")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"SARM teleop trace  (regressions={len(regression_steps)})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, stage_idx, "-", color="C1", lw=1.2, label="SARM pred")
    if any(g is not None for g in gt_stage_idx):
        gt_x = [s for s, g in zip(steps, gt_stage_idx) if g is not None]
        gt_y = [g for g in gt_stage_idx if g is not None]
        axes[1].step(gt_x, gt_y, where="post", color="green", lw=2, label="GT (user)")
    axes[1].plot(steps, max_so_far, "--", color="gray", lw=0.8, label="max_so_far")
    axes[1].fill_between(steps, 0, 6, where=regress_mask, color="red", alpha=0.15, label="regressions")
    for gs, gi, gn in gt_starts:
        axes[1].axvline(gs, color="green", lw=0.7, alpha=0.5)
    axes[1].set_ylabel("stage_idx")
    axes[1].set_yticks(range(7))
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, stage_conf, "-", color="C2", lw=1)
    axes[2].set_ylabel("stage_conf")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("env step")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(args.out) if args.out else p.with_suffix(".png")
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
