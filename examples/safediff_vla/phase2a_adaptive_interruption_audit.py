#!/usr/bin/env python
"""PHASE 2A: OFFLINE counterfactual interruption-trigger audit. NO rollout, NO policy behavior
change, NO queue clear/replan -- reads the 18 already-completed E50 `GroundedGrasp-v2` episodes
from Phase 1A (`outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline/<task>/episodes_raw.json`)
and, purely in post-hoc analysis, computes what a set of candidate "interrupt the 50-step chunk and
replan now" triggers WOULD have fired on, step by step. Every signal used already exists in that
saved JSON (`monitor_target_world`, `planner_target_world`, `ee_pos`, `planner_replanned_this_step`)
-- nothing new is simulated.

Per-step signals computed:
    monitor_jump_m         -- ||monitor_target(t) - monitor_target(t-1)||
    planner_monitor_gap_m  -- ||monitor_target(t) - planner_target(t)|| (staleness)
    recent_var_m           -- RMS spread of monitor_target over the trailing K=10 steps
                               (sqrt(mean ||x_i - window_mean||^2))
    planner_age_steps      -- steps since the last actual replan
    ee_to_planner_m        -- ||ee_pos(t) - planner_target(t)||
    ee_to_monitor_m        -- ||ee_pos(t) - monitor_target(t)||

Candidate triggers (thresholds fixed from this data's own empirical percentiles below, NOT grid
searched -- a small number of round, interpretable values):
    A: monitor_jump_m > TAU_JUMP (5cm -- same round threshold Phase 1A/1B already used for
       "large" jitter, `monitor_jitter_frac_gt_5cm`)
    B: planner_monitor_gap_m > TAU_GAP (10cm -- near the E50 cohort's own mean gap of 8.8cm,
       rounded to an interpretable number)
    C: recent_var_m > TAU_VAR (set at this cohort's own 80th percentile -- "top-quintile
       instability", printed below, not hand-picked)
    D: A AND B simultaneously (the conjunction the Phase 2A spec asks for)

Usage:
    uv run python examples/safediff_vla/phase2a_adaptive_interruption_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

SRC_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase2a_interruption_audit")
TAU_JUMP = 0.05
TAU_GAP = 0.10
K_WINDOW = 10
SUCCESS_EPISODE = ("select_poker", 1001)


def vec(steps: list[dict], key: str) -> np.ndarray:
    return np.array([s[key] if s[key] is not None else [np.nan, np.nan, np.nan] for s in steps], dtype=float)


def compute_signals(ep: dict) -> dict:
    steps = ep["steps"]
    n = len(steps)
    monitor = vec(steps, "monitor_target_world")
    planner = vec(steps, "planner_target_world")
    ee = vec(steps, "ee_pos")
    replanned = np.array([bool(s["planner_replanned_this_step"]) for s in steps], dtype=bool)
    ee_to_gt = np.array([s["ee_to_target_dist"] for s in steps], dtype=float)
    valid_m = ~np.isnan(monitor).any(axis=1)
    valid_p = ~np.isnan(planner).any(axis=1)

    monitor_jump = np.full(n, np.nan)
    for t in range(1, n):
        if valid_m[t] and valid_m[t - 1]:
            monitor_jump[t] = float(np.linalg.norm(monitor[t] - monitor[t - 1]))

    gap = np.full(n, np.nan)
    both = valid_m & valid_p
    gap[both] = np.linalg.norm((monitor - planner)[both], axis=1)

    recent_var = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - K_WINDOW + 1)
        window = monitor[lo : t + 1]
        window = window[~np.isnan(window).any(axis=1)]
        if len(window) >= 2:
            mean = window.mean(axis=0)
            recent_var[t] = float(np.sqrt(np.mean(np.linalg.norm(window - mean, axis=1) ** 2)))

    planner_age = np.full(n, np.nan)
    last_replan = None
    for t in range(n):
        if replanned[t]:
            last_replan = t
        planner_age[t] = (t - last_replan) if last_replan is not None else np.nan

    ee_to_planner = np.full(n, np.nan)
    ee_to_planner[valid_p] = np.linalg.norm((ee - planner)[valid_p], axis=1)
    ee_to_monitor = np.full(n, np.nan)
    ee_to_monitor[valid_m] = np.linalg.norm((ee - monitor)[valid_m], axis=1)

    return {
        "monitor_jump_m": monitor_jump, "planner_monitor_gap_m": gap, "recent_var_m": recent_var,
        "planner_age_steps": planner_age, "ee_to_planner_m": ee_to_planner, "ee_to_monitor_m": ee_to_monitor,
        "ee_to_gt_m": ee_to_gt, "replanned": replanned,
    }


def main() -> None:
    all_episodes = []
    for task in STAGE2_TASK_NAMES:
        raw_path = SRC_DIR / task / "episodes_raw.json"
        if not raw_path.exists():
            print(f"[{task}] MISSING {raw_path}")
            continue
        for ep in json.loads(raw_path.read_text()):
            ep["task"] = task
            all_episodes.append(ep)

    # Establish TAU_VAR from this cohort's own 80th percentile, pooled across all steps/episodes.
    all_recent_var = []
    per_ep_signals = {}
    for ep in all_episodes:
        sig = compute_signals(ep)
        per_ep_signals[(ep["task"], ep["seed"])] = sig
        all_recent_var.extend(sig["recent_var_m"][~np.isnan(sig["recent_var_m"])].tolist())
    tau_var = float(np.percentile(all_recent_var, 80))

    print(f"TAU_JUMP={TAU_JUMP:.3f}m  TAU_GAP={TAU_GAP:.3f}m  TAU_VAR(80th pct of recent_var_m)={tau_var:.4f}m  K={K_WINDOW}")
    print(f"(for reference: recent_var_m distribution -- p50={np.percentile(all_recent_var, 50):.4f}, "
          f"p80={tau_var:.4f}, p95={np.percentile(all_recent_var, 95):.4f})\n")

    report_rows = []
    for (task, seed), sig in per_ep_signals.items():
        n = len(sig["ee_to_gt_m"])
        trig_A = sig["monitor_jump_m"] > TAU_JUMP
        trig_B = sig["planner_monitor_gap_m"] > TAU_GAP
        trig_C = sig["recent_var_m"] > tau_var
        trig_D = trig_A & trig_B

        def count(t):
            return int(np.nansum(t.astype(float)))

        # "trigger precedes failure": does ee_to_gt increase over the next W=20 steps after a
        # trigger, more often than from a random/non-trigger step in the same episode (a cheap,
        # interpretable proxy for "the policy was about to diverge right when this fired").
        W = 20
        ee_gt = sig["ee_to_gt_m"]

        def frac_diverging_after(trig: np.ndarray) -> float | None:
            idxs = np.where(trig)[0]
            idxs = idxs[idxs < n - W]
            if len(idxs) == 0:
                return None
            diverges = [ee_gt[i + W] > ee_gt[i] for i in idxs]
            return float(np.mean(diverges))

        def frac_diverging_baseline() -> float:
            idxs = np.arange(0, n - W)
            diverges = [ee_gt[i + W] > ee_gt[i] for i in idxs]
            return float(np.mean(diverges))

        report_rows.append(
            {
                "task": task, "seed": seed, "n_steps": n,
                "count_A_jump": count(trig_A), "count_B_gap": count(trig_B),
                "count_C_var": count(trig_C), "count_D_AandB": count(trig_D),
                "frac_diverging_after_D": frac_diverging_after(trig_D),
                "frac_diverging_baseline": frac_diverging_baseline(),
            }
        )

    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_episode_trigger_counts.json").write_text(json.dumps(report_rows, indent=2))

    print("per-episode trigger counts (A=jump>5cm, B=gap>10cm, C=var>p80, D=A&B):")
    for r in report_rows:
        tag = " <-- SUCCESS EPISODE" if (r["task"], r["seed"]) == SUCCESS_EPISODE else ""
        print(
            f"  {r['task']:14s} seed={r['seed']} n={r['n_steps']:4d}  A={r['count_A_jump']:3d} "
            f"B={r['count_B_gap']:3d} C={r['count_C_var']:3d} D={r['count_D_AandB']:3d}  "
            f"div_after_D={r['frac_diverging_after_D']}  div_baseline={r['frac_diverging_baseline']:.2f}{tag}"
        )

    mean_D = np.mean([r["count_D_AandB"] for r in report_rows])
    mean_D_rate = np.mean([r["count_D_AandB"] / r["n_steps"] for r in report_rows])
    print(f"\nmean trigger-D count per episode: {mean_D:.1f}  (mean rate: {mean_D_rate * 100:.1f}% of steps)")

    succ_row = next((r for r in report_rows if (r["task"], r["seed"]) == SUCCESS_EPISODE), None)
    if succ_row:
        print(
            f"\nSUCCESS episode ({SUCCESS_EPISODE[0]} seed={SUCCESS_EPISODE[1]}, n={succ_row['n_steps']}): "
            f"A={succ_row['count_A_jump']} B={succ_row['count_B_gap']} C={succ_row['count_C_var']} D={succ_row['count_D_AandB']} "
            f"-> rate D={succ_row['count_D_AandB'] / succ_row['n_steps'] * 100:.1f}% of its steps"
        )

    # stable-but-biased candidate episodes (from Phase 1B's own heuristic tag source data): flag
    # episodes whose min ee_to_gt never got close AND whose monitor jitter stayed low -- same
    # definition `summarize_grounded_grasp_v2_execute_horizon_ablation.py`'s `tag_pattern` used.
    print("\nstable-but-biased candidate episodes (min_ee_to_gt>8cm & low monitor jitter) -- trigger-D rate:")
    for (task, seed), sig in per_ep_signals.items():
        ee_gt = sig["ee_to_gt_m"]
        jump = sig["monitor_jump_m"]
        jump_valid = jump[~np.isnan(jump)]
        frac_gt_2cm = float((jump_valid > 0.02).mean()) if len(jump_valid) else 1.0
        min_dist = float(np.nanmin(ee_gt))
        if frac_gt_2cm < 0.3 and min_dist > 0.08:
            row = next(r for r in report_rows if r["task"] == task and r["seed"] == seed)
            rate = row["count_D_AandB"] / row["n_steps"] * 100
            print(f"  {task:14s} seed={seed}  min_ee_to_gt={min_dist:.3f}m  jitter_frac2cm={frac_gt_2cm:.2f}  trigger-D rate={rate:.1f}%")

    # Plot: trigger-D count vs episode outcome, and signal traces for the success episode + one
    # representative stable-biased episode (select_book seed1002, same as Phase 1B).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    success_mask = [
        any(ep["task"] == r["task"] and ep["seed"] == r["seed"] and ep["success"] for ep in all_episodes)
        for r in report_rows
    ]
    counts = [r["count_D_AandB"] for r in report_rows]
    colors = ["green" if s else "tab:red" for s in success_mask]
    ax.bar(range(len(counts)), counts, color=colors)
    ax.set_xlabel("episode index (all 18)")
    ax.set_ylabel("trigger-D count")
    ax.set_title("Trigger-D count per episode (green=success)")

    ax = axes[1]
    sig = per_ep_signals[SUCCESS_EPISODE]
    n = len(sig["ee_to_gt_m"])
    ax.plot(sig["planner_monitor_gap_m"], label="planner-monitor gap (B signal)", c="tab:orange")
    ax.plot(sig["monitor_jump_m"], label="monitor jump (A signal)", c="tab:purple", lw=0.7)
    trig_D = (sig["monitor_jump_m"] > TAU_JUMP) & (sig["planner_monitor_gap_m"] > TAU_GAP)
    for t in np.where(trig_D)[0]:
        ax.axvline(t, color="red", lw=0.6, alpha=0.4)
    ax.axhline(TAU_GAP, color="black", ls=":", lw=0.7)
    ax.set_xlabel("step")
    ax.set_title(f"SUCCESS episode signals ({SUCCESS_EPISODE[0]} seed={SUCCESS_EPISODE[1]}), red=trigger-D")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "trigger_overview.png", dpi=110)
    plt.close(fig)
    print(f"\nwrote {OUT_DIR / 'trigger_overview.png'} and {OUT_DIR / 'per_episode_trigger_counts.json'}")


if __name__ == "__main__":
    main()
