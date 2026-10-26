#!/usr/bin/env python
"""Phase 2C analysis: compares condition A (unmodified E50, reused from Phase 1A) against
condition B (phase-preserving XY chunk reanchoring) on the 4 representative episodes, using the
SAME metric set Phase 1A/1B already established (`analyze_episode`, `extra_metrics` imported, not
duplicated) plus reanchor-specific metrics (event count, applied correction magnitude, chunk offset
at correction, gap before/after).

Usage:
    uv run python examples/safediff_vla/summarize_grounded_grasp_v2_phase_preserving_reanchor.py
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
from summarize_grounded_grasp_v2_execute_horizon_ablation import extra_metrics  # noqa: E402
from summarize_grounded_grasp_v2_monitor_baseline import _vec_arr, analyze_episode  # noqa: E402

# NOTE: condition A is re-run FRESH through this SAME wrapper script (enabled=False), not reused
# from Phase 1A's separately-invoked process. Two independent GPU process runs of the identical
# checkpoint are NOT bit-reproducible by default (no torch.use_deterministic_algorithms), and over
# a 500-step chaotic closed-loop rollout that can compound into a qualitatively different
# trajectory even with zero logical code difference (confirmed: select_fruit seed1001 showed a
# completely different reactive-close TIMING against the Phase-1A-reused baseline despite
# n_reanchor_events=0) -- re-running A through this same script keeps the per-step compute profile
# (including the extra monitor-only forward) identical between A and B, isolating the one real
# variable: whether a correction is ever applied to the queue.
COND_A_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_preserving_reanchor/condition_A_fresh")
COND_B_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_preserving_reanchor/condition_B")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_preserving_reanchor")
REPRESENTATIVE_EPISODES = [
    ("select_poker", 1001), ("select_fruit", 1001), ("select_book", 1002), ("select_mahjong", 1000),
]


def load_episode(src_dir: Path, task: str, seed: int) -> dict | None:
    raw_path = src_dir / task / "episodes_raw.json"
    if not raw_path.exists():
        return None
    for ep in json.loads(raw_path.read_text()):
        if ep["seed"] == seed:
            ep["task"] = task
            return ep
    return None


def reanchor_metrics(ep: dict) -> dict:
    steps = ep["steps"]
    events = [s for s in steps if s.get("reanchor_event")]
    return {
        "n_reanchor_events": len(events),
        "applied_correction_m": [s["applied_correction_m"] for s in events],
        "chunk_offset_at_correction": [s["chunk_offset"] for s in events],
        "gap_before_m": [s["gap_before_m"] for s in events],
        "gap_after_m": [s["gap_after_m"] for s in events],
    }


def full_metrics(ep: dict) -> dict:
    m = analyze_episode(ep)
    m.update(extra_metrics(ep))
    m["terminal_distance_m"] = m["last_ee_to_gt_dist_m"]
    m.update(reanchor_metrics(ep))
    return m


def plot_comparison(task: str, seed: int, ep_a: dict, ep_b: dict, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{task} seed={seed} -- condition A (original E50) vs B (phase-preserving reanchor)")

    for col, (label, ep, color) in enumerate([("A (original)", ep_a, "tab:blue"), ("B (reanchor)", ep_b, "tab:green")]):
        steps = ep["steps"]
        ee_xyz = _vec_arr(steps, "ee_pos")
        target_xyz = _vec_arr(steps, "target_pos")
        monitor_xyz = _vec_arr(steps, "monitor_target_world")
        planner_xyz = _vec_arr(steps, "planner_target_world")
        step_ix = np.arange(len(steps))
        valid_m = ~np.isnan(monitor_xyz).any(axis=1)
        valid_p = ~np.isnan(planner_xyz).any(axis=1)
        reanchor_idxs = [i for i, s in enumerate(steps) if s.get("reanchor_event")]

        ax = axes[0, col]
        ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], c=color, lw=1, label="EE actual")
        ax.scatter(target_xyz[0, 0], target_xyz[0, 1], c="green" if color != "green" else "darkgreen",
                   marker="*", s=200, label="GT target", zorder=5)
        if valid_m.any():
            ax.scatter(monitor_xyz[valid_m, 0], monitor_xyz[valid_m, 1], c=step_ix[valid_m], cmap="autumn", s=6, label="monitor target")
        if valid_p.any():
            ax.scatter(planner_xyz[valid_p, 0], planner_xyz[valid_p, 1], c="black", marker="x", s=25, label="planner anchor")
        for i in reanchor_idxs:
            ax.scatter(ee_xyz[i, 0], ee_xyz[i, 1], c="red", marker="D", s=60, zorder=6,
                       label="reanchor event" if i == reanchor_idxs[0] else None)
        ax.set_title(f"{label} | success={ep['success']} len={ep['episode_len']} | n_reanchor={len(reanchor_idxs)}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=6)
        ax.axis("equal")

        ax = axes[1, col]
        ee_to_gt = np.array([s["ee_to_target_dist"] for s in steps], dtype=float)
        replanned = np.array([bool(s["planner_replanned_this_step"]) for s in steps], dtype=bool)
        reactive_close = np.array([bool(s["reactive_close_event"]) for s in steps], dtype=bool)
        ax.plot(step_ix, ee_to_gt, c="tab:red", label="EE-to-GT distance")
        ax.axhline(0.03, color="black", ls=":", lw=0.8, label="close threshold")
        for r in np.where(replanned)[0]:
            ax.axvline(r, color="gray", lw=0.4, alpha=0.5)
        for c in np.where(reactive_close)[0]:
            ax.axvline(c, color="purple", lw=1.5, alpha=0.8, label="reactive-close" if c == np.where(reactive_close)[0][0] else None)
        for i in reanchor_idxs:
            ax.axvline(i, color="red", lw=1.0, alpha=0.6, ls="--", label="reanchor" if i == reanchor_idxs[0] else None)
        ax.set_xlabel("step")
        ax.set_ylabel("distance (m)")
        ax.set_title(f"{label} EE-to-GT (gray=replan, purple=close, red dashed=reanchor)")
        ax.legend(fontsize=6)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{task}_seed{seed}_A_vs_B.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    rows = []
    plots_dir = OUT_DIR / "comparison_plots"
    for task, seed in REPRESENTATIVE_EPISODES:
        ep_a = load_episode(COND_A_DIR, task, seed)
        ep_b = load_episode(COND_B_DIR, task, seed)
        if ep_a is None or ep_b is None:
            print(f"MISSING data for {task} seed={seed}: A={ep_a is not None} B={ep_b is not None}")
            continue
        m_a = full_metrics(ep_a)
        m_b = full_metrics(ep_b)
        rows.append({"task": task, "seed": seed, "A": m_a, "B": m_b})
        plot_comparison(task, seed, ep_a, ep_b, plots_dir)

    (OUT_DIR / "representative_comparison.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'task':14s} {'seed':5s} | {'success A/B':12s} {'n_reanchor':10s} {'min_ee_gt A->B':16s} {'osc A->B':12s} {'jerk A->B':14s} {'react_close A/B':16s}")
    for r in rows:
        a, b = r["A"], r["B"]
        print(
            f"{r['task']:14s} {r['seed']:<5d} | "
            f"{str(a['success']):5s}/{str(b['success']):5s}   "
            f"{b['n_reanchor_events']:<10d} "
            f"{a['min_ee_to_gt_dist_m']:.3f}->{b['min_ee_to_gt_dist_m']:.3f}  "
            f"{a['oscillation_count']:>4d}->{b['oscillation_count']:<4d} "
            f"{a['mean_abs_jerk_m']:.4f}->{b['mean_abs_jerk_m']:.4f}  "
            f"{a['n_reactive_close_events']}/{b['n_reactive_close_events']}"
        )
        if b["n_reanchor_events"] > 0:
            print(
                f"    reanchor details: offsets={b['chunk_offset_at_correction']} "
                f"applied_m={[round(x, 3) for x in b['applied_correction_m']]} "
                f"gap_before_m={[round(x, 3) for x in b['gap_before_m']]} "
                f"gap_after_m={[round(x, 3) for x in b['gap_after_m']]}"
            )

    print(f"\nwrote {OUT_DIR / 'representative_comparison.json'}, plots under {plots_dir}")


if __name__ == "__main__":
    main()
