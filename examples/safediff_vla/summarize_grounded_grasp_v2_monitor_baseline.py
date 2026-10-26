#!/usr/bin/env python
"""Phase 1A analysis: aggregates `eval_grounded_grasp_v2_monitor_baseline.py`'s per-task
`episodes_raw.json` files into the quantitative summary + per-episode trajectory plots requested
for the baseline diagnostic (6 tasks x seeds 1000-1002, `architecture="temporal_decoder_grounded_
grasp_v2"`, frozen checkpoint, NO behavior change -- see that script's own docstring for exactly
what is/isn't logged and why).

Metrics (per episode, then aggregated per-task and overall):
    - mean/last monitor_target_dist_to_gt, planner_target_dist_to_gt (target localization error)
    - monitor frame-to-frame displacement: mean, max, fraction of steps >2cm, fraction >5cm
      (this is the REAL jitter signal -- see the eval script's docstring on why the planner's own
      cached target is a step function, not a per-frame quantity, under the canonical
      action_horizon=execute_horizon=50 config)
    - planner chunk-boundary jump: displacement between consecutive REPLAN events only
    - planner-vs-monitor same-step gap (mean, last): how stale the planner's conditioning is
      relative to the current (monitor) grounding at the moment a chunk commits
    - min/last ee_to_target_dist (actual EE to GT target, world frame)
    - reactive-close event count + ee_to_target_dist at each such event (false/premature-close
      diagnostic)
    - native success (`episodes_raw.json`'s own `success`), is_grasped (privileged contact, any
      step), final phase

Trajectory plots (one PNG per episode, `<out_dir>/<task>/plots/episode_<ix>.png`): top-down XY
(EE actual vs GT target vs monitor target, colored by step index), Z height over time, monitor
target displacement over time (replan steps marked as vertical lines, reactive-close event marked),
ee_to_target_dist over time. Metric-only conclusions are explicitly disallowed by the governing
instructions for this branch -- these plots (and the already-saved `videos/eval_episode_*.mp4`)
are the required visual-verification artifact, not optional decoration.

Usage:
    uv run python examples/safediff_vla/summarize_grounded_grasp_v2_monitor_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline")
JUMP_THRESHOLDS_M = (0.02, 0.05)


def _arr(steps: list[dict], key: str) -> np.ndarray:
    return np.array([s[key] if s[key] is not None else np.nan for s in steps], dtype=float)


def _vec_arr(steps: list[dict], key: str) -> np.ndarray:
    return np.array(
        [s[key] if s[key] is not None else [np.nan, np.nan, np.nan] for s in steps], dtype=float
    )


def analyze_episode(ep: dict) -> dict[str, Any]:
    steps = ep["steps"]
    monitor_xyz = _vec_arr(steps, "monitor_target_world")
    planner_xyz = _vec_arr(steps, "planner_target_world")
    monitor_dist_gt = _arr(steps, "monitor_target_dist_to_gt")
    planner_dist_gt = _arr(steps, "planner_target_dist_to_gt")
    ee_to_gt = _arr(steps, "ee_to_target_dist")
    replanned = np.array([bool(s["planner_replanned_this_step"]) for s in steps], dtype=bool)
    reactive_close = np.array([bool(s["reactive_close_event"]) for s in steps], dtype=bool)
    is_grasped = np.array([bool(s["is_grasped"]) for s in steps], dtype=bool)

    # Monitor frame-to-frame displacement -- only between consecutive steps where BOTH have a
    # valid monitor reading (step 0, the post-reset record, never does: no select_action call yet).
    valid = ~np.isnan(monitor_xyz).any(axis=1)
    disp = np.full(len(steps), np.nan)
    for t in range(1, len(steps)):
        if valid[t] and valid[t - 1]:
            disp[t] = float(np.linalg.norm(monitor_xyz[t] - monitor_xyz[t - 1]))
    disp_valid = disp[~np.isnan(disp)]

    # Planner chunk-boundary jump: displacement between consecutive REPLAN events only.
    replan_idxs = np.where(replanned)[0]
    planner_jumps = []
    for i in range(1, len(replan_idxs)):
        a, b = replan_idxs[i - 1], replan_idxs[i]
        if valid[a] and valid[b]:
            planner_jumps.append(float(np.linalg.norm(planner_xyz[b] - planner_xyz[a])))
    planner_jumps = np.array(planner_jumps, dtype=float)

    # Planner-vs-monitor same-step gap (how stale the cached planner target is right now).
    both_valid = valid & ~np.isnan(planner_xyz).any(axis=1)
    gap = np.full(len(steps), np.nan)
    gap[both_valid] = np.linalg.norm((monitor_xyz - planner_xyz)[both_valid], axis=1)

    reactive_close_idxs = np.where(reactive_close)[0].tolist()
    reactive_close_ee_to_gt = [float(ee_to_gt[i]) for i in reactive_close_idxs]

    return {
        "episode_ix": ep["episode_ix"],
        "seed": ep["seed"],
        "episode_len": ep["episode_len"],
        "success": bool(ep["success"]),
        "ever_grasped_privileged": bool(is_grasped.any()),
        "final_phase": int(steps[-1]["phase"]) if steps[-1]["phase"] is not None else None,
        "n_replans": int(replanned.sum()),
        "mean_monitor_target_err_m": float(np.nanmean(monitor_dist_gt)),
        "last_monitor_target_err_m": float(monitor_dist_gt[~np.isnan(monitor_dist_gt)][-1]) if valid.any() else None,
        "mean_planner_target_err_m": float(np.nanmean(planner_dist_gt)),
        "last_planner_target_err_m": float(planner_dist_gt[~np.isnan(planner_dist_gt)][-1]) if both_valid.any() else None,
        "monitor_jitter_mean_m": float(disp_valid.mean()) if len(disp_valid) else None,
        "monitor_jitter_max_m": float(disp_valid.max()) if len(disp_valid) else None,
        "monitor_jitter_frac_gt_2cm": float((disp_valid > JUMP_THRESHOLDS_M[0]).mean()) if len(disp_valid) else None,
        "monitor_jitter_frac_gt_5cm": float((disp_valid > JUMP_THRESHOLDS_M[1]).mean()) if len(disp_valid) else None,
        "planner_chunk_jump_mean_m": float(planner_jumps.mean()) if len(planner_jumps) else None,
        "planner_chunk_jump_max_m": float(planner_jumps.max()) if len(planner_jumps) else None,
        "n_planner_chunk_jumps": int(len(planner_jumps)),
        "planner_vs_monitor_gap_mean_m": float(np.nanmean(gap)) if both_valid.any() else None,
        "planner_vs_monitor_gap_last_m": float(gap[both_valid][-1]) if both_valid.any() else None,
        "min_ee_to_gt_dist_m": float(np.nanmin(ee_to_gt)),
        "last_ee_to_gt_dist_m": float(ee_to_gt[-1]),
        "n_reactive_close_events": len(reactive_close_idxs),
        "reactive_close_ee_to_gt_dist_m": reactive_close_ee_to_gt,
    }


def plot_episode(task: str, ep: dict, plots_dir: Path) -> None:
    steps = ep["steps"]
    ee_xyz = _vec_arr(steps, "ee_pos")
    target_xyz = _vec_arr(steps, "target_pos")
    monitor_xyz = _vec_arr(steps, "monitor_target_world")
    ee_to_gt = _arr(steps, "ee_to_target_dist")
    replanned = np.array([bool(s["planner_replanned_this_step"]) for s in steps], dtype=bool)
    reactive_close = np.array([bool(s["reactive_close_event"]) for s in steps], dtype=bool)
    step_ix = np.arange(len(steps))

    valid = ~np.isnan(monitor_xyz).any(axis=1)
    disp = np.full(len(steps), np.nan)
    for t in range(1, len(steps)):
        if valid[t] and valid[t - 1]:
            disp[t] = float(np.linalg.norm(monitor_xyz[t] - monitor_xyz[t - 1]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    title = (
        f"{task} | episode {ep['episode_ix']} seed={ep['seed']} "
        f"success={ep['success']} len={ep['episode_len']}"
    )
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], c="tab:blue", lw=1, label="EE actual (xy)")
    ax.scatter(target_xyz[0, 0], target_xyz[0, 1], c="green", marker="*", s=200, label="GT target", zorder=5)
    valid_monitor = monitor_xyz[valid]
    if len(valid_monitor):
        sc = ax.scatter(
            valid_monitor[:, 0], valid_monitor[:, 1], c=step_ix[valid], cmap="autumn", s=8, label="monitor target (xy)"
        )
        fig.colorbar(sc, ax=ax, label="step_ix")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down XY trajectory")
    ax.legend(fontsize=7)
    ax.axis("equal")

    ax = axes[0, 1]
    ax.plot(step_ix, ee_xyz[:, 2], c="tab:blue", label="EE z (actual)")
    ax.plot(step_ix, target_xyz[:, 2], c="green", ls="--", label="GT target z")
    ax.plot(step_ix[valid], monitor_xyz[valid, 2], c="tab:orange", lw=0.7, label="monitor target z")
    for r in np.where(replanned)[0]:
        ax.axvline(r, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("z (m)")
    ax.set_title("Height over time (gray = chunk replan)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(step_ix, disp, c="tab:purple", lw=0.8, label="monitor frame-to-frame displacement")
    ax.axhline(0.02, color="orange", ls=":", lw=0.8, label="2cm")
    ax.axhline(0.05, color="red", ls=":", lw=0.8, label="5cm")
    for r in np.where(replanned)[0]:
        ax.axvline(r, color="gray", lw=0.5, alpha=0.4)
    for c in np.where(reactive_close)[0]:
        ax.axvline(c, color="red", lw=1.5, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("displacement (m)")
    ax.set_title("Monitor target jitter (red line = reactive-close event)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(step_ix, ee_to_gt, c="tab:red", label="EE-to-GT-target distance")
    ax.axhline(0.03, color="black", ls=":", lw=0.8, label="reactive-close threshold (0.03m)")
    for c in np.where(reactive_close)[0]:
        ax.axvline(c, color="red", lw=1.5, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("distance (m)")
    ax.set_title("EE-to-GT-target distance")
    ax.legend(fontsize=7)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"episode_{ep['episode_ix']}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    all_episode_metrics = []
    for task in args.tasks:
        raw_path = out_dir / task / "episodes_raw.json"
        if not raw_path.exists():
            print(f"[{task}] MISSING {raw_path}, skipping")
            continue
        episodes = json.loads(raw_path.read_text())
        plots_dir = out_dir / task / "plots"
        for ep in episodes:
            m = analyze_episode(ep)
            m["task"] = task
            all_episode_metrics.append(m)
            plot_episode(task, ep, plots_dir)

    (out_dir / "episode_metrics.json").write_text(json.dumps(all_episode_metrics, indent=2))

    def agg(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    numeric_keys = [
        "mean_monitor_target_err_m", "last_monitor_target_err_m",
        "mean_planner_target_err_m", "last_planner_target_err_m",
        "monitor_jitter_mean_m", "monitor_jitter_max_m",
        "monitor_jitter_frac_gt_2cm", "monitor_jitter_frac_gt_5cm",
        "planner_chunk_jump_mean_m", "planner_chunk_jump_max_m",
        "planner_vs_monitor_gap_mean_m", "planner_vs_monitor_gap_last_m",
        "min_ee_to_gt_dist_m", "last_ee_to_gt_dist_m",
    ]

    per_task_summary = {}
    for task in args.tasks:
        rows = [r for r in all_episode_metrics if r["task"] == task]
        if not rows:
            continue
        per_task_summary[task] = {
            "n_episodes": len(rows),
            "native_success_rate": float(np.mean([r["success"] for r in rows])),
            "privileged_grasp_rate": float(np.mean([r["ever_grasped_privileged"] for r in rows])),
            "mean_n_reactive_close_events": agg(rows, "n_reactive_close_events"),
            **{k: agg(rows, k) for k in numeric_keys},
        }

    overall = {
        "n_episodes": len(all_episode_metrics),
        "native_success_rate": float(np.mean([r["success"] for r in all_episode_metrics])),
        "privileged_grasp_rate": float(np.mean([r["ever_grasped_privileged"] for r in all_episode_metrics])),
        "mean_n_reactive_close_events": agg(all_episode_metrics, "n_reactive_close_events"),
        **{k: agg(all_episode_metrics, k) for k in numeric_keys},
    }

    summary = {"per_task": per_task_summary, "overall": overall}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'} and {out_dir / 'episode_metrics.json'}")
    print(f"per-episode plots under {out_dir}/<task>/plots/episode_<ix>.png")


if __name__ == "__main__":
    main()
