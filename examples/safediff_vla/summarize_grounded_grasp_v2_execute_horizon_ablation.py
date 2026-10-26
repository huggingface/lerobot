#!/usr/bin/env python
"""Phase 1B analysis: aggregates `eval_grounded_grasp_v2_execute_horizon_ablation.py`'s per-
(execute_horizon, task) `episodes_raw.json` files, adds the oscillation/smoothness + terminal-
distance metrics the Phase 1B spec additionally requires on top of Phase 1A's own metric set
(reused via `analyze_episode`, imported not duplicated), a simple stable-but-biased (A) vs.
chunk-boundary-whiplash (B) HEURISTIC per-episode tag (visual check is still required to confirm
-- see the side-by-side plots this script also produces), and side-by-side (one column per
execute_horizon) trajectory comparison plots for the 4 episodes the Phase 1B spec names as
required visual-verification targets.

New metrics (on top of Phase 1A's `analyze_episode`):
    terminal_distance_m -- alias of `last_ee_to_gt_dist_m` (EE-to-GT distance at episode end).
    oscillation_count -- number of sign reversals in EE z-velocity (a direct proxy for the
        descend/retreat "sawtooth" pattern Phase 1A found visually at chunk boundaries).
    mean_abs_jerk_m -- mean ||second difference of EE xyz position|| across the episode (a
        general trajectory-smoothness metric, independent of the z-oscillation proxy above).
    pattern_tag -- "A_stable_biased" if monitor jitter is low (frac>2cm < 0.3) AND the episode
        never got close to GT (min_ee_to_gt_dist_m > 0.08), "B_whiplash" if planner_chunk_jump_
        mean_m > 0.08 AND oscillation_count is in the episode's own top half, "mixed/unclear"
        otherwise. HEURISTIC ONLY -- the governing instructions require visual confirmation
        before this tag is trusted; it exists to triage which episodes to look at first.

Usage:
    uv run python examples/safediff_vla/summarize_grounded_grasp_v2_execute_horizon_ablation.py
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
from summarize_grounded_grasp_v2_monitor_baseline import _vec_arr, analyze_episode  # noqa: E402

OUT_DIR_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_execute_horizon_ablation")
HORIZONS = [50, 10, 5]
REPRESENTATIVE_EPISODES = [
    ("select_poker", 1001),
    ("select_fruit", 1001),
    ("select_book", 1002),
    ("select_mahjong", 1000),
]


def extra_metrics(ep: dict) -> dict[str, Any]:
    steps = ep["steps"]
    ee_xyz = _vec_arr(steps, "ee_pos")
    z = ee_xyz[:, 2]
    vz = np.diff(z)
    sign = np.sign(vz)
    sign = sign[sign != 0]
    oscillation_count = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0
    accel = np.diff(ee_xyz, n=2, axis=0)
    mean_abs_jerk = float(np.linalg.norm(accel, axis=1).mean()) if len(accel) else None
    return {"oscillation_count": oscillation_count, "mean_abs_jerk_m": mean_abs_jerk}


def tag_pattern(m: dict) -> str:
    low_jitter = (m["monitor_jitter_frac_gt_2cm"] or 1.0) < 0.3
    never_close = (m["min_ee_to_gt_dist_m"] or 0.0) > 0.08
    big_jump = (m["planner_chunk_jump_mean_m"] or 0.0) > 0.08
    if low_jitter and never_close:
        return "A_stable_biased"
    if big_jump and m["oscillation_count"] >= 4:
        return "B_whiplash"
    return "mixed/unclear"


def load_and_analyze(out_dir_root: Path, horizons: list[int], tasks: list[str]) -> list[dict]:
    rows = []
    for h in horizons:
        for task in tasks:
            raw_path = out_dir_root / f"E{h}" / task / "episodes_raw.json"
            if not raw_path.exists():
                print(f"[E{h}/{task}] MISSING {raw_path}, skipping")
                continue
            episodes = json.loads(raw_path.read_text())
            for ep in episodes:
                m = analyze_episode(ep)
                m.update(extra_metrics(ep))
                m["terminal_distance_m"] = m["last_ee_to_gt_dist_m"]
                m["pattern_tag"] = tag_pattern(m)
                m["task"] = task
                m["execute_horizon"] = h
                rows.append(m)
    return rows


def agg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


NUMERIC_KEYS = [
    "mean_monitor_target_err_m", "mean_planner_target_err_m",
    "monitor_jitter_mean_m", "monitor_jitter_frac_gt_2cm", "monitor_jitter_frac_gt_5cm",
    "planner_chunk_jump_mean_m", "planner_chunk_jump_max_m", "n_planner_chunk_jumps",
    "planner_vs_monitor_gap_mean_m", "min_ee_to_gt_dist_m", "terminal_distance_m",
    "oscillation_count", "mean_abs_jerk_m",
]


def build_tables(rows: list[dict], tasks: list[str], horizons: list[int]) -> dict:
    overall = {}
    per_task = {}
    for h in horizons:
        h_rows = [r for r in rows if r["execute_horizon"] == h]
        if not h_rows:
            continue
        overall[f"E{h}"] = {
            "n_episodes": len(h_rows),
            "native_success_rate": float(np.mean([r["success"] for r in h_rows])),
            "privileged_grasp_rate": float(np.mean([r["ever_grasped_privileged"] for r in h_rows])),
            "reactive_close_rate": float(np.mean([r["n_reactive_close_events"] > 0 for r in h_rows])),
            "mean_reactive_close_ee_to_gt_m": agg(
                [{"v": d} for r in h_rows for d in r["reactive_close_ee_to_gt_dist_m"]], "v"
            ),
            **{k: agg(h_rows, k) for k in NUMERIC_KEYS},
        }
        for task in tasks:
            t_rows = [r for r in h_rows if r["task"] == task]
            if not t_rows:
                continue
            per_task.setdefault(task, {})[f"E{h}"] = {
                "n_episodes": len(t_rows),
                "native_success_rate": float(np.mean([r["success"] for r in t_rows])),
                "privileged_grasp_rate": float(np.mean([r["ever_grasped_privileged"] for r in t_rows])),
                **{k: agg(t_rows, k) for k in NUMERIC_KEYS},
            }
    return {"overall": overall, "per_task": per_task}


def plot_side_by_side(out_dir_root: Path, task: str, seed: int, horizons: list[int], plots_dir: Path) -> None:
    episodes_by_h = {}
    for h in horizons:
        raw_path = out_dir_root / f"E{h}" / task / "episodes_raw.json"
        if not raw_path.exists():
            continue
        episodes = json.loads(raw_path.read_text())
        for ep in episodes:
            if ep["seed"] == seed:
                episodes_by_h[h] = ep
                break
    present = [h for h in horizons if h in episodes_by_h]
    if not present:
        print(f"no episodes found for {task} seed={seed}")
        return

    fig, axes = plt.subplots(2, len(present), figsize=(5 * len(present), 9), squeeze=False)
    fig.suptitle(f"{task} seed={seed} -- E{{{','.join(str(h) for h in present)}}} side-by-side")

    for col, h in enumerate(present):
        ep = episodes_by_h[h]
        steps = ep["steps"]
        ee_xyz = _vec_arr(steps, "ee_pos")
        target_xyz = _vec_arr(steps, "target_pos")
        monitor_xyz = _vec_arr(steps, "monitor_target_world")
        planner_xyz = _vec_arr(steps, "planner_target_world")
        ee_to_gt = np.array([s["ee_to_target_dist"] for s in steps], dtype=float)
        replanned = np.array([bool(s["planner_replanned_this_step"]) for s in steps], dtype=bool)
        reactive_close = np.array([bool(s["reactive_close_event"]) for s in steps], dtype=bool)
        step_ix = np.arange(len(steps))
        valid_m = ~np.isnan(monitor_xyz).any(axis=1)
        valid_p = ~np.isnan(planner_xyz).any(axis=1)

        ax = axes[0, col]
        ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], c="tab:blue", lw=1, label="EE actual")
        ax.scatter(target_xyz[0, 0], target_xyz[0, 1], c="green", marker="*", s=200, label="GT target", zorder=5)
        if valid_m.any():
            ax.scatter(monitor_xyz[valid_m, 0], monitor_xyz[valid_m, 1], c=step_ix[valid_m], cmap="autumn", s=6, label="monitor target")
        if valid_p.any():
            ax.scatter(planner_xyz[valid_p, 0], planner_xyz[valid_p, 1], c="black", marker="x", s=25, label="planner target (committed)")
        ax.set_title(f"E{h} | success={ep['success']} len={ep['episode_len']}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=6)
        ax.axis("equal")

        ax = axes[1, col]
        ax.plot(step_ix, ee_to_gt, c="tab:red", label="EE-to-GT distance")
        ax.axhline(0.03, color="black", ls=":", lw=0.8, label="close threshold")
        for r in np.where(replanned)[0]:
            ax.axvline(r, color="gray", lw=0.4, alpha=0.5)
        for c in np.where(reactive_close)[0]:
            ax.axvline(c, color="red", lw=1.5, alpha=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("distance (m)")
        ax.set_title(f"E{h} EE-to-GT (gray=replan, red=close)")
        ax.legend(fontsize=6)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{task}_seed{seed}_side_by_side.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--output-dir", default=str(OUT_DIR_ROOT))
    args = parser.parse_args()
    out_dir_root = Path(args.output_dir)

    rows = load_and_analyze(out_dir_root, args.horizons, args.tasks)
    (out_dir_root / "episode_metrics.json").write_text(json.dumps(rows, indent=2))

    tables = build_tables(rows, args.tasks, args.horizons)
    (out_dir_root / "summary.json").write_text(json.dumps(tables, indent=2))
    print(json.dumps(tables["overall"], indent=2))

    pattern_counts = {}
    for h in args.horizons:
        h_rows = [r for r in rows if r["execute_horizon"] == h]
        pattern_counts[f"E{h}"] = {
            tag: sum(1 for r in h_rows if r["pattern_tag"] == tag)
            for tag in ("A_stable_biased", "B_whiplash", "mixed/unclear")
        }
    print("\npattern_tag counts per horizon:")
    print(json.dumps(pattern_counts, indent=2))
    (out_dir_root / "pattern_counts.json").write_text(json.dumps(pattern_counts, indent=2))

    plots_dir = out_dir_root / "side_by_side_plots"
    for task, seed in REPRESENTATIVE_EPISODES:
        plot_side_by_side(out_dir_root, task, seed, args.horizons, plots_dir)

    print(f"\nwrote {out_dir_root / 'summary.json'}, {out_dir_root / 'episode_metrics.json'}, {out_dir_root / 'pattern_counts.json'}")
    print(f"side-by-side plots under {plots_dir}")


if __name__ == "__main__":
    main()
