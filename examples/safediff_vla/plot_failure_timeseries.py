#!/usr/bin/env python
"""Pure visualization (NO training, NO new rollout -- reads
`rollout_with_rotation_for_failure_plots.py`'s already-saved `episodes_raw.json` files): for each
FAILED (`success=False`) episode of the SELECTED `phase_aware_100k_50k` checkpoint's 18-episode dev
set (seeds 1000-1002, 6 tasks), plots ~20-30 steps immediately before the grasp/close event
(or the last ~25 steps of the episode, for the one episode that never entered LOCAL_GRASP at all)
on one shared time axis:

    predicted_target_xyz, GT_target_xyz, commanded_xyz (= ee_pos + local_delta_xyz_m),
    actual_TCP_xyz (= ee_pos) -- one subplot per x/y/z axis, all 4 quantities overlaid
    predicted_delta_rot_rad, actual_ee_euler_rad -- one subplot, 6 lines (3 axes x 2 quantities,
        dashed=predicted delta, solid=actual absolute -- NOT the same physical quantity, see
        `rollout_with_rotation_for_failure_plots.py`'s own docstring, plotted together only
        because it was requested that way)
    gripper (applied_gripper) -- one subplot
    phase (stage: 0=PRE_GRASP/1=LOCAL_GRASP/2=POST_GRASP) -- background shading across every
        subplot AND its own bottom subplot, for unambiguous reading

NOTE on reproducibility: re-rolling out all 18 dev episodes (needed for rotation, see
`rollout_with_rotation_for_failure_plots.py`) flipped the success/fail OUTCOME on 4/18 episodes
(~22%) relative to the original dev-eval table that this script's episode SELECTION is based on --
closed-loop rollouts on this branch are not perfectly reproducible run-to-run even for the "same"
seed. Each plot's title reflects what THIS re-rollout actually did, not the (possibly now-stale)
original label.

Usage:
    uv run python examples/safediff_vla/plot_failure_timeseries.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SUMMARY_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k/phase_aware_100k_50k")
RAW_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_with_rotation")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_failure_timeseries")

WINDOW_BEFORE = 25  # steps before the close/grasp event (or before episode end, if never entered LOCAL_GRASP)
WINDOW_AFTER = 5    # steps after, for a little post-close context

PHASE_COLORS = {0: "#e8f0ff", 1: "#fff3d6", 2: "#e6ffe6"}  # PRE_GRASP / LOCAL_GRASP / POST_GRASP
PHASE_NAMES = {0: "PRE_GRASP", 1: "LOCAL_GRASP", 2: "POST_GRASP"}
AXIS_COLORS = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}


def find_failed_episodes() -> list[tuple[str, int]]:
    failed = []
    for task in STAGE2_TASK_NAMES:
        summary = json.loads((SUMMARY_DIR / task / "summary.json").read_text())
        for e in summary["per_episode"]:
            if not e["success"]:
                failed.append((task, e["seed"]))
    return failed


def pick_close_step(steps: list[dict]) -> int | None:
    latch = next((s["step_ix"] for s in steps if s.get("just_latched_gripper")), None)
    if latch is not None:
        return latch
    grasped = next((s["step_ix"] for s in steps if s.get("is_grasped")), None)
    if grasped is not None:
        return grasped
    return None


def plot_episode(task: str, seed: int, steps: list[dict], out_path: Path, actual_success: bool) -> None:
    close_ix = pick_close_step(steps)
    if close_ix is not None:
        lo, hi = max(0, close_ix - WINDOW_BEFORE), min(len(steps) - 1, close_ix + WINDOW_AFTER)
        anchor_note = f"close_ix={close_ix}"
    else:
        hi = len(steps) - 1
        lo = max(0, hi - WINDOW_BEFORE)
        anchor_note = "never closed -- last steps of episode shown"

    window = steps[lo : hi + 1]
    t = [s["step_ix"] for s in window]

    def col(key: str, sub=None):
        vals = []
        for s in window:
            v = s.get(key)
            if v is None:
                vals.append(None)
            elif sub is not None:
                vals.append(v[sub])
            else:
                vals.append(v)
        return vals

    gt = np.array(col("target_pos"))
    tcp = np.array(col("ee_pos"))
    pred_target = [s.get("predicted_target_world") for s in window]
    local_delta = [s.get("local_delta_xyz_m") for s in window]
    commanded = [
        (np.array(s["ee_pos"]) + np.array(d)).tolist() if d is not None else None
        for s, d in zip(window, local_delta, strict=True)
    ]
    pred_rot = [s.get("predicted_delta_rot_rad") for s in window]
    actual_euler = np.array(col("actual_ee_euler_rad"))
    gripper = col("applied_gripper")
    phase = col("stage")

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    ax_x, ax_y, ax_z, ax_rot, ax_grip = axes

    # phase background shading, drawn on every axis
    phase_filled = [p if p is not None else -1 for p in phase]
    for ax in axes:
        seg_start = 0
        for i in range(1, len(t) + 1):
            if i == len(t) or phase_filled[i] != phase_filled[seg_start]:
                p = phase_filled[seg_start]
                if p in PHASE_COLORS:
                    ax.axvspan(t[seg_start], t[i - 1] + 1, color=PHASE_COLORS[p], zorder=0)
                seg_start = i

    def plot_xyz_axis(ax, axis_idx: int, axis_name: str) -> None:
        ax.plot(t, gt[:, axis_idx], "k*-", label="GT_target", markersize=5)
        ax.plot(t, tcp[:, axis_idx], "s-", color="tab:blue", label="actual_TCP")
        pt = [p[axis_idx] if p is not None else None for p in pred_target]
        cm = [c[axis_idx] if c is not None else None for c in commanded]
        if any(v is not None for v in pt):
            ax.plot(t, [v if v is not None else np.nan for v in pt], "o--", color="tab:orange", label="predicted_target", markersize=3)
        if any(v is not None for v in cm):
            ax.plot(t, [v if v is not None else np.nan for v in cm], "^--", color="tab:green", label="commanded", markersize=3)
        ax.set_ylabel(f"{axis_name} (m)")
        ax.legend(fontsize=7, loc="upper left", ncol=4)
        ax.grid(alpha=0.3)

    plot_xyz_axis(ax_x, 0, "x")
    plot_xyz_axis(ax_y, 1, "y")
    plot_xyz_axis(ax_z, 2, "z")

    for i, axis_name in enumerate(("rx", "ry", "rz")):
        color = list(AXIS_COLORS.values())[i]
        pr = [p[i] if p is not None else None for p in pred_rot]
        if any(v is not None for v in pr):
            ax_rot.plot(t, [v if v is not None else np.nan for v in pr], "--", color=color, label=f"predicted_delta_{axis_name}")
        ax_rot.plot(t, actual_euler[:, i], "-", color=color, label=f"actual_{axis_name}")
    ax_rot.set_ylabel("rotation (rad)")
    ax_rot.legend(fontsize=6, loc="upper left", ncol=3)
    ax_rot.grid(alpha=0.3)

    ax_grip.plot(t, [g if g is not None else np.nan for g in gripper], "o-", color="tab:purple", label="applied_gripper")
    ax_grip.set_ylabel("gripper")
    ax_grip.set_xlabel(f"step_ix   ({anchor_note})")
    ax_grip.legend(fontsize=7, loc="upper left")
    ax_grip.grid(alpha=0.3)

    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=PHASE_NAMES[p]) for p, c in PHASE_COLORS.items()
    ]
    fig.legend(handles=phase_handles, loc="upper right", fontsize=8, title="phase (background)")
    # `actual_success` is THIS re-rollout's own outcome, not the original dev-eval table's --
    # closed-loop rollouts are not perfectly reproducible run-to-run (see
    # `plot_failure_timeseries.py`'s module docstring / the reproducibility check that motivated
    # this), so a stale "FAILED" label would misdescribe the data actually plotted.
    outcome_label = "FAILED (this rollout)" if not actual_success else "SUCCEEDED THIS ROLLOUT (was FAILED in original dev eval)"
    fig.suptitle(f"{task} seed={seed} (phase_aware_100k_50k, {outcome_label}): grasp-window time series", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    failed = find_failed_episodes()
    logger.info("=== %d episodes originally scored FAILED (phase_aware_100k_50k dev eval) ===", len(failed))
    n_flipped = 0
    for task, seed in failed:
        raw_path = RAW_DIR / task / "episodes_raw.json"
        episodes = json.loads(raw_path.read_text())
        ep = next((e for e in episodes if e["seed"] == seed), None)
        if ep is None:
            logger.warning("[%s] seed=%d not found in %s -- skipped", task, seed, raw_path)
            continue
        # Closed-loop rollouts on this branch are NOT perfectly reproducible run-to-run (confirmed
        # separately: re-rolling out all 18 dev episodes flipped success/fail on 4/18 -- ~22%),
        # so THIS re-rollout's own `success` may differ from the original dev-eval table's. The
        # plot's title reflects what was ACTUALLY simulated here, not the stale original label.
        if ep["success"]:
            n_flipped += 1
            logger.warning("[%s] seed=%d: FLIPPED -- originally FAILED, this re-rollout SUCCEEDED", task, seed)
        out_path = OUT_DIR / task / f"seed{seed}_timeseries.png"
        plot_episode(task, seed, ep["steps"], out_path, actual_success=ep["success"])
        logger.info("[%s] seed=%d -> wrote %s", task, seed, out_path)
    logger.info("=== %d/%d flipped to success in this re-rollout (closed-loop non-reproducibility) ===", n_flipped, len(failed))


if __name__ == "__main__":
    main()
