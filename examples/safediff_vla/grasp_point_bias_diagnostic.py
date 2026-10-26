#!/usr/bin/env python
"""Pure diagnostic (NO architecture/training change) requested before any "final evaluation":
does the v2.5 multitask Stage 2 checkpoint show a SYSTEMATIC signed offset between where it aims
for a grasp and the real object center -- the same direction repeating across tasks -- rather than
just per-episode noise?

Reuses the Stage 2 isolated eval's ALREADY-COLLECTED per-step physics records
(`outputs/eval/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/multitask_stage2_v2_5_5k/<task>/
episodes_raw.json`, 6 tasks x 3 episodes, select_ingredient excluded per the standing train/eval
mismatch finding) -- NO new rollout, NO simulator call. Pure re-analysis of data already on disk.

For each episode, picks the grasp/close moment (in priority order, logged per episode so a reader
can see which rule fired): 1) the step LocalGraspRefiner's own gripper-close latch fires
(`just_latched_gripper`), 2) else the first ground-truth `is_grasped` step, 3) else the LOCAL_GRASP
step with the smallest EE<->object distance (episode never actually closed/grasped -- still the
closest thing to a "final approach point" available). Records, all in WORLD frame meters:

    predicted_target_xyz   -- v2's own TargetPointHead output at that step (grounding prediction)
    GT_object_center_xyz   -- real object position (`Entity.get_xpos`, ground truth)
    commanded_EE_xyz        -- current_EE + LocalGraspRefiner's own delta_xyz from the PRECEDING
                               step (i.e. where the controller told the arm to go to land here --
                               `delta_xyz` is emitted in robot-base frame, but robot-base frame is a
                               pure translation of world frame in this env -- see
                               `envs/vlabench.py`'s `pos_robot = pos_world - base`, no rotation --
                               so the delta vector itself is identical in both frames and can be
                               added directly to a world-frame EE position)
    actual_TCP_xyz          -- the resulting real EE/gripper-center position (physics, next frame)

and the 3 signed residual vectors this diagnostic exists to characterize:
    predicted_target - GT_center   (grounding bias)
    commanded_EE - GT_center        (controller's own aim bias)
    actual_TCP - GT_center          (where the arm actually ends up, physical bias)

Per-task AND overall means are reported as signed (not absolute) 3-vectors specifically so a
repeating same-direction offset shows up as a large mean relative to its std, instead of averaging
to ~0 the way genuine per-episode noise would.

Usage:
    uv run python examples/safediff_vla/grasp_point_bias_diagnostic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EPISODES_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/multitask_stage2_v2_5_5k")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_bias_diagnostic")


def pick_close_step(steps: list[dict]) -> tuple[int | None, str]:
    latch = next((s["step_ix"] for s in steps if s.get("just_latched_gripper")), None)
    if latch is not None:
        return latch, "gripper_close_latch"
    grasped = next((s["step_ix"] for s in steps if s.get("is_grasped")), None)
    if grasped is not None:
        return grasped, "first_is_grasped"
    local_steps = [s for s in steps if s.get("stage") == 1]  # LOCAL_GRASP == 1
    if local_steps:
        best = min(local_steps, key=lambda s: s["ee_to_target_dist"])
        return best["step_ix"], "min_dist_during_local_grasp_never_closed"
    return None, "no_local_grasp_entry"


def nearest_predicted_target(steps: list[dict], step_ix: int) -> list[float] | None:
    """`predicted_target_world` is only set on steps where a replan fired -- during LOCAL_GRASP
    that's every step, but fall back to the most recent prior value just in case."""
    for s in reversed(steps[: step_ix + 1]):
        if s.get("predicted_target_world") is not None:
            return s["predicted_target_world"]
    return None


def analyze_episode(steps: list[dict]) -> dict | None:
    close_ix, rule = pick_close_step(steps)
    if close_ix is None:
        return None
    close = steps[close_ix]
    gt_center = np.asarray(close["target_pos"], dtype=float)
    actual_tcp = np.asarray(close["ee_pos"], dtype=float)
    predicted_target = nearest_predicted_target(steps, close_ix)
    predicted_target = np.asarray(predicted_target, dtype=float) if predicted_target is not None else None

    commanded_ee = None
    if close_ix > 0:
        prev = steps[close_ix - 1]
        if prev.get("local_delta_xyz_m") is not None:
            commanded_ee = np.asarray(prev["ee_pos"], dtype=float) + np.asarray(prev["local_delta_xyz_m"], dtype=float)

    def resid(a: np.ndarray | None) -> list[float] | None:
        return (a - gt_center).tolist() if a is not None else None

    return {
        "close_step_ix": close_ix,
        "close_step_rule": rule,
        "predicted_target_xyz": predicted_target.tolist() if predicted_target is not None else None,
        "GT_object_center_xyz": gt_center.tolist(),
        "commanded_EE_xyz": commanded_ee.tolist() if commanded_ee is not None else None,
        "actual_TCP_xyz": actual_tcp.tolist(),
        "residual_predicted_target_minus_GT": resid(predicted_target),
        "residual_commanded_EE_minus_GT": resid(commanded_ee),
        "residual_actual_TCP_minus_GT": resid(actual_tcp),
    }


def mean_vec(vecs: list[list[float] | None]) -> dict:
    arr = np.asarray([v for v in vecs if v is not None], dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean_xyz": None, "std_xyz": None, "mean_norm_m": None, "mean_over_std": None}
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return {
        "n": len(arr),
        "mean_xyz": mean.tolist(),
        "std_xyz": std.tolist(),
        "mean_norm_m": float(np.linalg.norm(mean)),
        # signal-to-noise per axis: large |mean/std| means the offset repeats in the same direction
        # rather than averaging out -- std==0 (n=1) guarded.
        "mean_over_std_xyz": [float(m / s) if s > 1e-9 else None for m, s in zip(mean, std, strict=True)],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_task_episodes: dict[str, list[dict]] = {}
    all_records: list[dict] = []

    for task in STAGE2_TASK_NAMES:
        raw_path = EPISODES_DIR / task / "episodes_raw.json"
        episode_records = json.loads(raw_path.read_text())
        task_analyses = []
        for ep in episode_records:
            analysis = analyze_episode(ep["steps"])
            if analysis is None:
                logger.warning("[%s] episode %d: no LOCAL_GRASP entry at all -- skipped", task, ep["episode_ix"])
                continue
            analysis["task"] = task
            analysis["episode_ix"] = ep["episode_ix"]
            analysis["seed"] = ep.get("seed")
            task_analyses.append(analysis)
            all_records.append(analysis)
        per_task_episodes[task] = task_analyses
        logger.info(
            "[%s] %d/%d episodes analyzed (rules used: %s)",
            task, len(task_analyses), len(episode_records),
            [a["close_step_rule"] for a in task_analyses],
        )

    per_task_summary = {}
    for task, analyses in per_task_episodes.items():
        per_task_summary[task] = {
            "n_episodes": len(analyses),
            "predicted_target_minus_GT": mean_vec([a["residual_predicted_target_minus_GT"] for a in analyses]),
            "commanded_EE_minus_GT": mean_vec([a["residual_commanded_EE_minus_GT"] for a in analyses]),
            "actual_TCP_minus_GT": mean_vec([a["residual_actual_TCP_minus_GT"] for a in analyses]),
        }

    overall_summary = {
        "n_episodes": len(all_records),
        "predicted_target_minus_GT": mean_vec([a["residual_predicted_target_minus_GT"] for a in all_records]),
        "commanded_EE_minus_GT": mean_vec([a["residual_commanded_EE_minus_GT"] for a in all_records]),
        "actual_TCP_minus_GT": mean_vec([a["residual_actual_TCP_minus_GT"] for a in all_records]),
    }

    report = {
        "checkpoint": "outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/checkpoints/005000/pretrained_model",
        "coordinate_frame": "world (meters); see module docstring on why robot-base-frame deltas add directly",
        "per_episode": all_records,
        "per_task_summary": per_task_summary,
        "overall_summary": overall_summary,
    }
    (OUT_DIR / "grasp_point_bias_report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", OUT_DIR / "grasp_point_bias_report.json")

    logger.info("=== per-task mean signed residual (x, y, z) in meters ===")
    for task, s in per_task_summary.items():
        for name in ("predicted_target_minus_GT", "commanded_EE_minus_GT", "actual_TCP_minus_GT"):
            v = s[name]
            if v["mean_xyz"] is None:
                logger.info("%-14s %-28s n=0", task, name)
                continue
            mx, my, mz = v["mean_xyz"]
            logger.info(
                "%-14s %-28s n=%d mean=(%+.3f, %+.3f, %+.3f) |mean|=%.3f m/std=%s",
                task, name, v["n"], mx, my, mz, v["mean_norm_m"],
                [f"{x:+.1f}" if x is not None else "n/a" for x in v["mean_over_std_xyz"]],
            )
    logger.info("=== OVERALL (all 6 tasks pooled) ===")
    for name in ("predicted_target_minus_GT", "commanded_EE_minus_GT", "actual_TCP_minus_GT"):
        v = overall_summary[name]
        mx, my, mz = v["mean_xyz"]
        logger.info(
            "%-28s n=%d mean=(%+.3f, %+.3f, %+.3f) |mean|=%.3f m/std=%s",
            name, v["n"], mx, my, mz, v["mean_norm_m"],
            [f"{x:+.1f}" if x is not None else "n/a" for x in v["mean_over_std_xyz"]],
        )


if __name__ == "__main__":
    main()
