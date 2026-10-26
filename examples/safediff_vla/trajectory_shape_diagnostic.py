#!/usr/bin/env python
"""Final pre-eval diagnostic (NO new training/architecture change): the earlier world-frame and
image-plane SIGNED-bias diagnostics found no single consistent lateral direction across the 6
tasks. This script checks a different, more specific hypothesis the user's own video review
raised: not "always drifts one way" but "approaches the object fine, then wobbles/overshoots/
mistimes the final close instead of converging cleanly onto the contact point."

100% reuses already-collected data -- NO new rollout:
  - per-timestep physics (`outputs/eval/.../multitask_stage2_v2_5_5k/<task>/episodes_raw.json`,
    the Stage 2 checkpoint's isolated 6-task x 3-episode eval, every step already has GT object
    position, v2's predicted_target_world, LocalGraspRefiner's own delta_xyz, actual EE position,
    gripper state, and every LOCAL_GRASP/close-transition flag)
  - the per-(task,seed) camera projection matrices already computed by
    `camera_projection_bias_diagnostic.py`
  - the already-rendered rollout videos (`.../videos/eval_episode_N.mp4`) for the image-plane
    overlay background frame

For each episode, the window analyzed is LOCAL_GRASP entry -> close/grasp (+ up to 5 extra steps
to look for post-contact instability). Per timestep: GT_object_center_xyz, predicted_target_xyz,
commanded_EE_xyz (= that step's EE + LocalGraspRefiner's own delta_xyz -- the "where it's aiming
this step" quantity), actual_TCP_xyz (physics EE position), gripper value, distance to GT object,
distance to v2's own predicted target.

Two visualizations per episode (saved, not just aggregated -- every episode, per the request):
  1. 3D trajectory plot (GT center, predicted-target path, commanded-EE path, actual-TCP path,
     entry/close/grasp markers)
  2. Image-plane overlay: the close-step video frame with all 4 quantities' pixel-projected paths
     drawn on top, in time order

Last-10-steps-before-close geometry: `lateral`/`radial` decompose `actual_TCP - GT_center` against
a FIXED approach axis (defined once per episode: direction from the TCP's own position 10 steps
before close, toward GT center at close) -- `radial` is the signed progress along that axis
(positive = still on the approach side, negative = overshot past the object), `lateral` is the
perpendicular-distance magnitude (drift off that axis). `step-to-step movement direction` is the
cosine similarity between consecutive velocity vectors (near +1 = smooth, negative = reversing/
zigzagging).

Failure-mode classification is RULE-BASED on these already-logged numbers (thresholds tied to the
policy's OWN config constants -- `close_threshold_m=0.03`, `local_grasp_radius_m=0.10` -- not
invented ad hoc), and every trigger is logged alongside its threshold and actual value so the label
is auditable, not a black box. This script does NOT propose any architecture change -- it only
reports which failure mode(s), if any, actually recur.

Usage:
    uv run python examples/safediff_vla/trajectory_shape_diagnostic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EPISODES_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/multitask_stage2_v2_5_5k")
CAMERA_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_camera_projection_bias_diagnostic")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_trajectory_shape_diagnostic")

CLOSE_THRESHOLD_M = 0.03  # SafeDiffVLAConfig.grounded_grasp_close_threshold_m
LOCAL_RADIUS_M = 0.10  # SafeDiffVLAConfig.local_grasp_radius_m
POST_CONTACT_TAIL = 5
LAST_N_STEPS = 10

FAILURE_MODES = (
    "unstable_target_prediction", "local_lateral_drift", "overshoot", "premature_close",
    "delayed_close", "insufficient_approach", "post_contact_instability", "no_clear_local_failure",
)


def pick_close_step(steps: list[dict]) -> tuple[int | None, str]:
    latch = next((s["step_ix"] for s in steps if s.get("just_latched_gripper")), None)
    if latch is not None:
        return latch, "gripper_close_latch"
    grasped = next((s["step_ix"] for s in steps if s.get("is_grasped")), None)
    if grasped is not None:
        return grasped, "first_is_grasped"
    local_steps = [s for s in steps if s.get("stage") == 1]
    if local_steps:
        best = min(local_steps, key=lambda s: s["ee_to_target_dist"])
        return best["step_ix"], "min_dist_during_local_grasp_never_closed"
    return None, "no_local_grasp_entry"


def build_episode_trajectory(steps: list[dict]) -> dict | None:
    entry_ix = next((i for i, s in enumerate(steps) if s.get("stage") == 1), None)
    if entry_ix is None:
        return None
    close_ix, close_rule = pick_close_step(steps)
    end_ix = min(close_ix + POST_CONTACT_TAIL, len(steps) - 1)

    window = steps[entry_ix : end_ix + 1]
    traj: dict[str, list] = {
        "step_ix": [], "GT_center_xyz": [], "predicted_target_xyz": [], "commanded_EE_xyz": [],
        "actual_TCP_xyz": [], "gripper": [], "dist_to_gt": [], "dist_to_predicted": [], "stage": [],
        "is_grasped": [],
    }
    for i, s in enumerate(window):
        abs_ix = entry_ix + i
        gt = np.asarray(s["target_pos"], dtype=float)
        tcp = np.asarray(s["ee_pos"], dtype=float)
        pred = np.asarray(s["predicted_target_world"], dtype=float) if s.get("predicted_target_world") is not None else None
        cmd = tcp + np.asarray(s["local_delta_xyz_m"], dtype=float) if s.get("local_delta_xyz_m") is not None else None

        traj["step_ix"].append(abs_ix)
        traj["GT_center_xyz"].append(gt.tolist())
        traj["predicted_target_xyz"].append(pred.tolist() if pred is not None else None)
        traj["commanded_EE_xyz"].append(cmd.tolist() if cmd is not None else None)
        traj["actual_TCP_xyz"].append(tcp.tolist())
        traj["gripper"].append(s.get("applied_gripper"))
        traj["dist_to_gt"].append(s["ee_to_target_dist"])
        traj["dist_to_predicted"].append(float(np.linalg.norm(tcp - pred)) if pred is not None else None)
        traj["stage"].append(s.get("stage"))
        traj["is_grasped"].append(bool(s.get("is_grasped")))

    traj["entry_ix"] = entry_ix
    traj["close_ix"] = close_ix
    traj["close_rule"] = close_rule
    traj["end_ix"] = end_ix
    traj["grasp_ix"] = next((s["step_ix"] for s in steps if s.get("is_grasped")), None)
    return traj


def compute_full_window_geometry(traj: dict) -> dict:
    """lateral/radial for EVERY step from LOCAL_GRASP entry to end_ix, decomposed against ONE FIXED
    axis anchored at ENTRY (not at the last-N window's own start -- anchoring at the window's own
    start makes lateral trivially ~0 there by construction, which made an earlier version of this
    diagnostic's drift check fire on almost every episode regardless of actual behavior). Axis =
    direction from GT object center to the entry TCP position, so `radial` is signed progress along
    the entry->object line (positive = still short of the object, ~0 = at it, negative = overshot
    past it) and `lateral` is the perpendicular deviation off that line -- exactly what's needed for
    "does it converge monotonically" and "does it drift/overshoot" questions over the WHOLE
    approach, not just an artificially self-zeroed final slice."""
    tcps = np.asarray(traj["actual_TCP_xyz"])
    gt = np.asarray(traj["GT_center_xyz"][-1])  # ~static object; use the value nearest close
    entry_pos = tcps[0]
    axis_vec = entry_pos - gt
    axis_norm = np.linalg.norm(axis_vec)
    axis = axis_vec / axis_norm if axis_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    lateral, radial = [], []
    for p in tcps:
        v = p - gt
        r = float(np.dot(v, axis))
        lat = float(np.linalg.norm(v - r * axis))
        radial.append(r)
        lateral.append(lat)

    return {
        "step_ix": traj["step_ix"],
        "lateral_dist_m": lateral,
        "radial_signed_dist_m": radial,
        "axis_anchor": "entry->GT_center direction (fixed for whole episode)",
    }


def compute_last_n_geometry(traj: dict, full_geom: dict, n: int = LAST_N_STEPS) -> dict:
    close_ix = traj["close_ix"]
    idx_in_window = [i for i, six in enumerate(traj["step_ix"]) if six <= close_ix]
    idx_in_window = idx_in_window[-n:]
    if len(idx_in_window) < 2:
        return {"n": len(idx_in_window), "note": "fewer than 2 steps available before close"}

    tcps = np.asarray([traj["actual_TCP_xyz"][i] for i in idx_in_window])
    move_dirs = np.diff(tcps, axis=0)
    move_dir_cos_sim = []
    for a, b in zip(move_dirs[:-1], move_dirs[1:], strict=True):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        move_dir_cos_sim.append(float(np.dot(a, b) / (na * nb)) if na > 1e-6 and nb > 1e-6 else None)

    return {
        "n": len(idx_in_window),
        "step_ix": [traj["step_ix"][i] for i in idx_in_window],
        "lateral_dist_m": [full_geom["lateral_dist_m"][i] for i in idx_in_window],
        "radial_signed_dist_m": [full_geom["radial_signed_dist_m"][i] for i in idx_in_window],
        "move_dir_cos_sim_consecutive": move_dir_cos_sim,
        "mean_cos_sim": float(np.mean([c for c in move_dir_cos_sim if c is not None])) if move_dir_cos_sim else None,
    }


def classify_failure(traj: dict, last_n: dict) -> dict:
    triggers: dict[str, dict] = {}
    dist_to_gt = np.asarray(traj["dist_to_gt"])
    step_ix = np.asarray(traj["step_ix"])
    close_ix = traj["close_ix"]
    close_local_pos = int(np.where(step_ix == close_ix)[0][0])
    min_pos = int(np.argmin(dist_to_gt))
    min_dist = float(dist_to_gt[min_pos])
    min_step = int(step_ix[min_pos])
    close_dist = float(dist_to_gt[close_local_pos])

    # 1. unstable target prediction: predicted_target itself jitters while GT is static.
    preds = np.asarray([p for p in traj["predicted_target_xyz"] if p is not None])
    if len(preds) >= 3:
        jumps = np.linalg.norm(np.diff(preds, axis=0), axis=1)
        max_jump, mean_jump = float(jumps.max()), float(jumps.mean())
        if max_jump > 0.05 or mean_jump > 0.02:
            triggers["unstable_target_prediction"] = {"max_step_jump_m": max_jump, "mean_step_jump_m": mean_jump, "threshold": "max>0.05 or mean>0.02"}

    # 2. local lateral drift: lateral distance in the last-N window fails to shrink by a meaningful
    #    RELATIVE margin (not a fixed absolute cm bar -- that penalizes episodes that start the
    #    window already-close unfairly) AND the final lateral is still non-negligible in absolute
    #    terms (a final lateral of a few mm is not "drift" regardless of the relative ratio).
    lat = last_n.get("lateral_dist_m")
    if lat and len(lat) >= 3:
        shrank_enough = lat[-1] <= lat[0] * 0.7
        still_negligible = lat[-1] <= 0.005
        if not (shrank_enough or still_negligible):
            triggers["local_lateral_drift"] = {
                "lateral_at_window_start_m": lat[0], "lateral_at_close_m": lat[-1],
                "threshold": "flagged unless close <= 0.7x start OR close <= 0.005m",
            }

    # 3. overshoot: minimum distance reached BEFORE close, by a clear margin, i.e. arm passed the
    #    closest point and distance grew again before closing.
    if min_step < close_ix and (close_dist - min_dist) > 0.01:
        triggers["overshoot"] = {"min_dist_m": min_dist, "min_dist_step": min_step, "close_dist_m": close_dist, "close_step": close_ix, "threshold": "close_dist - min_dist > 0.01 and min occurs before close"}

    # 4. premature close: gripper closes well before reaching (close to) the episode's own minimum
    #    achievable distance.
    if close_dist > max(min_dist * 1.5, min_dist + 0.02) and close_dist > CLOSE_THRESHOLD_M:
        triggers["premature_close"] = {"close_dist_m": close_dist, "min_dist_m": min_dist, "threshold": "close_dist > 1.5x min_dist (or +2cm) and > close_threshold 0.03m"}

    # 5. delayed close: minimum distance reached many steps before close, with the gap large.
    step_gap = close_ix - min_step
    if step_gap > 5 and (close_dist - min_dist) > 0.005:
        triggers["delayed_close"] = {"steps_between_min_and_close": step_gap, "min_dist_m": min_dist, "close_dist_m": close_dist, "threshold": "gap>5 steps and close_dist-min_dist>0.5cm"}

    # 6. insufficient approach: `grounded_grasp_close_threshold_m` (0.03) is calibrated against
    #    EE<->PREDICTED-target distance (the model's own close-decision variable, see
    #    `modeling_safediff_vla.py`'s `_maybe_enter_local_grasp`/`_select_action_local_grasp`), NOT
    #    EE<->GT-object-CENTER distance -- comparing GT distance to that threshold conflates
    #    grounding/localization accuracy (already measured separately) with trajectory shape, and
    #    fires on almost every episode regardless of how the local approach actually looked (a
    #    grasp can succeed with the EE origin several cm from the object's mesh-origin "center" --
    #    finger geometry, not misapproach). Ground this in the actual outcome instead: the episode
    #    never made contact at all.
    if not any(traj["is_grasped"]):
        triggers["insufficient_approach"] = {"min_dist_m": min_dist, "note": "never achieved is_grasped=True in this window"}

    # 7. post-contact instability: TCP keeps moving a lot after the grasp/close.
    post_positions = np.asarray(traj["actual_TCP_xyz"])[close_local_pos:]
    if len(post_positions) >= 2:
        post_disp = float(np.linalg.norm(np.diff(post_positions, axis=0), axis=1).sum())
        if post_disp > 0.05:
            triggers["post_contact_instability"] = {"total_post_close_displacement_m": post_disp, "threshold": "> 0.05m summed over up to 5 post-close steps"}

    if not triggers:
        triggers["no_clear_local_failure"] = {"min_dist_m": min_dist, "close_dist_m": close_dist}

    return {
        "triggered_modes": list(triggers.keys()),
        "details": triggers,
        "min_dist_m": min_dist, "min_dist_step": min_step, "close_dist_m": close_dist, "close_step": close_ix,
    }


def project(matrix_3x4: list, point_xyz: list | None) -> tuple[float, float] | None:
    if point_xyz is None:
        return None
    m = np.asarray(matrix_3x4, dtype=float)
    x = m @ np.array([*point_xyz, 1.0])
    if abs(x[2]) < 1e-9:
        return None
    return float(x[0] / x[2]), float(x[1] / x[2])


def plot_3d_trajectory(traj: dict, task: str, seed: int, out_path: Path) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    gt = np.asarray(traj["GT_center_xyz"])
    pred = np.asarray([p for p in traj["predicted_target_xyz"] if p is not None])
    cmd = np.asarray([p for p in traj["commanded_EE_xyz"] if p is not None])
    tcp = np.asarray(traj["actual_TCP_xyz"])

    ax.scatter(*gt[-1], color="black", s=120, marker="*", label="GT object center")
    if len(pred):
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], "o-", color="orange", markersize=3, label="predicted_target path")
    if len(cmd):
        ax.plot(cmd[:, 0], cmd[:, 1], cmd[:, 2], "^--", color="green", markersize=3, label="commanded_EE path")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], "s-", color="blue", markersize=3, label="actual_TCP path")

    step_ix = traj["step_ix"]
    entry_local = 0
    close_local = step_ix.index(traj["close_ix"])
    ax.scatter(*tcp[entry_local], color="purple", s=100, marker="D", label="LOCAL_GRASP entry")
    ax.scatter(*tcp[close_local], color="red", s=100, marker="X", label=f"close ({traj['close_rule']})")
    if traj["grasp_ix"] is not None and traj["grasp_ix"] in step_ix:
        gi = step_ix.index(traj["grasp_ix"])
        ax.scatter(*tcp[gi], color="cyan", s=80, marker="P", label="first is_grasped")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{task} seed={seed}: 3D trajectory (LOCAL_GRASP entry -> close)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_image_overlay(traj: dict, task: str, seed: int, camera: dict, video_path: Path, out_path: Path) -> None:
    close_ix = traj["close_ix"]
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, close_ix)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        logger.warning("could not read frame %d from %s -- skipping overlay for %s seed=%d", close_ix, video_path, task, seed)
        return
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    matrix = camera["matrix_3x4"]
    gt_uv = project(matrix, traj["GT_center_xyz"][-1])
    pred_uv = [project(matrix, p) for p in traj["predicted_target_xyz"]]
    cmd_uv = [project(matrix, p) for p in traj["commanded_EE_xyz"]]
    tcp_uv = [project(matrix, p) for p in traj["actual_TCP_xyz"]]
    close_local = traj["step_ix"].index(close_ix)

    def xy(pts):
        valid = [p for p in pts if p is not None]
        return ([p[0] for p in valid], [p[1] for p in valid]) if valid else ([], [])

    fig, ax = plt.subplots(figsize=(6, 6))
    h, w = frame_rgb.shape[:2]
    ax.imshow(frame_rgb, extent=(0, w, h, 0))
    px, py = xy(pred_uv)
    ax.plot(px, py, "o-", color="orange", markersize=3, linewidth=1, label="predicted_target path")
    cx, cy = xy(cmd_uv)
    ax.plot(cx, cy, "^--", color="lime", markersize=3, linewidth=1, label="commanded_EE path")
    tx, ty = xy(tcp_uv)
    ax.plot(tx, ty, "s-", color="deepskyblue", markersize=3, linewidth=1, label="actual_TCP path")
    if gt_uv:
        ax.scatter(*gt_uv, color="white", edgecolor="black", s=140, marker="*", label="GT center", zorder=5)
    if tcp_uv[close_local] is not None:
        ax.scatter(*tcp_uv[close_local], color="red", s=100, marker="X", label="close point", zorder=5)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"{task} seed={seed}: image-plane trajectory overlay", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    camera_lookup: dict[tuple[str, int], dict] = {}
    for task in STAGE2_TASK_NAMES:
        cam_path = CAMERA_DIR / f"{task}.json"
        for rec in json.loads(cam_path.read_text()):
            camera_lookup[(task, rec["seed"])] = rec["camera"]

    all_episode_reports: list[dict] = []
    for task in STAGE2_TASK_NAMES:
        raw_path = EPISODES_DIR / task / "episodes_raw.json"
        episode_records = json.loads(raw_path.read_text())
        for ep in episode_records:
            seed = ep["seed"]
            traj = build_episode_trajectory(ep["steps"])
            episode_out = OUT_DIR / task / f"seed{seed}"
            if traj is None:
                logger.warning("[%s] seed=%d: no LOCAL_GRASP entry -- classifying insufficient_approach, no plots", task, seed)
                all_episode_reports.append({
                    "task": task, "seed": seed, "episode_ix": ep["episode_ix"],
                    "classification": {"triggered_modes": ["insufficient_approach"], "details": {"note": "never entered LOCAL_GRASP"}},
                    "last_n_geometry": None, "trajectory_summary": None,
                })
                continue

            full_geom = compute_full_window_geometry(traj)
            last_n = compute_last_n_geometry(traj, full_geom)
            classification = classify_failure(traj, last_n)

            plot_3d_trajectory(traj, task, seed, episode_out / "trajectory_3d.png")
            camera = camera_lookup.get((task, seed))
            video_path = EPISODES_DIR / task / "videos" / f"eval_episode_{ep['episode_ix']}.mp4"
            if camera is not None and video_path.exists():
                plot_image_overlay(traj, task, seed, camera, video_path, episode_out / "image_overlay.png")
            else:
                logger.warning("[%s] seed=%d: missing camera matrix or video -- skipping image overlay", task, seed)

            radial = np.asarray(full_geom["radial_signed_dist_m"])
            radial_diffs = np.diff(radial)
            monotonic_convergence = {
                "fraction_steps_radial_decreasing": float((radial_diffs < 0).mean()) if len(radial_diffs) else None,
                "radial_at_entry_m": float(radial[0]),
                "radial_at_close_m": float(radial[np.where(np.asarray(traj["step_ix"]) == traj["close_ix"])[0][0]]),
                "overshot_past_object_line": bool((radial < 0).any()),
            }

            episode_out.mkdir(parents=True, exist_ok=True)
            (episode_out / "trajectory.json").write_text(json.dumps(traj, indent=2))
            (episode_out / "full_window_geometry.json").write_text(json.dumps(full_geom, indent=2))
            (episode_out / "last_n_geometry.json").write_text(json.dumps(last_n, indent=2))
            (episode_out / "classification.json").write_text(json.dumps(classification, indent=2))

            logger.info(
                "[%s] seed=%d: entry=%d close=%d(%s) min_dist=%.3fm@%d close_dist=%.3fm -> %s",
                task, seed, traj["entry_ix"], traj["close_ix"], traj["close_rule"],
                classification["min_dist_m"], classification["min_dist_step"], classification["close_dist_m"],
                classification["triggered_modes"],
            )
            all_episode_reports.append({
                "task": task, "seed": seed, "episode_ix": ep["episode_ix"],
                "classification": classification, "last_n_geometry": last_n,
                "monotonic_convergence": monotonic_convergence,
                "trajectory_summary": {"entry_ix": traj["entry_ix"], "close_ix": traj["close_ix"], "close_rule": traj["close_rule"], "n_steps": len(traj["step_ix"])},
            })

    # aggregate: how often does each failure mode recur, per task and overall
    mode_counts_per_task: dict[str, dict[str, int]] = {t: dict.fromkeys(FAILURE_MODES, 0) for t in STAGE2_TASK_NAMES}
    mode_counts_overall: dict[str, int] = dict.fromkeys(FAILURE_MODES, 0)
    for r in all_episode_reports:
        for mode in r["classification"]["triggered_modes"]:
            mode_counts_per_task[r["task"]][mode] += 1
            mode_counts_overall[mode] += 1

    report = {
        "n_episodes": len(all_episode_reports),
        "per_episode": all_episode_reports,
        "failure_mode_counts_per_task": mode_counts_per_task,
        "failure_mode_counts_overall": mode_counts_overall,
    }
    (OUT_DIR / "trajectory_shape_report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", OUT_DIR / "trajectory_shape_report.json")

    logger.info("=== failure mode counts per task (n episodes per task in parens) ===")
    for task in STAGE2_TASK_NAMES:
        n_task = sum(1 for r in all_episode_reports if r["task"] == task)
        counts = mode_counts_per_task[task]
        active = {k: v for k, v in counts.items() if v > 0}
        logger.info("%-14s (n=%d): %s", task, n_task, active)
    logger.info("=== OVERALL (n=%d episodes) ===", len(all_episode_reports))
    for mode, c in mode_counts_overall.items():
        if c > 0:
            logger.info("  %-30s %d/%d episodes", mode, c, len(all_episode_reports))


if __name__ == "__main__":
    main()
