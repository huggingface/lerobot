#!/usr/bin/env python
"""Quantify LOCAL_GRASP -> POST_GRASP handoff discontinuity from `episodes_raw.json` produced by
`rollout_with_handoff_instrumentation.py`. Pure post-hoc analysis of existing rollout logs -- no
env stepping, no policy inference, no training.

For each episode, finds the LOCAL_GRASP->POST_GRASP transition (`just_transitioned_post_grasp`)
and computes, around that step:
    delta_xyz_handoff        = || first_POST_GRASP_commanded_xyz - last_LOCAL_GRASP_commanded_xyz ||
    delta_xyz_from_current   = || first_POST_GRASP_commanded_xyz - actual_TCP_at_transition ||
    delta_rot_handoff/_from_current -- same, but quaternion geodesic angle (NOT Euler difference;
        Euler wrap-around near +/-pi is not a physical discontinuity by itself)
    peak_post_tracking_error_m -- how far the ACTUAL ee_pos drifts from the freshly commanded
        target over the 3 steps after the replan
    peak_xyz_jump_m / peak_rot_jump_deg -- largest single-step ACTUAL motion in the +/-5 step
        window around the transition (physical jump, not just the planned target's jump)

"last LOCAL_GRASP commanded xyz/rotation" is derived (not a raw field) from the last LOCAL_GRASP
step's own `local_delta_xyz_m`/`predicted_delta_rot_rad` (each relative to THAT step's own
`ee_pos`/`actual_ee_euler_rad`). "first POST_GRASP commanded xyz/rotation" is the first step after
the transition where `commanded_action_xyz_world` is non-None (the global `TemporalActionDecoder`'s
forced replan immediately after LOCAL_GRASP hands off).

Usage:
    uv run python examples/safediff_vla/handoff_discontinuity_metrics.py \
        --base-dir outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_handoff \
        --out-json outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_handoff/handoff_discontinuity_analysis/handoff_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_TASKS = ["select_book", "select_drink", "select_toy", "select_mahjong", "select_poker", "select_fruit"]


def quat_wxyz_geodesic(q1, q2) -> float:
    """Geodesic angle (rad) between two [w,x,y,z] quaternions, sign/double-cover safe."""
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    q2 = np.asarray(q2, dtype=float) / np.linalg.norm(q2)
    d = min(1.0, max(-1.0, abs(np.dot(q1, q2))))
    return 2 * np.arccos(d)


def euler_to_quat_wxyz(euler_xyz) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return np.array([w, x, y, z])


def analyze_episode(task: str, ep: dict) -> dict | None:
    steps = ep["steps"]
    idx = {s["step_ix"]: s for s in steps}
    trans_ix = [s["step_ix"] for s in steps if s.get("just_transitioned_post_grasp")]
    if not trans_ix:
        return None
    T = trans_ix[0]
    sT = idx[T]

    T1 = None
    for k in range(T + 1, min(T + 5, len(steps))):
        if idx[k].get("commanded_action_xyz_world") is not None:
            T1 = k
            break
    result = {
        "task": task, "episode_ix": ep["episode_ix"], "seed": ep["seed"], "success": ep["success"],
        "transition_step": T, "first_post_replan_step": T1,
    }
    if T1 is None:
        result["error"] = "no replan found after transition"
        return result

    actual_tcp_at_transition = np.array(sT["ee_pos"])
    actual_quat_at_transition = np.array(sT["actual_ee_quat_wxyz"])

    last_local_commanded_xyz = np.array(sT["ee_pos"]) + np.array(sT["local_delta_xyz_m"])
    last_local_euler = np.array(sT["actual_ee_euler_rad"]) + np.array(sT["predicted_delta_rot_rad"])
    last_local_commanded_quat = euler_to_quat_wxyz(last_local_euler)

    sT1 = idx[T1]
    first_post_commanded_xyz = np.array(sT1["commanded_action_xyz_world"])
    first_post_commanded_quat = np.array(sT1["commanded_action_quat_wxyz"])

    delta_xyz_handoff = float(np.linalg.norm(first_post_commanded_xyz - last_local_commanded_xyz))
    delta_xyz_from_current = float(np.linalg.norm(first_post_commanded_xyz - actual_tcp_at_transition))
    delta_rot_handoff = quat_wxyz_geodesic(first_post_commanded_quat, last_local_commanded_quat)
    delta_rot_from_current = quat_wxyz_geodesic(first_post_commanded_quat, actual_quat_at_transition)

    window_lo, window_hi = max(0, T - 5), min(len(steps) - 1, T + 5)
    post_tracking_errors = [
        float(np.linalg.norm(np.array(idx[k]["ee_pos"]) - first_post_commanded_xyz))
        for k in range(T1, min(T1 + 3, len(steps)))
    ]

    peak_xyz_jump, peak_rot_jump = 0.0, 0.0
    for k in range(max(window_lo, T), min(T1 + 3, len(steps)) - 1):
        a, b = np.array(idx[k]["ee_pos"]), np.array(idx[k + 1]["ee_pos"])
        peak_xyz_jump = max(peak_xyz_jump, float(np.linalg.norm(b - a)))
        qa, qb = np.array(idx[k]["actual_ee_quat_wxyz"]), np.array(idx[k + 1]["actual_ee_quat_wxyz"])
        peak_rot_jump = max(peak_rot_jump, quat_wxyz_geodesic(qa, qb))

    result.update({
        "actual_tcp_at_transition": actual_tcp_at_transition.tolist(),
        "actual_quat_at_transition": actual_quat_at_transition.tolist(),
        "last_local_commanded_xyz": last_local_commanded_xyz.tolist(),
        "last_local_commanded_quat": last_local_commanded_quat.tolist(),
        "first_post_commanded_xyz": first_post_commanded_xyz.tolist(),
        "first_post_commanded_quat": first_post_commanded_quat.tolist(),
        "first_post_gripper": sT1["commanded_action_gripper_raw"],
        "delta_xyz_handoff": delta_xyz_handoff,
        "delta_xyz_from_current": delta_xyz_from_current,
        "delta_rot_handoff_deg": float(np.degrees(delta_rot_handoff)),
        "delta_rot_from_current_deg": float(np.degrees(delta_rot_from_current)),
        "peak_post_tracking_error_m": max(post_tracking_errors) if post_tracking_errors else None,
        "peak_xyz_jump_m": peak_xyz_jump,
        "peak_rot_jump_deg": float(np.degrees(peak_rot_jump)),
        "min_ee_to_target_dist_episode": min(s["ee_to_target_dist"] for s in steps),
        "final_is_grasped": steps[-1]["is_grasped"],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True, help="dir containing <task>/episodes_raw.json")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    base = Path(args.base_dir)

    all_results: dict[str, list] = {}
    for task in args.tasks:
        path = base / task / "episodes_raw.json"
        if not path.exists():
            continue
        eps = json.load(open(path))
        all_results[task] = [analyze_episode(task, ep) for ep in eps]

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(all_results, open(args.out_json, "w"), indent=2)

    for task, eps in all_results.items():
        for r in eps:
            if r is None:
                print(f"{task}: no transition found")
                continue
            if "error" in r:
                print(f"{task} ep{r['episode_ix']} seed{r['seed']}: {r['error']}")
                continue
            print(
                f"{task:15s} ep{r['episode_ix']} seed={r['seed']} success={str(r['success']):5} "
                f"T={r['transition_step']:3d} dxyz_handoff={r['delta_xyz_handoff']:.4f} "
                f"dxyz_cur={r['delta_xyz_from_current']:.4f} drot_handoff_deg={r['delta_rot_handoff_deg']:.2f} "
                f"drot_cur_deg={r['delta_rot_from_current_deg']:.2f} "
                f"peak_track_err={r['peak_post_tracking_error_m']:.4f} "
                f"peak_xyz_jump={r['peak_xyz_jump_m']:.4f} peak_rot_jump_deg={r['peak_rot_jump_deg']:.2f}"
            )


if __name__ == "__main__":
    main()
