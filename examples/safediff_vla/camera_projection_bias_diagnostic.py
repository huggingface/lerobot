#!/usr/bin/env python
"""Follow-up to `grasp_point_bias_diagnostic.py`: that script found NO consistent same-direction
world-frame (x/y) bias across the 6 tasks. But a world-frame direction can look like a CONSISTENT
image-plane direction once projected through each task's own differently-posed "forward" camera
(same key the policy's own "image" observation uses -- see `camera_mapping_diagnostic.py`'s
established fact that render()==policy input). This script re-projects the SAME 4 world points
`grasp_point_bias_diagnostic.py` already computed (`GT_object_center_xyz`, `predicted_target_xyz`,
`commanded_EE_xyz`, `actual_TCP_xyz`, read back from its report -- NOT recomputed) into that
camera's own image plane, and checks whether the SIGNED pixel-space residual (`du = u_X - u_gt`)
is consistent in direction across tasks/episodes -- the actual form of the "always drifts left in
the video" observation.

Needs a FRESH env reset per (task, seed) ONLY to read the camera's pose/FOV via dm_control's own
`engine.Camera` (the standard 3x4 camera-matrix projection utility -- see
https://en.wikipedia.org/wiki/Camera_matrix -- built from `physics.data.cam_xpos`/`cam_xmat`/
`model.cam_fovy`, i.e. the REALIZED world-frame camera pose, not the MJCF-authored one, so this is
correct even if the camera is nested under a moved body). NO policy rollout: the "forward"/"right"/
"wrist" cameras are scene-mounted (not attached to the robot -- confirmed by
`camera_mapping_diagnostic.py`'s report showing the SAME resolved camera name/order for the entire
7-task/all-seed sweep), so their pose is fixed for the whole episode and one `reset(seed=...)` is
enough to read it. Every (task, seed) reset runs in its own subprocess (`--only-task`), same
VLABench/MuJoCo cross-env-construction state-leak workaround `run_multitask_eval_isolated.py`
established.

NO architecture/weights/threshold/loss touched -- this reads physics/camera state only.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/camera_projection_bias_diagnostic.py --only-task select_poker
    uv run python examples/safediff_vla/camera_projection_bias_diagnostic.py --merge-only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

BIAS_REPORT_PATH = Path("outputs/eval/safediff_vla_grounded_grasp_bias_diagnostic/grasp_point_bias_report.json")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_camera_projection_bias_diagnostic")
IMAGE_HEIGHT = 480  # VLABenchEnv's default render_resolution -- matches every prior diagnostic capture
IMAGE_WIDTH = 480
CAMERA_NAME = "forward"  # resolves to the policy's own "image" observation key -- see module docstring


def get_camera_matrix(physics, camera_id: str, height: int, width: int) -> dict:
    from dm_control.mujoco.engine import Camera

    cam = Camera(physics, height=height, width=width, camera_id=camera_id)
    image, focal, rotation, translation = cam.matrices()
    matrix = image @ focal @ rotation @ translation
    fixedcamid = cam._render_camera.fixedcamid  # noqa: SLF001 -- dm_control has no public accessor for this
    return {
        "matrix_3x4": matrix.tolist(),
        "cam_xpos_world": physics.data.cam_xpos[fixedcamid].tolist(),
        "cam_xmat_world_flat": physics.data.cam_xmat[fixedcamid].tolist(),
        "cam_fovy_deg": float(physics.model.cam_fovy[fixedcamid]),
        "height": height,
        "width": width,
    }


def project(matrix_3x4: list, point_xyz: list | None) -> list[float] | None:
    if point_xyz is None:
        return None
    m = np.asarray(matrix_3x4, dtype=float)
    p_h = np.array([*point_xyz, 1.0])
    x = m @ p_h
    if abs(x[2]) < 1e-9:
        return None
    return [float(x[0] / x[2]), float(x[1] / x[2])]


def capture_camera_for_task_seed(task: str, seed: int) -> dict:
    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs[task][0]
    captured = {}
    try:
        from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl

        orig_reset = VLABenchEnvImpl.reset

        def patched_reset(self, seed=None, **kwargs):
            result = orig_reset(self, seed=seed, **kwargs)
            if seed is not None and "camera" not in captured:
                captured["camera"] = get_camera_matrix(self._env.physics, CAMERA_NAME, IMAGE_HEIGHT, IMAGE_WIDTH)
            return result

        VLABenchEnvImpl.reset = patched_reset
        try:
            vec_env.reset(seed=[seed])
        finally:
            VLABenchEnvImpl.reset = orig_reset
    finally:
        vec_env.close()
    return captured["camera"]


def run_task(task: str, bias_report: dict, out_dir: Path) -> None:
    episodes = [e for e in bias_report["per_episode"] if e["task"] == task]
    records = []
    for ep in episodes:
        seed = ep["seed"]
        logger.info("=== [%s] seed=%d: capturing camera ===", task, seed)
        camera = capture_camera_for_task_seed(task, seed)
        matrix = camera["matrix_3x4"]

        u_gt, v_gt = project(matrix, ep["GT_object_center_xyz"])
        pred_uv = project(matrix, ep.get("predicted_target_xyz"))
        cmd_uv = project(matrix, ep.get("commanded_EE_xyz"))
        tcp_uv = project(matrix, ep.get("actual_TCP_xyz"))

        def resid(uv):
            return [uv[0] - u_gt, uv[1] - v_gt] if uv is not None else None

        records.append({
            "task": task, "episode_ix": ep["episode_ix"], "seed": seed,
            "camera": camera,
            "GT_center_uv": [u_gt, v_gt],
            "predicted_target_uv": pred_uv,
            "commanded_EE_uv": cmd_uv,
            "actual_TCP_uv": tcp_uv,
            "d_predicted_target": resid(pred_uv),
            "d_commanded_EE": resid(cmd_uv),
            "d_actual_TCP": resid(tcp_uv),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{task}.json").write_text(json.dumps(records, indent=2))
    logger.info("wrote %s", out_dir / f"{task}.json")


def summarize(name: str, du_list: list[float], dv_list: list[float]) -> dict:
    du, dv = np.asarray(du_list), np.asarray(dv_list)
    n = len(du)
    if n == 0:
        return {"n": 0}
    n_du_pos, n_du_neg = int((du > 0).sum()), int((du < 0).sum())
    return {
        "n": n,
        "mean_du_px": float(du.mean()), "std_du_px": float(du.std()),
        "mean_dv_px": float(dv.mean()), "std_dv_px": float(dv.std()),
        "mean_normalized_du": float(du.mean() / IMAGE_WIDTH), "mean_normalized_dv": float(dv.mean() / IMAGE_HEIGHT),
        "du_sign_consistency": f"{n_du_pos}+/{n_du_neg}-/{n - n_du_pos - n_du_neg}=0 of {n}",
        "du_mean_over_std": float(du.mean() / du.std()) if du.std() > 1e-9 else None,
    }


def merge_and_report(out_dir: Path) -> None:
    all_records = []
    for task in STAGE2_TASK_NAMES:
        path = out_dir / f"{task}.json"
        if not path.exists():
            logger.warning("missing %s -- run --only-task %s first", path, task)
            continue
        all_records.extend(json.loads(path.read_text()))

    per_task_summary = {}
    for task in STAGE2_TASK_NAMES:
        task_records = [r for r in all_records if r["task"] == task]
        per_task_summary[task] = {
            key: summarize(
                key,
                [r[f"d_{key}"][0] for r in task_records if r.get(f"d_{key}") is not None],
                [r[f"d_{key}"][1] for r in task_records if r.get(f"d_{key}") is not None],
            )
            for key in ("predicted_target", "commanded_EE", "actual_TCP")
        }

    overall_summary = {
        key: summarize(
            key,
            [r[f"d_{key}"][0] for r in all_records if r.get(f"d_{key}") is not None],
            [r[f"d_{key}"][1] for r in all_records if r.get(f"d_{key}") is not None],
        )
        for key in ("predicted_target", "commanded_EE", "actual_TCP")
    }

    report = {"image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT, "camera_name": CAMERA_NAME,
              "per_episode": all_records, "per_task_summary": per_task_summary, "overall_summary": overall_summary}
    (out_dir / "camera_projection_bias_report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out_dir / "camera_projection_bias_report.json")

    logger.info("=== per-task image-plane signed residual (pixels, +u=right, +v=down) ===")
    for task, s in per_task_summary.items():
        for key in ("predicted_target", "commanded_EE", "actual_TCP"):
            v = s[key]
            if v["n"] == 0:
                continue
            logger.info(
                "%-14s %-18s n=%d du=%+.1fpx (norm=%+.4f, m/std=%s) dv=%+.1fpx (norm=%+.4f) sign=%s",
                task, key, v["n"], v["mean_du_px"], v["mean_normalized_du"],
                f"{v['du_mean_over_std']:+.2f}" if v["du_mean_over_std"] is not None else "n/a",
                v["mean_dv_px"], v["mean_normalized_dv"], v["du_sign_consistency"],
            )
    logger.info("=== OVERALL (pooled) ===")
    for key in ("predicted_target", "commanded_EE", "actual_TCP"):
        v = overall_summary[key]
        logger.info(
            "%-18s n=%d du=%+.1fpx (norm=%+.4f, m/std=%s) dv=%+.1fpx (norm=%+.4f) sign=%s",
            key, v["n"], v["mean_du_px"], v["mean_normalized_du"],
            f"{v['du_mean_over_std']:+.2f}" if v["du_mean_over_std"] is not None else "n/a",
            v["mean_dv_px"], v["mean_normalized_dv"], v["du_sign_consistency"],
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only-task", default=None, choices=STAGE2_TASK_NAMES)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.merge_only:
        merge_and_report(out_dir)
        return

    bias_report = json.loads(BIAS_REPORT_PATH.read_text())
    tasks = [args.only_task] if args.only_task else STAGE2_TASK_NAMES
    for task in tasks:
        run_task(task, bias_report, out_dir)


if __name__ == "__main__":
    main()
