#!/usr/bin/env python
"""Pure diagnostic (NO architecture/training/eval changes): for each of the 7 B-track tasks,
reset once and record exactly which MuJoCo camera backs each of the policy's own observation keys
("image"/"second_image"/"wrist_image") and what the saved eval video's frame source is.

Code-level fact established by reading `envs/vlabench.py` before running this (see
`VLABenchEnv._resolve_camera_order`/`._get_obs`/`.render`, and `lerobot_eval.py`'s
`render_frame` at `env.envs[i].render()`): the saved eval video and the policy's own "image"
observation are NOT two separate camera paths -- `VLABenchEnv.render()` literally returns
`self._get_obs()["pixels"]["image"]`, the exact same array the policy receives as its "image" key
(after `rename_map` renames it to `observation.images.camera1`). So whatever this script finds for
"image" per task IS both the policy input and the video source at once, by construction -- there is
no code path where the video could show a different camera than the policy sees. What CAN still
differ, and is what this script actually checks empirically: whether `_resolve_camera_order`'s
name-matching (looking for "forward"/"front"/"wrist"/"right" in `physics.model.camera(i).name`)
resolves to the SAME real camera position/framing convention across all 7 tasks' MuJoCo scenes, or
to a genuinely different camera per task (e.g. because a shelf/fridge-scene task's "front" camera
was authored at a different position than an open-workspace task's "front" camera).

For each task, saves under `--out-dir/<task>/`:
    image.png, second_image.png, wrist_image.png, saved_video_source.png (== image.png, expected
    to be byte-identical given the code fact above -- checked and logged, not assumed)
and appends one record to `camera_mapping_report.json` with: task, available camera names (in
MuJoCo index order), resolved camera index for each of image/second_image/wrist_image, and the
resolved camera NAME for each (so two tasks can be compared by camera identity, not just index).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/camera_mapping_diagnostic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import TASK_NAMES  # noqa: E402

from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SEED = 1000
OUT_DIR = Path("outputs/eval/safediff_vla_camera_mapping_diagnostic")


def save_png(arr: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr)


def diagnose_task(task: str, out_dir: Path) -> dict:
    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs[task][0]

    captured: dict = {}

    orig_reset = VLABenchEnvImpl.reset

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None and "obs" not in captured:
            ncam = self._env.physics.model.ncam
            names = [self._env.physics.model.camera(i).name for i in range(ncam)]
            captured["camera_names"] = names
            captured["camera_order"] = list(self._camera_order) if self._camera_order is not None else None
            obs, _info = result
            captured["obs"] = obs
            captured["render"] = self.render()
        return result

    VLABenchEnvImpl.reset = patched_reset
    try:
        vec_env.reset(seed=[SEED])
    finally:
        VLABenchEnvImpl.reset = orig_reset
        vec_env.close()

    camera_names = captured["camera_names"]
    camera_order = captured["camera_order"]  # [front_idx, second_idx, wrist_idx] or None
    pixels = captured["obs"]["pixels"]
    render_frame = captured["render"]

    task_dir = out_dir / task
    save_png(pixels["image"], task_dir / "image.png")
    save_png(pixels["second_image"], task_dir / "second_image.png")
    save_png(pixels["wrist_image"], task_dir / "wrist_image.png")
    save_png(render_frame, task_dir / "saved_video_source.png")

    identical_to_image = bool(np.array_equal(render_frame, pixels["image"]))

    def cam_name(idx: int | None) -> str | None:
        return camera_names[idx] if idx is not None and idx < len(camera_names) else None

    record = {
        "task": task,
        "available_camera_names_by_mujoco_index": camera_names,
        "n_cameras": len(camera_names),
        "resolved_camera_order_indices": camera_order,  # [image_idx, second_image_idx, wrist_image_idx]
        "image_camera": {"index": camera_order[0] if camera_order else None, "name": cam_name(camera_order[0] if camera_order else None)},
        "second_image_camera": {"index": camera_order[1] if camera_order else None, "name": cam_name(camera_order[1] if camera_order else None)},
        "wrist_image_camera": {"index": camera_order[2] if camera_order else None, "name": cam_name(camera_order[2] if camera_order else None)},
        "video_frame_source": "VLABenchEnv.render() -> obs['pixels']['image'] (same array object the policy receives as its own 'image' observation key, see module docstring)",
        "video_frame_identical_to_image_obs": identical_to_image,
        "saved_files": {
            "image": str(task_dir / "image.png"),
            "second_image": str(task_dir / "second_image.png"),
            "wrist_image": str(task_dir / "wrist_image.png"),
            "saved_video_source": str(task_dir / "saved_video_source.png"),
        },
    }
    logger.info(
        "[%s] n_cameras=%d names=%s -> image=%s second_image=%s wrist_image=%s (video==image: %s)",
        task, len(camera_names), camera_names,
        record["image_camera"]["name"], record["second_image_camera"]["name"], record["wrist_image_camera"]["name"],
        identical_to_image,
    )
    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    report_path = out_dir / "camera_mapping_report.json"
    existing: dict[str, dict] = {}
    if report_path.exists():
        existing = {r["task"]: r for r in json.loads(report_path.read_text())}

    for task in args.tasks:
        logger.info("=== diagnosing task=%s ===", task)
        existing[task] = diagnose_task(task, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = [existing[t] for t in TASK_NAMES if t in existing]
    report_path.write_text(json.dumps(records, indent=2))
    logger.info("wrote %s", out_dir / "camera_mapping_report.json")

    # Cross-task consistency check: same camera NAME used for "image" across every task?
    image_cam_names = {r["task"]: r["image_camera"]["name"] for r in records}
    unique_names = set(image_cam_names.values())
    logger.info("=== 'image' camera name by task: %s ===", image_cam_names)
    if len(unique_names) > 1:
        logger.warning(
            "INCONSISTENT: 'image' resolves to %d different camera names across tasks: %s",
            len(unique_names), unique_names,
        )
    else:
        logger.info("CONSISTENT: every task's 'image' resolves to the same camera name: %s", unique_names)


if __name__ == "__main__":
    main()
