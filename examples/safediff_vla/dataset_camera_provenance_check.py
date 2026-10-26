#!/usr/bin/env python
"""Pure diagnostic (NO architecture/training changes): for each of the 7 B-track tasks, decode
frame 0 of ONE `lerobot/vlabench_unified` training episode's `image`/`second_image`/`wrist_image`
and save it next to the corresponding fresh-env-reset frame already saved by
`camera_mapping_diagnostic.py`, for a side-by-side visual role comparison.

This does NOT check pixel-exact match (scene layout/lighting differ episode to episode) -- it
checks CAMERA ROLE: does the dataset's own "image" for a given task show the same kind of view
(wide workspace-with-robot vs. tight container-interior close-up) that the live eval env's
"image" showed for that same task in `camera_mapping_diagnostic.py`'s report? If the dataset was
recorded with the SAME VLABench env code (same per-scene camera authoring), the two should match
in ROLE even though the exact pose/objects differ.

Usage:
    uv run python examples/safediff_vla/dataset_camera_provenance_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import REPO_ID, TASK_NAMES, build_episode_task_map  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/eval/safediff_vla_camera_mapping_diagnostic")
IMAGE_KEYS = ["image", "second_image", "wrist_image"]


def save_png(tensor, path: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np

    arr = tensor.permute(1, 2, 0).numpy().astype(np.uint8)  # CHW -> HWC
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr)


def main() -> None:
    logger.info("=== building episode -> task map ===")
    episode_task = build_episode_task_map()
    task_to_first_episode: dict[str, int] = {}
    for ep_idx in sorted(episode_task):
        t = episode_task[ep_idx]
        if t not in task_to_first_episode:
            task_to_first_episode[t] = ep_idx

    records = []
    for task in TASK_NAMES:
        ep_idx = task_to_first_episode[task]
        logger.info("=== [%s] dataset episode %d, frame 0 ===", task, ep_idx)
        ds = LeRobotDataset(REPO_ID, episodes=[ep_idx], return_uint8=True)
        item = ds[0]
        assert int(item["episode_index"]) == ep_idx
        assert int(item["frame_index"]) == 0

        task_dir = OUT_DIR / task
        saved = {}
        for key in IMAGE_KEYS:
            out_path = task_dir / f"dataset_{key}.png"
            save_png(item[f"observation.images.{key}"], out_path)
            saved[key] = str(out_path)

        records.append({"task": task, "dataset_episode_index": ep_idx, "saved_files": saved})

    (OUT_DIR / "dataset_camera_provenance_report.json").write_text(json.dumps(records, indent=2))
    logger.info("wrote %s", OUT_DIR / "dataset_camera_provenance_report.json")
    for r in records:
        logger.info("[%s] episode %d -> %s", r["task"], r["dataset_episode_index"], r["saved_files"])


if __name__ == "__main__":
    main()
