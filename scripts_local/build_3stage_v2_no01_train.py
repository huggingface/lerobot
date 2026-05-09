"""Drop 0/6 + 1/6 partial-fail eps from sim_3stage_v2_train_fs.

In sim_3stage_v2_train_fs (158 eps):
  - 0/6 eps: idx 119-127 (9 eps)
  - 1/6 eps: idx 128-137 (10 eps)

Output: domrachev03/sim_3stage_v2_no01_train_fs (139 eps).

Tests whether 1/6 eps (visually similar to early success approach_box) also harm
the tau head. iter8 escalation if iter7 (drop 0/6 only) still fails.
"""
from __future__ import annotations
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

SRC = "domrachev03/sim_3stage_v2_train_fs"
OUT = "domrachev03/sim_3stage_v2_no01_train_fs"
DROP_EPS = list(range(119, 138))  # 19 eps: 119-127 (0/6) + 128-137 (1/6)


def main() -> None:
    out_root = HF_LEROBOT_HOME / OUT
    if out_root.exists():
        print(f"removing existing {out_root}")
        shutil.rmtree(out_root)

    src = LeRobotDataset(repo_id=SRC)
    print(f"src: {src.num_episodes} eps, {src.num_frames} frames")

    new_ds = delete_episodes(
        dataset=src,
        episode_indices=DROP_EPS,
        repo_id=OUT,
        output_dir=out_root,
    )
    print(f"out: {new_ds.num_episodes} eps, {new_ds.num_frames} frames @ {out_root}")


if __name__ == "__main__":
    main()
