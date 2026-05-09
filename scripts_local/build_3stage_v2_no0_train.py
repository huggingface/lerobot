"""Drop 0/6 partial-fail eps (idx 119-127) from sim_3stage_v2_train_fs.

Rationale: 0/6 eps have no subtask annotations, GT progress=0 throughout. They
visually resemble the early frames of success eps; the model cannot distinguish.
Training on them leaks "stuck near box" → progress=0 vs "starting box approach"
→ progress=0.05, confusing the tau head.

Output: domrachev03/sim_3stage_v2_no0_train_fs (149 eps).
Then: rebuild CLIP cache + temporal_proportions.

Notes:
- We only drop from TRAIN set. The val_fs keeps the 9 0/6 eps so we can still
  see whether the model regresses on them (regression is acceptable since their
  visuals are ambiguous; we just don't want the model to overshoot to ~0.6).
"""
from __future__ import annotations
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

SRC = "domrachev03/sim_3stage_v2_train_fs"
OUT = "domrachev03/sim_3stage_v2_no0_train_fs"
DROP_EPS = list(range(119, 128))  # 9 eps: 119..127 (0/6 partials)


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
