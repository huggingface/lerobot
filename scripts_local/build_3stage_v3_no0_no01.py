"""Drop 0/6 (and 0+1/6) partial-fail eps from sim_3stage_v3_train_fs.

Outputs:
  - domrachev03/sim_3stage_v3_no0_train_fs   (drop 0/6 = ep 59-63)
  - domrachev03/sim_3stage_v3_no01_train_fs  (drop 0+1/6 = ep 59-68)

After: rebuild CLIP cache + temporal_proportions for each.
"""
from __future__ import annotations
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

SRC = "domrachev03/sim_3stage_v3_train_fs"

VARIANTS = [
    ("domrachev03/sim_3stage_v3_no0_train_fs", [59, 60, 61, 62, 66]),
    ("domrachev03/sim_3stage_v3_no01_train_fs", [59, 60, 61, 62, 63, 64, 65, 66, 67, 70]),
]

COLS = (
    "sparse_subtask_names", "sparse_subtask_start_frames", "sparse_subtask_end_frames",
    "dense_subtask_names", "dense_subtask_start_frames", "dense_subtask_end_frames",
)


def build_one(out_repo: str, drop_eps: list[int]) -> None:
    print(f"== {out_repo} (drop {drop_eps}) ==")
    out_root = HF_LEROBOT_HOME / out_repo
    if out_root.exists():
        shutil.rmtree(out_root)
    src = LeRobotDataset(repo_id=SRC)
    new_ds = delete_episodes(
        dataset=src,
        episode_indices=drop_eps,
        repo_id=out_repo,
        output_dir=out_root,
    )
    print(f"  out: {new_ds.num_episodes} eps {new_ds.num_frames} fr")

    # Patch subtask cols (delete_episodes drops them)
    src_meta_path = HF_LEROBOT_HOME / SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    dst_meta_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    src_meta = pq.read_table(src_meta_path).to_pandas().sort_values("episode_index").reset_index(drop=True)
    kept = src_meta[~src_meta["episode_index"].isin(drop_eps)].reset_index(drop=True)
    dst = pq.read_table(dst_meta_path).to_pandas().sort_values("episode_index").reset_index(drop=True)
    for col in COLS:
        if col in kept.columns:
            dst[col] = kept[col].values
    dst.to_parquet(dst_meta_path, index=False)
    print(f"  patched subtask cols: {len(dst)} rows")


def main() -> None:
    for repo, drop in VARIANTS:
        build_one(repo, drop)


if __name__ == "__main__":
    main()
