"""Merge 100 success + 58 partials → sim_3stage_with_partials_v2.

Then frame-stride split (N=10) → train_fs / val_fs.

Sources:
  - domrachev03/sim_assemble_sarm_multistage_three_stages_success (100 eps)
  - domrachev03/sim_assemble_sarm_multistage_three_stages_failures (58 eps)

Outputs:
  - domrachev03/sim_3stage_with_partials_v2 (158 eps)
  - domrachev03/sim_3stage_v2_train_fs
  - domrachev03/sim_3stage_v2_val_fs
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

SRC_SUCCESS = "domrachev03/sim_assemble_sarm_multistage_three_stages_success"
SRC_PARTIALS = "domrachev03/sim_assemble_sarm_multistage_three_stages_failures"
OUT_FULL = "domrachev03/sim_3stage_with_partials_v2"
META_OUT = Path("outputs/sim_3stage_v2_merged_meta.json")


def main() -> None:
    full_root = HF_LEROBOT_HOME / OUT_FULL
    if full_root.exists():
        print(f"removing existing {full_root}")
        shutil.rmtree(full_root)

    ds_success = LeRobotDataset(repo_id=SRC_SUCCESS)
    ds_partials = LeRobotDataset(repo_id=SRC_PARTIALS)
    print(f"src_success: {ds_success.num_episodes} eps, {ds_success.num_frames} frames")
    print(f"src_partials: {ds_partials.num_episodes} eps, {ds_partials.num_frames} frames")

    merged = merge_datasets(
        datasets=[ds_success, ds_partials],
        output_repo_id=OUT_FULL,
    )
    print(f"merged: {merged.num_episodes} eps, {merged.num_frames} frames")

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps({
        "src_success": SRC_SUCCESS,
        "src_partials": SRC_PARTIALS,
        "n_success": ds_success.num_episodes,
        "n_partials": ds_partials.num_episodes,
        "out_full": OUT_FULL,
        "n_total": merged.num_episodes,
        "n_total_frames": merged.num_frames,
    }, indent=2))
    print(f"meta → {META_OUT}")


if __name__ == "__main__":
    main()
