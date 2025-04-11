"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.1 to
3.0. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

```bash
python lerobot/common/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=lerobot/pusht
```

"""

import argparse
import logging

from datasets import Dataset
from huggingface_hub import snapshot_download

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.utils import (
    load_episodes_stats,
)

V21 = "v2.1"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
):
    root = HF_LEROBOT_HOME / repo_id
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=V21,
        local_dir=root,
    )

    # Concatenate videos

    # Create

    """
    -------------------------
    OLD
    data/chunk-000/episode_000000.parquet

    NEW
    data/chunk-000/file_000.parquet
    -------------------------
    OLD
    videos/chunk-000/CAMERA/episode_000000.mp4

    NEW
    videos/chunk-000/file_000.mp4
    -------------------------
    OLD
    episodes.jsonl
    {"episode_index": 1, "tasks": ["Put the blue block in the green bowl"], "length": 266}

    NEW
    meta/episodes/chunk-000/episodes_000.parquet
    episode_index | video_chunk_index | video_file_index | data_chunk_index | data_file_index | tasks | length
    -------------------------
    OLD
    tasks.jsonl
    {"task_index": 1, "task": "Put the blue block in the green bowl"}

    NEW
    meta/tasks/chunk-000/file_000.parquet
    task_index | task
    -------------------------
    OLD
    episodes_stats.jsonl

    NEW
    meta/episodes_stats/chunk-000/file_000.parquet
    episode_index | mean | std | min | max
    -------------------------
    UPDATE
    meta/info.json
    -------------------------
    """

    new_root = HF_LEROBOT_HOME / f"{repo_id}_v30"
    new_root.mkdir(parents=True, exist_ok=True)

    episodes_stats = load_episodes_stats(root)
    hf_dataset = Dataset.from_dict(episodes_stats)  # noqa: F841

    meta_ep_st_ch = new_root / "meta/episodes_stats/chunk-000"
    meta_ep_st_ch.mkdir(parents=True, exist_ok=True)

    # hf_dataset.to_parquet(meta_ep_st_ch / 'file_000.parquet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
