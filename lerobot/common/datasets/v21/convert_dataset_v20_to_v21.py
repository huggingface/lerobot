"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.0 to
2.1. It performs the following:

- Generates per-episodes stats and writes them in `episodes_stats.jsonl`
- Removes the deprecated `stats.json` (by default)
- Updates codebase_version in `info.json`

Usage:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial
```

"""

import argparse

from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats, convert_stats


def main(
    repo_id: str,
    test_branch: str | None = None,
    delete_old_stats: bool = False,
    num_workers: int = 4,
):
    dataset = LeRobotDataset(repo_id)
    if (dataset.root / EPISODES_STATS_PATH).is_file():
        raise FileExistsError("episodes_stats.jsonl already exists.")

    convert_stats(dataset, num_workers=num_workers)
    ref_stats = load_stats(dataset.root)
    check_aggregate_stats(dataset, ref_stats)

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    dataset.push_to_hub(branch=test_branch, create_card=False, allow_patterns="meta/")

    if delete_old_stats:
        if (dataset.root / STATS_PATH).is_file:
            (dataset.root / STATS_PATH).unlink()
        hub_api = HfApi()
        if hub_api.file_exists(
            STATS_PATH, repo_id=dataset.repo_id, revision=test_branch, repo_type="dataset"
        ):
            hub_api.delete_file(
                STATS_PATH, repo_id=dataset.repo_id, revision=test_branch, repo_type="dataset"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--test-branch",
        type=str,
        default=None,
        help="Repo branch to test your conversion first (e.g. 'v2.0.test')",
    )
    parser.add_argument(
        "--delete-old-stats",
        type=bool,
        default=False,
        help="Delete the deprecated `stats.json`",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing compute",
    )

    args = parser.parse_args()
    main(**vars(args))
