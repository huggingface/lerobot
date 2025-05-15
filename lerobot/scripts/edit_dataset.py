"""
Edit your dataset in-place.

Example of usage:
```bash
python lerobot/scripts/edit_dataset.py remove \
    --root data \
    --repo-id cadene/koch_bimanual_folding_2 \
    --episodes 0 4 7 10 34 54 69
```
"""

import argparse
import shutil
from pathlib import Path

import torch

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)


def remove_episodes(dataset, episodes):
    if not dataset.video:
        raise NotImplementedError()

    repo_id = dataset.repo_id
    info = dataset.info
    hf_dataset = dataset.hf_dataset
    # TODO(rcadene): implement tags
    # if None, should use the same tags
    tags = None

    local_dir = dataset.videos_dir.parent
    train_dir = local_dir / "train"
    new_train_dir = local_dir / "new_train"
    meta_data_dir = local_dir / "meta_data"

    new_hf_dataset = hf_dataset.filter(lambda row: row["episode_index"] not in episodes)

    unique_episode_idxs = torch.stack(new_hf_dataset["episode_index"]).unique().tolist()

    episode_idx_to_reset_idx_mapping = {}
    for new_ep_idx, ep_idx in enumerate(sorted(unique_episode_idxs)):
        episode_idx_to_reset_idx_mapping[ep_idx] = new_ep_idx

        for key in dataset.video_frame_keys:
            path = dataset.videos_dir / f"{key}_episode_{ep_idx:06d}.mp4"
            new_path = dataset.videos_dir / f"{key}_episode_{new_ep_idx:06d}.mp4"
            path.rename(new_path)

    def modify_ep_idx(row):
        new_ep_idx = episode_idx_to_reset_idx_mapping[row["episode_index"].item()]

        for key in dataset.video_frame_keys:
            fname = f"{key}_episode_{new_ep_idx:06d}.mp4"
            row[key]["path"] = f"videos/{fname}"

        row["episode_index"] = new_ep_idx
        return row

    new_hf_dataset = new_hf_dataset.map(modify_ep_idx)

    episode_data_index = calculate_episode_data_index(new_hf_dataset)

    new_dataset = LeRobotDataset.from_preloaded(
        repo_id=dataset.repo_id,
        hf_dataset=new_hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=dataset.videos_dir,
    )
    stats = compute_stats(new_dataset)

    new_hf_dataset = new_hf_dataset.with_format(None)  # to remove transforms that cant be saved

    new_hf_dataset.save_to_disk(str(new_train_dir))
    shutil.rmtree(train_dir)
    new_train_dir.rename(train_dir)

    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    new_hf_dataset.push_to_hub(repo_id, revision="main")
    push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
    push_dataset_card_to_hub(repo_id, revision="main", tags=tags)
    if dataset.video:
        push_videos_to_hub(repo_id, dataset.videos_dir, revision="main")
    create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    base_parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )

    remove_calib = subparsers.add_parser("remove", parents=[base_parser])
    remove_calib.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Episode indices to remove (e.g. `0 1 5 6`).",
    )

    args = parser.parse_args()

    input("It is recommended to make a copy of your dataset before modifying it. Press enter to continue.")

    dataset = LeRobotDataset(args.repo_id, root=args.root)

    if args.mode == "remove":
        remove_episodes(dataset, args.episodes)
