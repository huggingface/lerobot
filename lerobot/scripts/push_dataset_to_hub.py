"""
Use this script to convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub,
or store it locally. LeRobot dataset format is lightweight, fast to load from, and does not require any
installation of neural net specific packages like pytorch, tensorflow, jax.

Example:
```
python lerobot/scripts/push_dataset_to_hub.py \
--data-dir data \
--dataset-id pusht \
--raw-format pusht_zarr \
--community-id lerobot \
--revision v1.2 \
--dry-run 1 \
--save-to-disk 1 \
--save-tests-to-disk 0 \
--debug 1

python lerobot/scripts/push_dataset_to_hub.py \
--data-dir data \
--dataset-id xarm_lift_medium \
--raw-format xarm_pkl \
--community-id lerobot \
--revision v1.2 \
--dry-run 1 \
--save-to-disk 1 \
--save-tests-to-disk 0 \
--debug 1

python lerobot/scripts/push_dataset_to_hub.py \
--data-dir data \
--dataset-id aloha_sim_insertion_scripted \
--raw-format aloha_hdf5 \
--community-id lerobot \
--revision v1.2 \
--dry-run 1 \
--save-to-disk 1 \
--save-tests-to-disk 0 \
--debug 1

python lerobot/scripts/push_dataset_to_hub.py \
--data-dir data \
--dataset-id umi_cup_in_the_wild \
--raw-format umi_zarr \
--community-id lerobot \
--revision v1.2 \
--dry-run 1 \
--save-to-disk 1 \
--save-tests-to-disk 0 \
--debug 1
```
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
from lerobot.common.datasets.push_dataset_to_hub.compute_stats import compute_stats
from lerobot.common.datasets.utils import flatten_dict


def get_from_raw_to_lerobot_format_fn(raw_format):
    if raw_format == "pusht_zarr":
        from lerobot.common.datasets.push_dataset_to_hub.pusht_zarr_format import from_raw_to_lerobot_format
    elif raw_format == "umi_zarr":
        from lerobot.common.datasets.push_dataset_to_hub.umi_zarr_format import from_raw_to_lerobot_format
    elif raw_format == "aloha_hdf5":
        from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import from_raw_to_lerobot_format
    elif raw_format == "xarm_pkl":
        from lerobot.common.datasets.push_dataset_to_hub.xarm_pkl_format import from_raw_to_lerobot_format
    else:
        raise ValueError(raw_format)

    return from_raw_to_lerobot_format


def save_meta_data(info, stats, episode_data_index, meta_data_dir):
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # save info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)

    # save stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    # save episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)


def push_meta_data_to_hub(repo_id, meta_data_dir, revision):
    """Expect all meta data files to be all stored in a single "meta_data" directory.
    On the hugging face repositery, they will be uploaded in a "meta_data" directory at the root.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=meta_data_dir,
        path_in_repo="meta_data",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
    )


def push_videos_to_hub(repo_id, videos_dir, revision):
    """Expect mp4 files to be all stored in a single "videos" directory.
    On the hugging face repositery, they will be uploaded in a "videos" directory at the root.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=videos_dir,
        path_in_repo="videos",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns="*.mp4",
    )


def push_dataset_to_hub(
    data_dir: Path,
    dataset_id: str,
    raw_format: str | None,
    community_id: str,
    revision: str,
    dry_run: bool,
    save_to_disk: bool,
    tests_data_dir: Path,
    save_tests_to_disk: bool,
    fps: int | None,
    video: bool,
    batch_size: int,
    num_workers: int,
    debug: bool,
):
    repo_id = f"{community_id}/{dataset_id}"

    raw_dir = data_dir / f"{dataset_id}_raw"

    out_dir = data_dir / repo_id
    meta_data_dir = out_dir / "meta_data"
    videos_dir = out_dir / "videos"

    tests_out_dir = tests_data_dir / repo_id
    tests_meta_data_dir = tests_out_dir / "meta_data"
    tests_videos_dir = tests_out_dir / "videos"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    if tests_out_dir.exists() and save_tests_to_disk:
        shutil.rmtree(tests_out_dir)

    if not raw_dir.exists():
        download_raw(raw_dir, dataset_id)

    if raw_format is None:
        # TODO(rcadene, adilzouitine): implement auto_find_raw_format
        raise NotImplementedError()
        # raw_format = auto_find_raw_format(raw_dir)

    from_raw_to_lerobot_format = get_from_raw_to_lerobot_format_fn(raw_format)

    # convert dataset from original raw format to LeRobot format
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(raw_dir, out_dir, fps, video, debug)

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        version=revision,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    if save_to_disk:
        hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
        hf_dataset.save_to_disk(str(out_dir / "train"))

    if not dry_run or save_to_disk:
        # mandatory for upload
        save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if not dry_run:
        hf_dataset.push_to_hub(repo_id, token=True, revision="main")
        hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision=revision)

        if video:
            push_videos_to_hub(repo_id, videos_dir, revision="main")
            push_videos_to_hub(repo_id, videos_dir, revision=revision)

    if save_tests_to_disk:
        # get the first episode
        num_items_first_ep = episode_data_index["to"][0] - episode_data_index["from"][0]
        test_hf_dataset = hf_dataset.select(range(num_items_first_ep))

        test_hf_dataset = test_hf_dataset.with_format(None)
        test_hf_dataset.save_to_disk(str(tests_out_dir / "train"))

        # copy meta data to tests directory
        shutil.copytree(meta_data_dir, tests_meta_data_dir)

        # copy videos of first episode to tests directory
        episode_index = 0
        tests_videos_dir.mkdir(parents=True, exist_ok=True)
        for key in lerobot_dataset.video_frame_keys:
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            shutil.copy(videos_dir / fname, tests_videos_dir / fname)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing datasets (e.g. `data` or `tmp/data` or `/tmp/lerobot/data`).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Name of the dataset (e.g. `pusht`, `aloha_sim_insertion_human`), which matches the folder where the data is stored (e.g. `data/pusht`).",
    )
    parser.add_argument(
        "--raw-format",
        type=str,
        help="Dataset type (e.g. `pusht_zarr`, `umi_zarr`, `aloha_hdf5`, `xarm_pkl`). If not provided, will be detected automatically.",
    )
    parser.add_argument(
        "--community-id",
        type=str,
        default="lerobot",
        help="Community or user ID under which the dataset will be hosted on the Hub.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=CODEBASE_VERSION,
        help="Codebase version used to generate the dataset.",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="Run everything without uploading to hub, for testing purposes or storing a dataset locally.",
    )
    parser.add_argument(
        "--save-to-disk",
        type=int,
        default=1,
        help="Save the dataset in the directory specified by `--data-dir`.",
    )
    parser.add_argument(
        "--tests-data-dir",
        type=Path,
        default="tests/data",
        help="Directory containing tests artifacts datasets.",
    )
    parser.add_argument(
        "--save-tests-to-disk",
        type=int,
        default=1,
        help="Save the dataset with 1 episode used for unit tests in the directory specified by `--tests-data-dir`.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frame rate used to collect videos. If not provided, use the default one specified in the code.",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=1,
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Debug mode process the first episode only.",
    )

    args = parser.parse_args()
    push_dataset_to_hub(**vars(args))


if __name__ == "__main__":
    main()
