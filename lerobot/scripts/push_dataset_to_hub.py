import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Protocol

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
from lerobot.common.datasets.push_dataset_to_hub.aloha_processor import (
    AlohaProcessor,
)
from lerobot.common.datasets.push_dataset_to_hub.pusht_processor import PushTProcessor
from lerobot.common.datasets.push_dataset_to_hub.umi_processor import UmiProcessor
from lerobot.common.datasets.push_dataset_to_hub.xarm_processor import XarmProcessor
from lerobot.common.datasets.utils import compute_stats, flatten_dict


def push_lerobot_dataset_to_hub(
    hf_dataset: Dataset,
    episode_data_index: dict[str, list[int]],
    info: dict[str, Any],
    stats: dict[str, dict[str, torch.Tensor]],
    root: Path,
    revision: str,
    dataset_id: str,
    community_id: str = "lerobot",
    dry_run: bool = False,
) -> None:
    """
    Pushes a dataset to the Hugging Face Hub.

    Args:
        hf_dataset (Dataset): The dataset to be pushed.
        episode_data_index (dict[str, list[int]]): The index of episode data.
        info (dict[str, Any]): Information about the dataset, eg. fps.
        stats (dict[str, dict[str, torch.Tensor]]): Statistics of the dataset.
        root (Path): The root directory of the dataset.
        revision (str): The revision of the dataset.
        dataset_id (str): The ID of the dataset.
        community_id (str, optional): The ID of the community or the user where the
            dataset will be stored. Defaults to "lerobot".
        dry_run (bool, optional): If True, performs a dry run without actually pushing the dataset. Defaults to False.
    """
    if not dry_run:
        # push to main to indicate latest version
        hf_dataset.push_to_hub(f"{community_id}/{dataset_id}", token=True)

        # push to version branch
        hf_dataset.push_to_hub(f"{community_id}/{dataset_id}", token=True, revision=revision)

    # create and store meta_data
    meta_data_dir = root / community_id / dataset_id / "meta_data"
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # info
    info_path = meta_data_dir / "info.json"

    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)
    # stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    # episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)

    if not dry_run:
        api = HfApi()

        api.upload_file(
            path_or_fileobj=info_path,
            path_in_repo=str(info_path).replace(f"{root}/{community_id}/{dataset_id}", ""),
            repo_id=f"{community_id}/{dataset_id}",
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=info_path,
            path_in_repo=str(info_path).replace(f"{root}/{community_id}/{dataset_id}", ""),
            repo_id=f"{community_id}/{dataset_id}",
            repo_type="dataset",
            revision=revision,
        )

        # stats
        api.upload_file(
            path_or_fileobj=stats_path,
            path_in_repo=str(stats_path).replace(f"{root}/{community_id}/{dataset_id}", ""),
            repo_id=f"{community_id}/{dataset_id}",
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=stats_path,
            path_in_repo=str(stats_path).replace(f"{root}/{community_id}/{dataset_id}", ""),
            repo_id=f"{community_id}/{dataset_id}",
            repo_type="dataset",
            revision=revision,
        )

        api.upload_file(
            path_or_fileobj=ep_data_idx_path,
            path_in_repo=str(ep_data_idx_path).replace(f"{root}/{community_id}/{dataset_id}", ""),
            repo_id=f"{community_id}/{dataset_id}",
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=ep_data_idx_path,
            path_in_repo=str(ep_data_idx_path).replace(f"{root}/{community_id}/{dataset_id}", ""),
            repo_id=f"{community_id}/{dataset_id}",
            repo_type="dataset",
            revision=revision,
        )

    # copy in tests folder, the first episode and the meta_data directory
    num_items_first_ep = episode_data_index["to"][0] - episode_data_index["from"][0]
    hf_dataset.select(range(num_items_first_ep)).with_format("torch").save_to_disk(
        f"tests/data/{community_id}/{dataset_id}/train"
    )
    if Path(f"tests/data/{community_id}/{dataset_id}/meta_data").exists():
        shutil.rmtree(f"tests/data/{community_id}/{dataset_id}/meta_data")
    shutil.copytree(meta_data_dir, f"tests/data/{community_id}/{dataset_id}/meta_data")


def push_dataset_to_hub(
    dataset_id: str,
    root: Path,
    fps: int | None,
    dataset_folder: Path | None = None,
    dry_run: bool = False,
    revision: str = "v1.1",
    community_id: str = "lerobot",
    no_preprocess: bool = False,
    path_save_to_disk: str | None = None,
    **kwargs,
) -> None:
    """
    Download a raw dataset if needed or access a local raw dataset, detect the raw format (e.g. aloha, pusht, umi) and process it accordingly in a common data format which is then pushed to the Hugging Face Hub.

    Args:
        dataset_id (str): The ID of the dataset.
        root (Path): The root directory where the dataset will be downloaded.
        fps (int | None): The desired frames per second for the dataset.
        dataset_folder (Path | None, optional): The path to the dataset folder. If not provided, the dataset will be downloaded using the dataset ID. Defaults to None.
        dry_run (bool, optional): If True, performs a dry run without actually pushing the dataset. Defaults to False.
        revision (str, optional): Version of the `push_dataset_to_hub.py` codebase used to preprocess the dataset. Defaults to "v1.1".
        community_id (str, optional): The ID of the community. Defaults to "lerobot".
        no_preprocess (bool, optional): If True, does not preprocesses the dataset. Defaults to False.
        path_save_to_disk (str | None, optional): The path to save the dataset to disk. Works when `dry_run` is True, which allows to only save on disk without uploading. By default, the dataset is not saved on disk.
        **kwargs: Additional keyword arguments for the preprocessor init method.


    """
    if dataset_folder is None:
        dataset_folder = download_raw(root=root, dataset_id=dataset_id)

    if not no_preprocess:
        processor = guess_dataset_type(dataset_folder=dataset_folder, fps=fps, **kwargs)
        data_dict, episode_data_index = processor.preprocess()
        hf_dataset = processor.to_hf_dataset(data_dict)

        info = {
            "fps": processor.fps,
        }
        stats: dict[str, dict[str, torch.Tensor]] = compute_stats(hf_dataset)

        push_lerobot_dataset_to_hub(
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
            stats=stats,
            root=root,
            revision=revision,
            dataset_id=dataset_id,
            community_id=community_id,
            dry_run=dry_run,
        )
        if path_save_to_disk:
            hf_dataset.with_format("torch").save_to_disk(dataset_path=str(path_save_to_disk))

        processor.cleanup()


class DatasetProcessor(Protocol):
    """A class for processing datasets.

    This class provides methods for validating, preprocessing, and converting datasets.

    Args:
        folder_path (str): The path to the folder containing the dataset.
        fps (int | None): The frames per second of the dataset. If None, the default value is used.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, folder_path: str, fps: int | None, *args, **kwargs) -> None: ...

    def is_valid(self) -> bool:
        """Check if the dataset is valid.

        Returns:
            bool: True if the dataset is valid, False otherwise.
        """
        ...

    def preprocess(self) -> tuple[dict, dict]:
        """Preprocess the dataset.

        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries representing the preprocessed data.
        """
        ...

    def to_hf_dataset(self, data_dict: dict) -> Dataset:
        """Convert the preprocessed data to a Hugging Face dataset.

        Args:
            data_dict (dict): The preprocessed data.

        Returns:
            Dataset: The converted Hugging Face dataset.
        """
        ...

    @property
    def fps(self) -> int:
        """Get the frames per second of the dataset.

        Returns:
            int: The frames per second.
        """
        ...

    def cleanup(self):
        """Clean up any resources used by the dataset processor."""
        ...


def guess_dataset_type(dataset_folder: Path, **processor_kwargs) -> DatasetProcessor:
    if (processor := AlohaProcessor(folder_path=dataset_folder, **processor_kwargs)).is_valid():
        return processor
    if (processor := XarmProcessor(folder_path=dataset_folder, **processor_kwargs)).is_valid():
        return processor
    if (processor := PushTProcessor(folder_path=dataset_folder, **processor_kwargs)).is_valid():
        return processor
    if (processor := UmiProcessor(folder_path=dataset_folder, **processor_kwargs)).is_valid():
        return processor
    # TODO: Propose a registration mechanism for new dataset types
    raise ValueError(f"Could not guess dataset type for folder {dataset_folder}")


def main():
    """
    Main function to process command line arguments and push dataset to Hugging Face Hub.

    Parses command line arguments to get dataset details and conditions under which the dataset
    is processed and pushed. It manages dataset preparation and uploading based on the user-defined parameters.
    """
    parser = argparse.ArgumentParser(
        description="Push a dataset to the Hugging Face Hub with optional parameters for customization.",
        epilog="""
        Example usage:
            python -m lerobot.scripts.push_dataset_to_hub --dataset-folder /path/to/dataset --dataset-id example_dataset --root /path/to/root --dry-run --revision v2.0 --community-id example_community --fps 30 --path-save-to-disk /path/to/save --no-preprocess

        This processes and optionally pushes 'example_dataset' located in '/path/to/dataset' to Hugging Face Hub,
        with various parameters to control the processing and uploading behavior.
        """,
    )

    parser.add_argument(
        "--dataset-folder",
        type=Path,
        default=None,
        help="The filesystem path to the dataset folder. If not provided, the dataset must be identified and managed by other means.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Unique identifier for the dataset to be processed and uploaded.",
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Root directory where the dataset operations are managed."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the push process without uploading any data, for testing purposes.",
    )
    parser.add_argument(
        "--community-id",
        type=str,
        default="lerobot",
        help="Community or user ID under which the dataset will be hosted on the Hub.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Target frame rate for video or image sequence datasets. Optional and applicable only if the dataset includes temporal media.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v1.0",
        help="Dataset version identifier to manage different iterations of the dataset.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Does not preprocess the dataset, set this flag if you only want dowload the dataset raw.",
    )
    parser.add_argument(
        "--path-save-to-disk",
        type=Path,
        help="Optional path where the processed dataset can be saved locally.",
    )

    args = parser.parse_args()

    push_dataset_to_hub(
        dataset_folder=args.dataset_folder,
        dataset_id=args.dataset_id,
        root=args.root,
        fps=args.fps,
        dry_run=args.dry_run,
        community_id=args.community_id,
        revision=args.revision,
        no_preprocess=args.no_preprocess,
        path_save_to_disk=args.path_save_to_disk,
    )


if __name__ == "__main__":
    main()
