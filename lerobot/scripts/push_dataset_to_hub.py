import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any, Protocol

import torch
import tqdm
from datasets import Dataset
from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.common.datasets.push_dataset_to_hub.aloha_processor import (
    AlohaProcessor,
)
from lerobot.common.datasets.push_dataset_to_hub.pusht_processor import PushTProcessor
from lerobot.common.datasets.push_dataset_to_hub.umi_processor import UmiProcessor
from lerobot.common.datasets.push_dataset_to_hub.xarm_processor import XarmProcessor
from lerobot.common.datasets.utils import (
    compute_stats,
)


def download_raw(root, dataset_id) -> Path:
    """
    Downloads the raw dataset based on the given dataset_id.

    Args:
        root (str): The root directory where the dataset will be downloaded.
        dataset_id (str): The identifier for the dataset.

    Returns:
        Path: The path to the downloaded raw dataset.

    Raises:
        ValueError: If the dataset_id is not recognized.

    """
    if "pusht" in dataset_id:
        return download_pusht(root=root, dataset_id=dataset_id)
    elif "xarm" in dataset_id:
        return download_xarm(root=root, dataset_id=dataset_id)
    elif "aloha" in dataset_id:
        return download_aloha(root=root, dataset_id=dataset_id)
    elif "umi" in dataset_id:
        return download_umi(root=root, dataset_id=dataset_id)
    else:
        raise ValueError(dataset_id)


def download_and_extract_zip(url: str, destination_folder: Path) -> bool:
    import zipfile

    import requests

    print(f"downloading from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)

        zip_file = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                zip_file.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()

        zip_file.seek(0)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        return True
    else:
        return False


def download_pusht(root: str, dataset_id: str = "pusht", fps: int = 10) -> Path:
    pusht_url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
    pusht_zarr = Path("pusht/pusht_cchi_v7_replay.zarr")

    root = Path(root)
    raw_dir: Path = root / f"{dataset_id}_raw"
    zarr_path: Path = (raw_dir / pusht_zarr).resolve()
    if not zarr_path.is_dir():
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(pusht_url, raw_dir)
    return zarr_path


def download_xarm(root: str, dataset_id: str, fps: int = 15) -> Path:
    root = Path(root)
    raw_dir: Path = root / "xarm_datasets_raw"
    if not raw_dir.exists():
        import zipfile

        import gdown

        raw_dir.mkdir(parents=True, exist_ok=True)
        # from https://github.com/fyhMer/fowm/blob/main/scripts/download_datasets.py
        url = "https://drive.google.com/uc?id=1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya"
        zip_path = raw_dir / "data.zip"
        gdown.download(url, str(zip_path), quiet=False)
        print("Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as zip_f:
            for member in zip_f.namelist():
                if member.startswith("data/xarm") and member.endswith(".pkl"):
                    print(member)
                    zip_f.extract(member=member)
        zip_path.unlink()

    dataset_path: Path = root / f"{dataset_id}"
    return dataset_path


def download_aloha(root: str, dataset_id: str) -> Path:
    # TODO(rcadene): use hugging face utils to download from google drive
    folder_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/drive/folders/1RgyD0JgTX30H4IM5XZn8I3zSV_mr8pyF",
        "aloha_sim_insertion_scripted": "https://drive.google.com/drive/folders/1TsojQQSXtHEoGnqgJ3gmpPQR2DPLtS2N",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/drive/folders/1sc-E4QYW7A0o23m1u2VWNGVq5smAsfCo",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj",
    }

    ep48_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/file/d/18Cudl6nikDtgRolea7je8iF_gGKzynOP/view?usp=drive_link",
        "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/1wfMSZ24oOh5KR_0aaP3Cnu_c4ZCveduB/view?usp=drive_link",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/18smMymtr8tIxaNUQ61gW6dG50pt3MvGq/view?usp=drive_link",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1pnGIOd-E4-rhz2P3VxpknMKRZCoKt6eI/view?usp=drive_link",
    }

    ep49_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/file/d/1C1kZYyROzs-PrLc0SkDgUgMi4-L3lauE/view?usp=drive_link",
        "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/17EuCUWS6uCCr6yyNzpXdcdE-_TTNCKtf/view?usp=drive_link",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/1Nk7l53d9sJoGDBKAOnNrExX5nLacATc6/view?usp=drive_link",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1GKReZHrXU73NMiC5zKCq_UtqPVtYq8eo/view?usp=drive_link",
    }
    num_episodes = {  # noqa: F841 # we keep this for reference
        "aloha_sim_insertion_human": 50,
        "aloha_sim_insertion_scripted": 50,
        "aloha_sim_transfer_cube_human": 50,
        "aloha_sim_transfer_cube_scripted": 50,
    }

    episode_len = {  # noqa: F841 # we keep this for reference
        "aloha_sim_insertion_human": 500,
        "aloha_sim_insertion_scripted": 400,
        "aloha_sim_transfer_cube_human": 400,
        "aloha_sim_transfer_cube_scripted": 400,
    }

    cameras = {  # noqa: F841 # we keep this for reference
        "aloha_sim_insertion_human": ["top"],
        "aloha_sim_insertion_scripted": ["top"],
        "aloha_sim_transfer_cube_human": ["top"],
        "aloha_sim_transfer_cube_scripted": ["top"],
    }
    root = Path(root)
    raw_dir: Path = root / f"{dataset_id}_raw"
    if not raw_dir.is_dir():
        import gdown

        assert dataset_id in folder_urls
        assert dataset_id in ep48_urls
        assert dataset_id in ep49_urls

        raw_dir.mkdir(parents=True, exist_ok=True)

        gdown.download_folder(folder_urls[dataset_id], output=str(raw_dir))

        # because of the 50 files limit per directory, two files episode 48 and 49 were missing
        gdown.download(ep48_urls[dataset_id], output=str(raw_dir / "episode_48.hdf5"), fuzzy=True)
        gdown.download(ep49_urls[dataset_id], output=str(raw_dir / "episode_49.hdf5"), fuzzy=True)
    return raw_dir


def download_umi(root: str, dataset_id: str) -> Path:
    url_cup_in_the_wild = "https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip"
    cup_in_the_wild_zarr = Path("umi/cup_in_the_wild/cup_in_the_wild.zarr")

    root = Path(root)
    raw_dir: Path = root / f"{dataset_id}_raw"
    zarr_path: Path = (raw_dir / cup_in_the_wild_zarr).resolve()
    if not zarr_path.is_dir():
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(url_cup_in_the_wild, zarr_path)
    return zarr_path


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
    if dry_run:
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

    # episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)

    if dry_run:
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
    preprocess: bool = True,
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
        preprocess (bool, optional): If True, preprocesses the dataset. Defaults to True.
        path_save_to_disk (str | None, optional): The path to save the dataset to disk. Works when `dry_run` is True, which allows to only save on disk without uploading. By default, the dataset is not saved on disk.
        **kwargs: Additional keyword arguments for the preprocessor init method.


    """
    if dataset_folder is None:
        dataset_folder = download_raw(root=root, dataset_id=dataset_id)

    if preprocess:
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
            python -m lerobot.scripts.push_dataset_to_hub --dataset-folder /path/to/dataset --dataset-id example_dataset --root /path/to/root --dry-run --revision v2.0 --community-id example_community --fps 30 --path-save-to-disk /path/to/save --preprocess

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
        "--preprocess",
        action="store_true",
        help="Preprocess the dataset, set to false if you want dowload the dataset raw.",
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
        preprocess=args.preprocess,
        path_save_to_disk=args.path_save_to_disk,
    )


if __name__ == "__main__":
    main()
