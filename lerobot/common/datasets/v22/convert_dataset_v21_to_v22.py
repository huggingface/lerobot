import argparse
import logging
import re
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import load_info, write_info
from lerobot.common.utils.utils import init_logging


def process_parquet_files(root_path: Path) -> None:
    """
    Processes parquet files in data/chunk-* folders.

    For each `episode_XXX.parquet` file found directly within a `chunk-YYY` directory:
    1. Removes the 'episode_index' column.
    2. Saves the modified data to a new directory structure:
       `data/chunk-YYY/episode_index=XXX/episode_XXX.parquet`.
    3. Records the original file path for later removal.

    Args:
        root_path: The root directory of the LeRobot dataset.

    Returns:
        A list of relative paths (from dataset root) of the original parquet files
        that were processed and removed locally.
    """
    data_folder = root_path / "data"

    chunk_pattern = re.compile(r"^chunk-(\d+)$")
    episode_pattern = re.compile(r"^episode_(\d+)\.parquet$")

    # Process only valid chunk directories
    chunk_dirs = [d for d in data_folder.iterdir() if d.is_dir() and chunk_pattern.match(d.name)]

    for chunk_path in chunk_dirs:
        logging.info(f"Processing chunk directory: {chunk_path.name}")

        # Find all episode files in this chunk directory
        episode_files = [f for f in chunk_path.iterdir() if f.is_file() and episode_pattern.match(f.name)]

        for parquet_file_path in episode_files:
            match = episode_pattern.match(parquet_file_path.name)
            if not match:
                continue

            episode_index = match.group(1)
            try:
                # Load, process, and save in new location
                ds = Dataset.from_parquet(str(parquet_file_path))
                ds = ds.remove_columns(["episode_index"])

                new_folder = chunk_path / f"episode_index={episode_index}"
                new_file_path = new_folder / parquet_file_path.name

                new_folder.mkdir(exist_ok=True)
                ds.to_parquet(new_file_path)

                # Record for removal and delete original
                parquet_file_path.unlink()

                logging.debug(f"Processed {parquet_file_path.name} successfully")

            except Exception as e:
                logging.error(f"Error processing {parquet_file_path.name}: {e}")


def rewrite_info_file(root_path: Path) -> None:
    """
    Updates the info.json file with the new data_path format and codebase_version.

    Args:
        root_path: The root directory of the LeRobot dataset.
    """
    info = load_info(root_path)
    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/episode_index={episode_index:06d}/episode_{episode_index:06d}.parquet"
    )
    info["codebase_version"] = "v2.2"
    write_info(info, root_path)
    logging.info("Updated info.json with new data path format and codebase version")


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and restructure parquet files in a LeRobot dataset, "
        "removing the 'episode_index' column and updating the directory structure. "
        "Optionally pushes changes to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of the Hugging Face repository containing the LeRobotDataset (e.g., `lerobot/pusht`).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally. If None, it will use the Hugging Face cache "
        "or download from the Hub.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,  # Use None to default to 'main'
        help="The git revision (branch, tag, or commit hash) of the dataset to process and push to. Defaults to 'main'.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload to Hugging Face hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, the repository on the Hub will be private",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="List of tags to apply to the dataset on the Hub",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="If set, upload large folders to the Hub using the `upload_large_folder` method",
    )
    parser.add_argument("--license", type=str, default=None, help="License to use for the dataset on the Hub")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    logging.info(f"Loading dataset '{args.repo_id}'...")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    logging.info(f"Dataset loaded from {dataset.root}")

    process_parquet_files(dataset.root)
    rewrite_info_file(dataset.root)

    if args.push_to_hub:
        logging.info(f"Pushing changes to the Hugging Face Hub repository '{args.repo_id}'...")
        hub_api = HfApi()
        hub_api.delete_folder(
            repo_id=args.repo_id,
            path_in_repo="data/",
            repo_type="dataset",
            commit_message="chore: remove old data folder",
        )

        dataset.push_to_hub(
            branch=args.branch,
            private=bool(args.private),
            tags=args.tags,
            license=args.license,
            upload_large_folder=bool(args.upload_large_folder),
        )
        logging.info("Changes pushed to the Hugging Face Hub successfully.")
    else:
        logging.info("Skipping push to Hub. Dataset updated locally only.")


if __name__ == "__main__":
    init_logging()
    main()
