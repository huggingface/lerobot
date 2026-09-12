#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to download a dataset from HuggingFace Hub and upload it to a new repository.

This utility script allows you to:
- Download a dataset from a source HuggingFace Hub repository
- Upload it to a target repository (useful for creating copies or forks)

Usage:

Download and upload a dataset:
```bash
python src/lerobot/datasets/download_and_upload_dataset.py \
    --source-repo lerobot/aloha_mobile_cabinet \
    --target-repo user/my_dataset_copy \
    --download-videos
```

Download without videos:
```bash
python src/lerobot/datasets/download_and_upload_dataset.py \
    --source-repo user/dataset \
    --target-repo user/new_dataset \
    --no-download-videos
```

Specify a specific revision to avoid API calls:
```bash
python src/lerobot/datasets/download_and_upload_dataset.py \
    --source-repo user/dataset \
    --target-repo user/new_dataset \
    --revision v3.0
```
"""

import argparse
import logging
import socket
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def check_internet_connection() -> bool:
    """Check if there's an active internet connection.

    Returns:
        bool: True if internet connection is available, False otherwise.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=3)
            return True
        except OSError:
            return False


def download_and_upload_dataset(
    source_repo_id: str,
    target_repo_id: str,
    download_videos: bool = True,
    private: bool = False,
    branch: str | None = None,
    tags: list[str] | None = None,
    revision: str | None = None,
) -> None:
    """Download a dataset from source repository and upload it to target repository.

    The dataset will be downloaded to the local cache directory (typically
    ~/.cache/huggingface/lerobot/{source_repo_id}) and then uploaded to the target
    repository on HuggingFace Hub.

    Args:
        source_repo_id: Repository ID of the source dataset (e.g., "user/dataset").
        target_repo_id: Repository ID for the target dataset (e.g., "user/new_dataset").
        download_videos: Whether to download video files. Defaults to True.
        private: Whether the target repository should be private. Defaults to False.
        branch: Branch name for the target repository. Defaults to None (uses default branch).
        tags: List of tags to add to the dataset card. Defaults to None.
        revision: Specific revision/version to download (e.g., "v3.0", "main").
            If None, will try to auto-detect. Use "v3.0" to avoid API calls that require
            internet connection.
    """
    logger.info(f"Downloading dataset from: {source_repo_id}")
    logger.info(f"Target repository: {target_repo_id}")
    logger.info(f"Download videos: {download_videos}")
    if revision:
        logger.info(f"Using revision: {revision}")
    else:
        logger.warning(
            "No revision specified. This may require internet connection to detect version. "
            "Consider using --revision v3.0 to avoid API calls."
        )

    # Check internet connection
    if not check_internet_connection():
        logger.error(
            "No internet connection detected. Cannot download dataset from HuggingFace Hub. "
            "Please check your network connection and try again."
        )
        sys.exit(1)

    # Download the dataset
    # LeRobotDataset automatically downloads when instantiated with a repo_id
    # Specify revision to avoid API calls that might fail without internet
    logger.info("Instantiating LeRobotDataset (this will trigger download if not cached)...")
    dataset = LeRobotDataset(
        repo_id=source_repo_id,
        download_videos=download_videos,
        revision=revision,  # Use explicit revision to avoid get_safe_version API call
    )

    logger.info("Dataset downloaded successfully!")
    logger.info(f"  Total episodes: {dataset.meta.total_episodes}")
    logger.info(f"  Total frames: {dataset.meta.total_frames}")
    logger.info(f"  FPS: {dataset.meta.fps}")
    logger.info(f"  Root directory: {dataset.root}")

    # Change the repo_id to the target repository
    dataset.repo_id = target_repo_id

    # Upload to the target repository
    logger.info(f"Uploading dataset to: {target_repo_id}")
    dataset.push_to_hub(
        branch=branch,
        tags=tags,
        push_videos=download_videos,
        private=private,
    )

    logger.info(f"Dataset successfully uploaded to: {target_repo_id}")
    logger.info(f"View at: https://huggingface.co/datasets/{target_repo_id}")


def main() -> None:
    """Main entry point for the download and upload dataset script."""
    init_logging()

    parser = argparse.ArgumentParser(
        description="Download a dataset from HuggingFace Hub and upload it to a new repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and upload with videos
  python download_and_upload_dataset.py \\
      --source-repo user/dataset \\
      --target-repo user/new_dataset \\
      --download-videos

  # Download without videos
  python download_and_upload_dataset.py \\
      --source-repo user/dataset \\
      --target-repo user/new_dataset \\
      --no-download-videos

  # Specify revision to avoid API calls
  python download_and_upload_dataset.py \\
      --source-repo user/dataset \\
      --target-repo user/new_dataset \\
      --revision v3.0
        """,
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        required=True,
        help="Source repository ID (e.g., 'user/dataset')",
    )
    parser.add_argument(
        "--target-repo",
        type=str,
        required=True,
        help="Target repository ID (e.g., 'user/new_dataset')",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        default=True,
        help="Download video files (default: True)",
    )
    parser.add_argument(
        "--no-download-videos",
        dest="download_videos",
        action="store_false",
        help="Skip downloading video files",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the target repository private",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch name for the target repository (default: main)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags to add to the dataset card (e.g., --tags robotics manipulation)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific revision/version to download (e.g., 'v3.0', 'main'). "
        "Use 'v3.0' to avoid API calls that require internet connection.",
    )

    args = parser.parse_args()

    download_and_upload_dataset(
        source_repo_id=args.source_repo,
        target_repo_id=args.target_repo,
        download_videos=args.download_videos,
        private=args.private,
        branch=args.branch,
        tags=args.tags,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
