#!/usr/bin/env python3
"""
Script to push a local LERobot dataset to the Hugging Face Hub.

You can call this script from the command line:

    python push_dataset_to_hub.py \
      --repo-id username/dataset_name \
      --root /path/to/local/dataset \
      --dataset-info /path/to/dataset_info.yaml \
      [--branch BRANCH] \
      [--tags tag1 tag2 ...] \
      [--license apache-2.0] \
      [--no-videos] \
      [--no-tag-version] \
      [--private] \
      [--allow-patterns "*.png" "*.json"] \
      [--upload-large-folder] \
      [--verbose]
"""

import argparse
import contextlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from huggingface_hub import HfApi
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.datasets.lerobot_dataset import create_lerobot_dataset_card

CODEBASE_VERSION = "v2.1"


def load_dataset_info(path: str) -> Dict[str, Any]:
    """
    Load dataset metadata from a JSON or YAML file.
    """
    with open(path, "r") as f:
        if path.lower().endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def push_to_hub(
    repo_id: str,
    root: str,
    revision: Optional[str] = None,
    branch: Optional[str] = None,
    tags: Optional[List[str]] = None,
    license: Optional[str] = "apache-2.0",
    tag_version: bool = True,
    push_videos: bool = True,
    private: bool = False,
    allow_patterns: Optional[Union[List[str], str]] = None,
    upload_large_folder: bool = False,
    dataset_info: Optional[Dict[str, Any]] = None,
    **card_kwargs: Any,
) -> None:
    """
    Push a local dataset directory to the Hugging Face Hub.

    Args:
        repo_id: HF dataset repo identifier (e.g. "user/dataset").
        root: Local dataset folder path.
        revision: Base git revision or commit ID.
        branch: Branch name to create/use on the remote.
        tags: Tags to add to the dataset card.
        license: License identifier (e.g. "apache-2.0").
        tag_version: Whether to tag the push with CODEBASE_VERSION.
        push_videos: If False, skip uploading the 'videos/' folder.
        private: Create the dataset repo as private.
        allow_patterns: File patterns to include (passed to upload).
        upload_large_folder: Use the large-folder upload API.
        dataset_info: Metadata dict for the dataset card.
        **card_kwargs: Extra kwargs to pass into create_lerobot_dataset_card.
    """
    ignore_patterns = ["images/"]
    if not push_videos:
        ignore_patterns.append("videos/")

    api = HfApi()

    logging.info(f"Ensuring dataset repo exists: {repo_id}")
    api.create_repo(repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True)

    if branch:
        logging.info(f"Creating branch '{branch}' on {repo_id}")
        api.create_branch(
            repo_id=repo_id,
            branch=branch,
            revision=revision,
            repo_type="dataset",
            exist_ok=True,
        )

    upload_kwargs = {
        "repo_id": repo_id,
        "folder_path": root,
        "repo_type": "dataset",
        "revision": branch,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
    }

    if upload_large_folder:
        logging.info("Uploading with upload_large_folder()")
        api.upload_large_folder(**upload_kwargs)
    else:
        logging.info("Uploading with upload_folder()")
        api.upload_folder(**upload_kwargs)

    # Add a dataset card if not already present
    if not api.file_exists(repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
        logging.info("Pushing dataset card to hub")
        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=dataset_info, license=license, **card_kwargs
        )
        card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=branch)

    # Tag the push with the codebase version
    if tag_version:
        with contextlib.suppress(RevisionNotFoundError):
            logging.info(f"Deleting existing tag '{CODEBASE_VERSION}'")
            api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
        logging.info(f"Creating tag '{CODEBASE_VERSION}'")
        api.create_tag(repo_id=repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Push a local dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo (e.g. user/dataset)")
    parser.add_argument("--root", required=True, help="Local dataset folder path")
    parser.add_argument("--revision", help="Base git revision or commit ID", default=None)
    parser.add_argument("--branch", help="Target branch to create/push to", default=None)
    parser.add_argument("--tags", nargs="+", help="Tags for the dataset card")
    parser.add_argument("--license", default="apache-2.0", help="License for the dataset card")
    parser.add_argument(
        "--no-tag-version",
        dest="tag_version",
        action="store_false",
        help="Disable tagging with CODEBASE_VERSION",
    )
    parser.add_argument(
        "--no-videos",
        dest="push_videos",
        action="store_false",
        help="Exclude 'videos/' folder from upload",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset as private")
    parser.add_argument(
        "--allow-patterns", nargs="+", help="File patterns to allow in upload"
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use the upload_large_folder API",
    )
    parser.add_argument(
        "--dataset-info",
        required=True,
        help="Path to JSON/YAML file containing dataset metadata for the card",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset_info = load_dataset_info(args.dataset_info)

    push_to_hub(
        repo_id=args.repo_id,
        root=args.root,
        revision=args.revision,
        branch=args.branch,
        tags=args.tags,
        license=args.license,
        tag_version=args.tag_version,
        push_videos=args.push_videos,
        private=args.private,
        allow_patterns=args.allow_patterns,
        upload_large_folder=args.upload_large_folder,
        dataset_info=dataset_info,
    )


if __name__ == "__main__":
    main()