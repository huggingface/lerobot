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
Edit LeRobot datasets using various transformation tools.

This script allows you to delete episodes, split datasets, merge datasets,
remove features, and add features. When new_repo_id is specified, creates a new dataset.

Usage Examples:

Delete episodes 0, 2, and 5 from a dataset:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Delete episodes and save to a new dataset:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_filtered \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Split dataset by fractions:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.8, "val": 0.2}'

Split dataset by episode indices:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": [0, 1, 2, 3], "val": [4, 5]}'

Split into more than two splits:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.6, "val": 0.2, "test": 0.2}'

Merge multiple datasets:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

Remove camera feature:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type remove_feature \
        --operation.feature_names "['observation.images.top']"

Add feature from numpy file:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type add_feature \
        --operation.features '{"reward": {"file": "rewards.npy", "dtype": "float32", "shape": [1], "names": null}}'

Using JSON config file:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --config_path path/to/edit_config.json
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


@dataclass
class DeleteEpisodesConfig:
    type: str = "delete_episodes"
    episode_indices: list[int] | None = None


@dataclass
class SplitConfig:
    type: str = "split"
    splits: dict[str, float | list[int]] | None = None


@dataclass
class MergeConfig:
    type: str = "merge"
    repo_ids: list[str] | None = None


@dataclass
class RemoveFeatureConfig:
    type: str = "remove_feature"
    feature_names: list[str] | None = None


@dataclass
class AddFeatureConfig:
    type: str = "add_feature"
    features: dict[str, dict] | None = None


@dataclass
class EditDatasetConfig:
    repo_id: str
    operation: DeleteEpisodesConfig | SplitConfig | MergeConfig | RemoveFeatureConfig | AddFeatureConfig
    root: str | None = None
    new_repo_id: str | None = None
    push_to_hub: bool = False


def get_output_path(repo_id: str, new_repo_id: str | None, root: Path | None) -> tuple[str, Path]:
    if new_repo_id:
        output_repo_id = new_repo_id
        output_dir = root / new_repo_id if root else HF_LEROBOT_HOME / new_repo_id
    else:
        output_repo_id = repo_id
        dataset_path = root / repo_id if root else HF_LEROBOT_HOME / repo_id
        old_path = Path(str(dataset_path) + "_old")

        if dataset_path.exists():
            if old_path.exists():
                shutil.rmtree(old_path)
            shutil.move(str(dataset_path), str(old_path))

        output_dir = dataset_path

    return output_repo_id, output_dir


def handle_delete_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, DeleteEpisodesConfig):
        raise ValueError("Operation config must be DeleteEpisodesConfig")

    if not cfg.operation.episode_indices:
        raise ValueError("episode_indices must be specified for delete_episodes operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    logging.info(f"Deleting episodes {cfg.operation.episode_indices} from {cfg.repo_id}")
    new_dataset = delete_episodes(
        dataset,
        episode_indices=cfg.operation.episode_indices,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}, Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_split(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, SplitConfig):
        raise ValueError("Operation config must be SplitConfig")

    if not cfg.operation.splits:
        raise ValueError(
            "splits dict must be specified with split names as keys and fractions/episode lists as values"
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    logging.info(f"Splitting dataset {cfg.repo_id} with splits: {cfg.operation.splits}")
    split_datasets = split_dataset(dataset, splits=cfg.operation.splits)

    for split_name, split_ds in split_datasets.items():
        split_repo_id = f"{cfg.repo_id}_{split_name}"
        logging.info(
            f"{split_name}: {split_ds.meta.total_episodes} episodes, {split_ds.meta.total_frames} frames"
        )

        if cfg.push_to_hub:
            logging.info(f"Pushing {split_name} split to hub as {split_repo_id}")
            LeRobotDataset(split_ds.repo_id, root=split_ds.root).push_to_hub()


def handle_merge(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, MergeConfig):
        raise ValueError("Operation config must be MergeConfig")

    if not cfg.operation.repo_ids:
        raise ValueError("repo_ids must be specified for merge operation")

    if not cfg.repo_id:
        raise ValueError("repo_id must be specified as the output repository for merged dataset")

    logging.info(f"Loading {len(cfg.operation.repo_ids)} datasets to merge")
    datasets = [LeRobotDataset(repo_id, root=cfg.root) for repo_id in cfg.operation.repo_ids]

    output_dir = Path(cfg.root) / cfg.repo_id if cfg.root else HF_LEROBOT_HOME / cfg.repo_id

    logging.info(f"Merging datasets into {cfg.repo_id}")
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=cfg.repo_id,
        output_dir=output_dir,
    )

    logging.info(f"Merged dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {merged_dataset.meta.total_episodes}, Frames: {merged_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {cfg.repo_id}")
        LeRobotDataset(merged_dataset.repo_id, root=output_dir).push_to_hub()


def handle_remove_feature(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RemoveFeatureConfig):
        raise ValueError("Operation config must be RemoveFeatureConfig")

    if not cfg.operation.feature_names:
        raise ValueError("feature_names must be specified for remove_feature operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    logging.info(f"Removing features {cfg.operation.feature_names} from {cfg.repo_id}")
    new_dataset = remove_feature(
        dataset,
        feature_names=cfg.operation.feature_names,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Remaining features: {list(new_dataset.meta.features.keys())}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_add_feature(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, AddFeatureConfig):
        raise ValueError("Operation config must be AddFeatureConfig")

    if not cfg.operation.features:
        raise ValueError("features must be specified for add_feature operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    # Process features config to load data and prepare for add_features
    features_dict = {}
    for feature_name, feature_config in cfg.operation.features.items():
        # Extract feature info (dtype, shape, names)
        shape = feature_config.get("shape")
        dtype = feature_config.get("dtype")
        # Convert and validate shape before assignment
        if isinstance(shape, list):
            shape = tuple(shape)
        elif isinstance(shape, tuple) or shape is None:
            pass  # shape is already valid
        else:
            raise ValueError(
                f"Feature '{feature_name}' has invalid shape type: {type(shape).__name__}. "
                "Shape must be a list, tuple, or None."
            )
        # Validate required metadata fields
        if dtype is None:
            raise ValueError(f"Feature '{feature_name}' must specify a 'dtype' (data type)")
        if shape is None:
            raise ValueError(f"Feature '{feature_name}' must specify a 'shape'")

        feature_info = {
            "dtype": dtype,
            "shape": shape,
            "names": feature_config.get("names"),
        }

        # Load feature data from file
        feature_file = feature_config.get("file")
        if not feature_file:
            raise ValueError(f"Feature '{feature_name}' must specify a 'file' path to load data from")

        file_path = Path(feature_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        # Load numpy array
        if file_path.suffix == ".npy":
            feature_data = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format for feature '{feature_name}': {file_path.suffix}")

        # Validate data length matches dataset
        expected_length = dataset.meta.total_frames
        if len(feature_data) != expected_length:
            raise ValueError(
                f"Feature '{feature_name}' data length ({len(feature_data)}) "
                f"does not match dataset length ({expected_length})"
            )

        features_dict[feature_name] = (feature_data, feature_info)

    logging.info(f"Adding features {list(features_dict.keys())} to {cfg.repo_id}")
    new_dataset = add_features(
        dataset,
        features=features_dict,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Updated features: {list(new_dataset.meta.features.keys())}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


@parser.wrap()
def edit_dataset(cfg: EditDatasetConfig) -> None:
    operation_type = cfg.operation.type

    if operation_type == "delete_episodes":
        handle_delete_episodes(cfg)
    elif operation_type == "split":
        handle_split(cfg)
    elif operation_type == "merge":
        handle_merge(cfg)
    elif operation_type == "remove_feature":
        handle_remove_feature(cfg)
    elif operation_type == "add_feature":
        handle_add_feature(cfg)
    else:
        raise ValueError(
            f"Unknown operation type: {operation_type}\n"
            f"Available operations: delete_episodes, split, merge, remove_feature, add_feature"
        )


def main() -> None:
    init_logging()
    edit_dataset()


if __name__ == "__main__":
    main()
