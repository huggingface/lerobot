#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from typing import List
from pprint import pformat

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.constants import HF_LEROBOT_HOME

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps

def resolve_delta_timestamps_without_config(
    ds_meta: LeRobotDatasetMetadata, action_delta_indices: List, observation_delta_indices: List = None
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action" and action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in action_delta_indices]
        if key == "safety_violation_index" and action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in action_delta_indices]
        if key.startswith("observation.state") and observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    if cfg.dataset.repo_id.startswith('['):
        datasets = cfg.dataset.repo_id.strip('[]').split(',')
        datasets = [x.strip() for x in datasets]
        delta_timestamps = {}
        for ds in datasets:
            ds_meta = LeRobotDatasetMetadata(
                ds,
                root=HF_LEROBOT_HOME / ds,
            )
            d_ts = resolve_delta_timestamps(cfg.policy, ds_meta)
            delta_timestamps[ds] = d_ts
        dataset = MultiLeRobotDataset(
            datasets,
            # TODO(aliberts): add proper support for multi dataset
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            force_cache_sync=True,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index , indent=2)}"
        )
    else:
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            force_cache_sync=True,
        )

    if cfg.dataset.use_imagenet_stats:
        if isinstance(dataset, MultiLeRobotDataset):
            for ds in dataset._datasets:
                for key in ds.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        else:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

def make_dataset_without_config(
    repo_id: str,
    action_delta_indices: List,
    observation_delta_indices: List = None,
    root: str = None,
    video_backend: str = "pyav",
    episodes: list[int] | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    use_imagenet_stats: bool = True,
    force_cache_sync: bool = False,
) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    if repo_id.startswith('['):
        datasets = repo_id.strip('[]').split(',')
        datasets = [x.strip() for x in datasets]
        delta_timestamps = {}
        for ds in datasets:
            ds_meta = LeRobotDatasetMetadata(
                ds,
                root=HF_LEROBOT_HOME / ds,
                force_cache_sync=force_cache_sync,
            )
            d_ts = resolve_delta_timestamps_without_config(ds_meta, action_delta_indices, observation_delta_indices)
            delta_timestamps[ds] = d_ts
        dataset = MultiLeRobotDataset(
            datasets,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            force_cache_sync=force_cache_sync,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index , indent=2)}"
        )
    else:
        ds_meta = LeRobotDatasetMetadata(repo_id, local_files_only=local_files_only)
        delta_timestamps = resolve_delta_timestamps_without_config(ds_meta, action_delta_indices, observation_delta_indices)
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            revision=revision,
            video_backend=video_backend,
        )

    if use_imagenet_stats:
        if isinstance(dataset, MultiLeRobotDataset):
            for ds in dataset._datasets:
                for key in ds.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        else:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
