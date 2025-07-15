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
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from lerobot.datasets.utils_must import EPISODES_DATASET_MAPPING, FEATURE_KEYS_MAPPING


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
            returns `None` if the resulting dict is empty.
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
    if "," in cfg.dataset.repo_id:
        repo_id = cfg.dataset.repo_id.split(",")
        repo_id = [r for r in repo_id if r]
    else:
        repo_id = cfg.dataset.repo_id
    sampling_weights = cfg.dataset.sampling_weights.split(",") if cfg.dataset.sampling_weights else None
    feature_keys_mapping = FEATURE_KEYS_MAPPING
    if isinstance(repo_id, str):
        revision = getattr(cfg.dataset, "revision", None)
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            feature_keys_mapping=feature_keys_mapping,
            revision=revision,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=getattr(cfg.dataset, "root", None),
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=revision,
            video_backend=cfg.dataset.video_backend,
            download_videos=True,
            feature_keys_mapping=feature_keys_mapping,
            max_action_dim=cfg.dataset.max_action_dim,
            max_state_dim=cfg.dataset.max_state_dim,
            max_num_images=cfg.dataset.max_num_images,
            max_image_dim=cfg.dataset.max_image_dim,
        )
    else:
        delta_timestamps = {}
        episodes = {}
        for i in range(len(repo_id)):
            ds_meta = LeRobotDatasetMetadata(
                repo_id[i],
                feature_keys_mapping=feature_keys_mapping,
            )  # FIXME(mshukor): ?
            delta_timestamps[repo_id[i]] = resolve_delta_timestamps(cfg.policy, ds_meta)
            episodes[repo_id[i]] = EPISODES_DATASET_MAPPING.get(repo_id[i], cfg.dataset.episodes)
        # training_features = TRAINING_FEATURES.get(cfg.dataset.features_version, None)
        # FIXME: (jadechoghari): check support for training features
        training_features = None
        dataset = MultiLeRobotDataset(
            repo_id,
            # TODO(aliberts): add proper support for multi dataset
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            download_videos=True,
            sampling_weights=sampling_weights,
            feature_keys_mapping=feature_keys_mapping,
            max_action_dim=cfg.policy.max_action_dim,
            max_state_dim=cfg.policy.max_state_dim,
            max_num_images=cfg.dataset.max_num_images,
            max_image_dim=cfg.dataset.max_image_dim,
            train_on_all_features=cfg.dataset.train_on_all_features,
            training_features=training_features,
            discard_first_n_frames=cfg.dataset.discard_first_n_frames,
            min_fps=cfg.dataset.min_fps,
            max_fps=cfg.dataset.max_fps,
            discard_first_idle_frames=cfg.dataset.discard_first_idle_frames,
            motion_threshold=cfg.dataset.motion_threshold,
            motion_window_size=cfg.dataset.motion_window_size,
            motion_buffer=cfg.dataset.motion_buffer,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
