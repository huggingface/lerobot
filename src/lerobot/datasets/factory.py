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

from lerobot.configs.default import DatasetConfig, MultiDatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.multi_dataset import NewMultiLeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

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
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | NewMultiLeRobotDataset:
    """Create a single or multi-dataset depending on the config type.

    Returns:
        LeRobotDataset | NewMultiLeRobotDataset
    """
    if isinstance(cfg.dataset, MultiDatasetConfig):
        return _make_multi_dataset(cfg)

    return _make_single_dataset(cfg)


def _make_single_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset:
    ds_cfg: DatasetConfig = cfg.dataset  # type: ignore[assignment]
    image_transforms = (
        ImageTransforms(ds_cfg.image_transforms) if ds_cfg.image_transforms.enable else None
    )
    ds_meta = LeRobotDatasetMetadata(ds_cfg.repo_id, root=ds_cfg.root, revision=ds_cfg.revision)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if not ds_cfg.streaming:
        dataset = LeRobotDataset(
            ds_cfg.repo_id,
            root=ds_cfg.root,
            episodes=ds_cfg.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=ds_cfg.revision,
            video_backend=ds_cfg.video_backend,
            tolerance_s=cfg.tolerance_s,
        )
    else:
        dataset = StreamingLeRobotDataset(
            ds_cfg.repo_id,
            root=ds_cfg.root,
            episodes=ds_cfg.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=ds_cfg.revision,
            max_num_shards=cfg.num_workers,
            tolerance_s=cfg.tolerance_s,
        )

    if ds_cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats_val in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats_val, dtype=torch.float32)

    return dataset


def _make_multi_dataset(cfg: TrainPipelineConfig) -> NewMultiLeRobotDataset:
    multi_cfg: MultiDatasetConfig = cfg.dataset  # type: ignore[assignment]
    image_transforms = (
        ImageTransforms(multi_cfg.image_transforms) if multi_cfg.image_transforms.enable else None
    )

    dataset = NewMultiLeRobotDataset(
        configs=multi_cfg.datasets,
        image_transforms=image_transforms,
        tolerance_s=cfg.tolerance_s,
    )

    logging.info(
        "MultiLeRobotDataset created with %d sub-datasets:\n%s",
        len(multi_cfg.datasets),
        pformat(
            {i: c.repo_id for i, c in enumerate(multi_cfg.datasets)},
            indent=2,
        ),
    )

    if multi_cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats_val in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats_val, dtype=torch.float32)

    return dataset
