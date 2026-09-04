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
import math
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
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


def make_dataset(
    cfg: TrainPipelineConfig,
    *,
    episodes: list[int] | None = None,
) -> LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset:
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

    if isinstance(cfg.dataset.repo_id, str):
        selected_episodes = cfg.dataset.episodes if episodes is None else episodes
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=selected_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                local_episode_loading=cfg.dataset.local_episode_loading,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=selected_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=max(1, cfg.num_workers),
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                video_backend=cfg.dataset.video_backend,
                data_root=cfg.dataset.streaming_data_root,
                episode_pool_size=cfg.dataset.streaming_episode_pool_size,
                prefetch_episodes=cfg.dataset.streaming_prefetch_episodes,
                byte_budget_gb=cfg.dataset.streaming_byte_budget_gb,
                decode_threads=cfg.dataset.streaming_decode_threads,
                decoded_queue_size=cfg.dataset.streaming_decoded_queue_size,
                max_open_decoders=cfg.dataset.streaming_max_open_decoders,
                native_http_connections=cfg.dataset.streaming_native_http_connections,
                native_http_subranges=cfg.dataset.streaming_native_http_subranges,
                repeat=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def resolve_train_eval_episode_indices(
    meta: LeRobotDatasetMetadata,
    episodes: list[int] | None,
    eval_split: float,
) -> tuple[list[int], list[int]]:
    """Return deterministic per-task train/eval episode selections."""
    base_episodes = episodes if episodes is not None else list(range(meta.total_episodes))
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        tasks = meta.episodes[ep_idx]["tasks"]
        task_to_episodes.setdefault(tasks[0] if tasks else "", []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for task_episodes in task_to_episodes.values():
        n_eval = math.ceil(len(task_episodes) * eval_split)
        split = len(task_episodes) - n_eval
        train_episodes.extend(task_episodes[:split])
        eval_episodes.extend(task_episodes[split:])
    return train_episodes, eval_episodes


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
    *,
    train_episodes: list[int] | None = None,
) -> tuple[
    LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset,
    LeRobotDataset | None,
]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    if train_episodes is None:
        full_dataset = make_dataset(cfg)
        if cfg.dataset.eval_split == 0.0:
            return full_dataset, None
        meta = full_dataset.meta
    else:
        if cfg.dataset.eval_split == 0.0:
            return make_dataset(cfg, episodes=train_episodes), None
        meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
    split_train_episodes, eval_episodes = resolve_train_eval_episode_indices(
        meta, cfg.dataset.episodes, cfg.dataset.eval_split
    )
    if not split_train_episodes:
        base_count = len(cfg.dataset.episodes) if cfg.dataset.episodes is not None else meta.total_episodes
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {base_count} total."
        )
    selected_train_episodes = split_train_episodes if train_episodes is None else train_episodes

    logging.info(
        f"Train/eval split: {len(split_train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split})"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, meta)
    train_dataset = make_dataset(cfg, episodes=selected_train_episodes)

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        depth_output_unit=cfg.dataset.depth_output_unit,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
