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
from typing import Literal, cast

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STR

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .storage import DEFAULT_STORAGE_FORMAT, is_bucket_root, load_dataset_metadata
from .streaming_dataset import DEFAULT_STREAMING_SEED, StreamingLeRobotDataset
from .utils import resolve_episode_indices

logger = logging.getLogger(__name__)


def _observation_type(policy_key: str, feature: dict) -> FeatureType | None:
    """The type a dataset feature has as the policy's observation ``policy_key``."""
    if feature.get("dtype") in ("image", "video") or policy_key.startswith(OBS_IMAGE):
        return FeatureType.VISUAL
    if policy_key == OBS_ENV_STATE:
        return FeatureType.ENV
    if policy_key.startswith(OBS_STR + "."):
        return FeatureType.STATE
    return None


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str] | None = None,
) -> dict[str, list[float]] | None:
    """Resolves delta_timestamps from the config's ``training_deltas``.

    Each dataset feature gets the offsets its type declares (see
    :class:`~lerobot.configs.types.Deltas`); ``observation.action`` is read from the
    ``action`` column at its own offsets.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config whose ``training_deltas`` to load.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.
        rename_map: Dataset keys renamed to the policy's.

    Returns:
        dict[str, list[float]] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "action": [0, 0.02, 0.04]
            }
            returns `None` if the resulting dict is empty.
    """
    deltas = cfg.training_deltas
    types = {}
    for key, feature in ds_meta.features.items():
        policy_key = (rename_map or {}).get(key, key)
        types[key] = _observation_type(policy_key, feature)
    delta_timestamps: dict[str, list[float]] = {}
    for key in ds_meta.features:
        policy_key = (rename_map or {}).get(key, key)
        offsets = deltas.of(policy_key, types[key])
        if offsets is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in offsets]
    if (offsets := deltas.of(OBS_ACTION)) is not None:
        if ACTION not in ds_meta.features:
            raise ValueError(f"{type(cfg).__name__} reads observation.action but the dataset has no action.")
        delta_timestamps[OBS_ACTION] = [i / ds_meta.fps for i in offsets]

    # A policy asking for an image history that no dataset key can supply would train
    # on single frames without any error, so fail instead of degrading silently.
    image_offsets = deltas.observation.get(FeatureType.VISUAL, ())
    if len(image_offsets) > 1 and FeatureType.VISUAL not in types.values():
        raise ValueError(
            f"{type(cfg).__name__} requests {len(image_offsets)} frames per camera, but no "
            f"dataset feature maps to an image key. Dataset features: {sorted(ds_meta.features)}. "
            "Image keys must be named `observation.image*` after applying `--rename_map`."
        )
    # A window declared for one feature that the dataset lacks would be dropped silently too.
    policy_keys = {(rename_map or {}).get(key, key) for key in ds_meta.features}
    keys = [key for key in deltas.observation if not isinstance(key, FeatureType)]
    if missing := sorted(key for key in keys if key not in policy_keys):
        raise ValueError(
            f"{type(cfg).__name__} declares frames of {missing}, which no dataset feature maps to. "
            f"Dataset features: {sorted(ds_meta.features)}; rename them with `--rename_map`."
        )

    if len(delta_timestamps) == 0:
        return None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | StreamingLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | StreamingLeRobotDataset: a StreamingLeRobotDataset when `cfg.dataset.streaming` is set.
    """
    # On the "gpu" backend the workers only decode; lerobot-train augments each batch on the policy device.
    image_transforms_cfg = cfg.dataset.image_transforms
    image_transforms = (
        ImageTransforms(image_transforms_cfg)
        if image_transforms_cfg.enable and image_transforms_cfg.backend == "dataloader"
        else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        root, repo_type_name = cfg.dataset.root, cfg.dataset.repo_type
        if cfg.dataset.streaming and is_bucket_root(cfg.dataset.repo_id, root):
            # Streaming reads default-format buckets directly, so a bucket URI root selects
            # bucket mode instead of the object-store localization used by other formats.
            root, repo_type_name = None, "bucket"
        repo_type = cast(Literal["dataset", "bucket"], repo_type_name)
        # Storage-aware loader: same as LeRobotDatasetMetadata(...), plus support
        # for datasets whose root is an object-store URI (e.g. ``hf://``).
        ds_meta = load_dataset_metadata(
            cfg.dataset.repo_id,
            root=root,
            revision=cfg.dataset.revision,
            repo_type=repo_type,
            storage_options=cfg.dataset.storage_options,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta, cfg.rename_map)
        episodes = resolve_episode_indices(
            cfg.dataset.episodes, ds_meta.total_episodes, cfg.dataset.exclude_episodes
        )
        if cfg.dataset.streaming and ds_meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"dataset.streaming=True is not supported for storage_format="
                f"{ds_meta.storage_format!r}: StreamingLeRobotDataset only reads the default "
                f"{DEFAULT_STORAGE_FORMAT!r} layout. Note that some formats (e.g. 'lance') "
                "support remote map-style access without streaming mode."
            )
        if not cfg.dataset.streaming:
            if repo_type == "bucket" and ds_meta.storage_format == DEFAULT_STORAGE_FORMAT:
                raise ValueError(
                    f"repo_type='bucket' is streaming-only for the default {DEFAULT_STORAGE_FORMAT!r} "
                    "storage format: set dataset.streaming=true to train from an HF Storage Bucket."
                )
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                repo_type=repo_type,
                video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
                storage_options=cfg.dataset.storage_options,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                seed=cfg.seed if cfg.seed is not None else DEFAULT_STREAMING_SEED,
                max_num_shards=max(1, cfg.num_workers),
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                video_backend=cfg.dataset.video_backend,
                episode_pool_size=cfg.dataset.streaming_episode_pool_size,
                sampling_strategy=cfg.dataset.streaming_sampling_strategy,
                prefetch_episodes=cfg.dataset.streaming_prefetch_episodes,
                byte_budget_gb=cfg.dataset.streaming_byte_budget_gb,
                decode_threads=cfg.dataset.streaming_decode_threads,
                decoded_queue_size=cfg.dataset.streaming_decoded_queue_size,
                video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
                native_http_connections=cfg.dataset.streaming_native_http_connections,
                native_http_subranges=cfg.dataset.streaming_native_http_subranges,
                sidecar_lock_timeout_s=cfg.dataset.streaming_sidecar_lock_timeout_s,
                repeat=True,
                repo_type=repo_type,
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
        logger.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | StreamingLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logger.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta, cfg.rename_map)

    # On the "gpu" backend the workers only decode; lerobot-train augments each batch on the policy device.
    image_transforms_cfg = cfg.dataset.image_transforms
    train_image_transforms = (
        ImageTransforms(image_transforms_cfg)
        if image_transforms_cfg.enable and image_transforms_cfg.backend == "dataloader"
        else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
        video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
        storage_options=cfg.dataset.storage_options,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
        video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
        storage_options=cfg.dataset.storage_options,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                if key in ds.meta.depth_keys:
                    continue
                ds.meta.stats.setdefault(key, {})
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
