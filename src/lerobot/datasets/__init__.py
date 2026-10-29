#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports, require_package

require_package("datasets", extra="dataset")
require_package("av", extra="dataset")

# Most of these import torch, so each is imported the first time it is used.
if TYPE_CHECKING:
    from .aggregate import aggregate_datasets
    from .compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
    from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
    from .dataset_reader import BaseDatasetReader
    from .dataset_tools import (
        add_features,
        convert_image_to_video_dataset,
        delete_episodes,
        merge_datasets,
        modify_features,
        modify_tasks,
        recompute_stats,
        reencode_dataset,
        remove_feature,
        split_dataset,
    )
    from .factory import make_dataset, make_train_eval_datasets, resolve_delta_timestamps
    from .image_writer import safe_stop_image_writer
    from .io_utils import load_episodes, write_stats
    from .language import EVENT_ONLY_STYLES, PERSISTENT_STYLES, STYLE_REGISTRY, column_for_style
    from .lerobot_dataset import LeRobotDataset
    from .multi_dataset import MultiLeRobotDataset
    from .pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from .pyav_utils import check_video_encoder_parameters_pyav, detect_available_encoders_pyav
    from .sampler import EpisodeAwareSampler, compute_sampler_state
    from .storage import register_dataset_reader
    from .streaming_dataset import StreamingLeRobotDataset
    from .utils import DEFAULT_EPISODES_PATH, create_lerobot_dataset_card, resolve_episode_indices
    from .video_utils import VideoEncodingManager
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "aggregate_datasets": ".aggregate.aggregate_datasets",
            "DEFAULT_QUANTILES": ".compute_stats.DEFAULT_QUANTILES",
            "aggregate_stats": ".compute_stats.aggregate_stats",
            "get_feature_stats": ".compute_stats.get_feature_stats",
            "CODEBASE_VERSION": ".dataset_metadata.CODEBASE_VERSION",
            "LeRobotDatasetMetadata": ".dataset_metadata.LeRobotDatasetMetadata",
            "BaseDatasetReader": ".dataset_reader.BaseDatasetReader",
            "add_features": ".dataset_tools.add_features",
            "convert_image_to_video_dataset": ".dataset_tools.convert_image_to_video_dataset",
            "delete_episodes": ".dataset_tools.delete_episodes",
            "merge_datasets": ".dataset_tools.merge_datasets",
            "modify_features": ".dataset_tools.modify_features",
            "modify_tasks": ".dataset_tools.modify_tasks",
            "recompute_stats": ".dataset_tools.recompute_stats",
            "reencode_dataset": ".dataset_tools.reencode_dataset",
            "remove_feature": ".dataset_tools.remove_feature",
            "split_dataset": ".dataset_tools.split_dataset",
            "make_dataset": ".factory.make_dataset",
            "make_train_eval_datasets": ".factory.make_train_eval_datasets",
            "resolve_delta_timestamps": ".factory.resolve_delta_timestamps",
            "safe_stop_image_writer": ".image_writer.safe_stop_image_writer",
            "load_episodes": ".io_utils.load_episodes",
            "write_stats": ".io_utils.write_stats",
            "EVENT_ONLY_STYLES": ".language.EVENT_ONLY_STYLES",
            "PERSISTENT_STYLES": ".language.PERSISTENT_STYLES",
            "STYLE_REGISTRY": ".language.STYLE_REGISTRY",
            "column_for_style": ".language.column_for_style",
            "LeRobotDataset": ".lerobot_dataset.LeRobotDataset",
            "MultiLeRobotDataset": ".multi_dataset.MultiLeRobotDataset",
            "aggregate_pipeline_dataset_features": ".pipeline_features.aggregate_pipeline_dataset_features",
            "create_initial_features": ".pipeline_features.create_initial_features",
            "check_video_encoder_parameters_pyav": ".pyav_utils.check_video_encoder_parameters_pyav",
            "detect_available_encoders_pyav": ".pyav_utils.detect_available_encoders_pyav",
            "EpisodeAwareSampler": ".sampler.EpisodeAwareSampler",
            "compute_sampler_state": ".sampler.compute_sampler_state",
            "register_dataset_reader": ".storage.register_dataset_reader",
            "StreamingLeRobotDataset": ".streaming_dataset.StreamingLeRobotDataset",
            "DEFAULT_EPISODES_PATH": ".utils.DEFAULT_EPISODES_PATH",
            "create_lerobot_dataset_card": ".utils.create_lerobot_dataset_card",
            "resolve_episode_indices": ".utils.resolve_episode_indices",
            "VideoEncodingManager": ".video_utils.VideoEncodingManager",
        },
    )

# NOTE: Low-level I/O functions (cast_stats_to_numpy, get_parquet_file_size_in_mb, etc.)
# and legacy migration constants are intentionally NOT re-exported here.
# Import directly: ``from lerobot.datasets.io_utils import ...``

__all__ = [
    "BaseDatasetReader",
    "CODEBASE_VERSION",
    "DEFAULT_EPISODES_PATH",
    "DEFAULT_QUANTILES",
    "EVENT_ONLY_STYLES",
    "EpisodeAwareSampler",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
    "PERSISTENT_STYLES",
    "STYLE_REGISTRY",
    "StreamingLeRobotDataset",
    "VideoEncodingManager",
    "register_dataset_reader",
    "check_video_encoder_parameters_pyav",
    "detect_available_encoders_pyav",
    "add_features",
    "aggregate_datasets",
    "aggregate_pipeline_dataset_features",
    "aggregate_stats",
    "convert_image_to_video_dataset",
    "create_initial_features",
    "compute_sampler_state",
    "create_lerobot_dataset_card",
    "column_for_style",
    "delete_episodes",
    "get_feature_stats",
    "load_episodes",
    "make_dataset",
    "make_train_eval_datasets",
    "merge_datasets",
    "modify_features",
    "modify_tasks",
    "recompute_stats",
    "reencode_dataset",
    "remove_feature",
    "resolve_delta_timestamps",
    "resolve_episode_indices",
    "safe_stop_image_writer",
    "split_dataset",
    "write_stats",
]
