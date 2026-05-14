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

from lerobot.utils.import_utils import require_package

require_package("datasets", extra="dataset")
require_package("av", extra="dataset")

from .aggregate import aggregate_datasets
from .compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .dataset_tools import (
    add_features,
    convert_image_to_video_dataset,
    delete_episodes,
    merge_datasets,
    modify_features,
    modify_tasks,
    recompute_stats,
    remove_feature,
    split_dataset,
)
from .factory import make_dataset, resolve_delta_timestamps
from .image_writer import safe_stop_image_writer
from .io_utils import load_episodes, write_stats
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from .pyav_utils import check_video_encoder_parameters_pyav, detect_available_encoders_pyav
from .sampler import EpisodeAwareSampler
from .streaming_dataset import StreamingLeRobotDataset
from .utils import DEFAULT_EPISODES_PATH, create_lerobot_dataset_card
from .video_utils import VideoEncodingManager

# NOTE: Low-level I/O functions (cast_stats_to_numpy, get_parquet_file_size_in_mb, etc.)
# and legacy migration constants are intentionally NOT re-exported here.
# Import directly: ``from lerobot.datasets.io_utils import ...``

__all__ = [
    "CODEBASE_VERSION",
    "DEFAULT_EPISODES_PATH",
    "DEFAULT_QUANTILES",
    "EpisodeAwareSampler",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
    "StreamingLeRobotDataset",
    "VideoEncodingManager",
    "check_video_encoder_parameters_pyav",
    "detect_available_encoders_pyav",
    "add_features",
    "aggregate_datasets",
    "aggregate_pipeline_dataset_features",
    "aggregate_stats",
    "convert_image_to_video_dataset",
    "create_initial_features",
    "create_lerobot_dataset_card",
    "delete_episodes",
    "get_feature_stats",
    "load_episodes",
    "make_dataset",
    "merge_datasets",
    "modify_features",
    "modify_tasks",
    "recompute_stats",
    "remove_feature",
    "resolve_delta_timestamps",
    "safe_stop_image_writer",
    "split_dataset",
    "write_stats",
]
