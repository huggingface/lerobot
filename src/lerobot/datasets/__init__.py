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
from .compute_stats import (
    DEFAULT_QUANTILES,
    RunningQuantileStats,
    aggregate_stats,
    get_feature_stats,
    sample_indices,
)
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
from .io_utils import (
    cast_stats_to_numpy,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    load_episodes,
    load_info,
    load_tasks,
    write_episodes,
    write_info,
    write_stats,
    write_table_one_row_group_per_episode,
    write_tasks,
)
from .language import (
    DEFAULT_TOOLS,
    EVENT_ONLY_STYLES,
    PERSISTENT_STYLES,
    SAY_TOOL_SCHEMA,
    STYLE_REGISTRY,
    column_for_style,
    is_view_dependent_style,
    language_feature_info,
    validate_camera_field,
)
from .language_render import render_sample
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from .pyav_utils import check_video_encoder_parameters_pyav, detect_available_encoders_pyav
from .recipe import TrainingRecipe, render_message_turns, resolve_recipe_override
from .sampler import EpisodeAwareSampler, compute_sampler_state
from .storage import register_dataset_reader
from .streaming_dataset import StreamingLeRobotDataset
from .streaming_sidecar import (
    SIDECAR_REPO_BRANCH,
    build_mp4_sidecar,
    make_sidecar_spec,
    published_sidecar_url,
)
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    DatasetInfo,
    create_lerobot_dataset_card,
    resolve_episode_indices,
    update_chunk_file_indices,
)
from .video_utils import (
    VideoEncodingManager,
    concatenate_video_files,
    decode_video_frames,
    get_video_duration_in_s,
    reencode_video,
)

__all__ = [
    "BaseDatasetReader",
    "CODEBASE_VERSION",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_DATA_FILE_SIZE_IN_MB",
    "DEFAULT_DATA_PATH",
    "DEFAULT_EPISODES_PATH",
    "DEFAULT_QUANTILES",
    "DEFAULT_TASKS_PATH",
    "DEFAULT_TOOLS",
    "DEFAULT_VIDEO_FILE_SIZE_IN_MB",
    "DEFAULT_VIDEO_PATH",
    "DatasetInfo",
    "EVENT_ONLY_STYLES",
    "EpisodeAwareSampler",
    "INFO_PATH",
    "LEGACY_EPISODES_PATH",
    "LEGACY_EPISODES_STATS_PATH",
    "LEGACY_TASKS_PATH",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
    "PERSISTENT_STYLES",
    "RunningQuantileStats",
    "SAY_TOOL_SCHEMA",
    "SIDECAR_REPO_BRANCH",
    "STYLE_REGISTRY",
    "StreamingLeRobotDataset",
    "TrainingRecipe",
    "VideoEncodingManager",
    "add_features",
    "aggregate_datasets",
    "aggregate_pipeline_dataset_features",
    "aggregate_stats",
    "build_mp4_sidecar",
    "cast_stats_to_numpy",
    "check_video_encoder_parameters_pyav",
    "column_for_style",
    "compute_sampler_state",
    "concatenate_video_files",
    "convert_image_to_video_dataset",
    "create_initial_features",
    "create_lerobot_dataset_card",
    "decode_video_frames",
    "delete_episodes",
    "detect_available_encoders_pyav",
    "get_feature_stats",
    "get_file_size_in_mb",
    "get_parquet_file_size_in_mb",
    "get_parquet_num_frames",
    "get_video_duration_in_s",
    "is_view_dependent_style",
    "language_feature_info",
    "load_episodes",
    "load_info",
    "load_tasks",
    "make_dataset",
    "make_sidecar_spec",
    "make_train_eval_datasets",
    "merge_datasets",
    "modify_features",
    "modify_tasks",
    "published_sidecar_url",
    "recompute_stats",
    "reencode_dataset",
    "reencode_video",
    "register_dataset_reader",
    "remove_feature",
    "render_message_turns",
    "render_sample",
    "resolve_delta_timestamps",
    "resolve_episode_indices",
    "resolve_recipe_override",
    "safe_stop_image_writer",
    "sample_indices",
    "split_dataset",
    "update_chunk_file_indices",
    "validate_camera_field",
    "write_episodes",
    "write_info",
    "write_stats",
    "write_table_one_row_group_per_episode",
    "write_tasks",
]
