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

"""
Public API for lerobot configuration types and base config classes.
"""

from . import parser  # registers the draccus codec of torch.dtype
from .dataset import DatasetRecordConfig
from .default import DatasetConfig, EvalConfig, JobConfig, PeftConfig, TrackioConfig
from .parallelism import ParallelismConfig, world_size_from_env
from .policies import PreTrainedConfig
from .rewards import RewardModelConfig
from .types import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
    RTCAttentionSchedule,
    StreamingSamplingStrategy,
)
from .video import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_PIX_FMT,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_UNIT,
    DEFAULT_DEPTH_USE_LOG,
    DEPTH_ENCODER_INFO_FIELD_NAMES,
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    DEPTH_QMAX,
    VALID_VIDEO_CODECS,
    VIDEO_ENCODER_INFO_KEYS,
    DepthEncoderConfig,
    RGBEncoderConfig,
    VideoEncoderConfig,
    depth_encoder_defaults,
    encoder_config_from_video_info,
    infer_depth_unit,
    is_depth_map,
    rgb_encoder_defaults,
)

# NOTE: TrainPipelineConfig, CheckpointFormat and TRAIN_CONFIG_NAME (configs.train) and EvalPipelineConfig
# (configs.eval) are intentionally NOT re-exported here to avoid a cycle (those modules import lerobot.envs
# and lerobot.policies).
# Import from submodules: ``from lerobot.configs.train import TrainPipelineConfig``

__all__ = [
    # Types
    "FeatureType",
    "NormalizationMode",
    "PipelineFeatureType",
    "PolicyFeature",
    "RTCAttentionSchedule",
    "StreamingSamplingStrategy",
    # Config classes
    "DatasetRecordConfig",
    "DatasetConfig",
    "EvalConfig",
    "JobConfig",
    "ParallelismConfig",
    "PeftConfig",
    "PreTrainedConfig",
    "RewardModelConfig",
    "TrackioConfig",
    "VideoEncoderConfig",
    "RGBEncoderConfig",
    "DepthEncoderConfig",
    # Defaults
    "rgb_encoder_defaults",
    "depth_encoder_defaults",
    # Factories
    "encoder_config_from_video_info",
    "infer_depth_unit",
    "is_depth_map",
    "world_size_from_env",
    # Constants
    "DEFAULT_DEPTH_MAX",
    "DEFAULT_DEPTH_MIN",
    "DEFAULT_DEPTH_PIX_FMT",
    "DEFAULT_DEPTH_SHIFT",
    "DEFAULT_DEPTH_UNIT",
    "DEFAULT_DEPTH_USE_LOG",
    "DEPTH_ENCODER_INFO_FIELD_NAMES",
    "DEPTH_METER_UNIT",
    "DEPTH_MILLIMETER_UNIT",
    "DEPTH_QMAX",
    "VALID_VIDEO_CODECS",
    "VIDEO_ENCODER_INFO_KEYS",
    # Modules
    "parser",
]
