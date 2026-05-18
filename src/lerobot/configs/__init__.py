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

NOTE: TrainPipelineConfig, EvalPipelineConfig, and TrainRLServerPipelineConfig
are intentionally NOT re-exported here to avoid circular dependencies
(they import lerobot.envs and lerobot.policies at module level).
Import them directly: ``from lerobot.configs.train import TrainPipelineConfig``
"""

from .dataset import DatasetRecordConfig
from .default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .types import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
    RTCAttentionSchedule,
)
from .video import (
    VALID_VIDEO_CODECS,
    VIDEO_ENCODER_INFO_KEYS,
    VideoEncoderConfig,
    camera_encoder_defaults,
)

__all__ = [
    # Types
    "FeatureType",
    "NormalizationMode",
    "PipelineFeatureType",
    "PolicyFeature",
    "RTCAttentionSchedule",
    # Config classes
    "DatasetRecordConfig",
    "DatasetConfig",
    "EvalConfig",
    "PeftConfig",
    "PreTrainedConfig",
    "WandBConfig",
    "VideoEncoderConfig",
    # Defaults
    "camera_encoder_defaults",
    # Constants
    "VALID_VIDEO_CODECS",
    "VIDEO_ENCODER_INFO_KEYS",
]
