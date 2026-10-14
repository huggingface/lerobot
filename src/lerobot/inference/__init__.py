# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Policy execution contracts, scheduling and synchronous/asynchronous backends."""

from .base import InferenceEngine, InferenceRobot, PolicyQuery, QueryAnswer
from .chunk_settings import chunk_settings
from .contracts import (
    DEFAULT_MAX_TEXT_INPUT_CHARS,
    DEFAULT_MAX_TEXT_OUTPUT_CHARS,
    ActionChunk,
    ActionProvenance,
    ChunkPolicySpec,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
    QueryKind,
)
from .dispatch import commit_action, dispatch_permitted, dispatch_tick, next_interpolated_action
from .execution import ChunkRequest, ChunkRuntime, HoldState, estimate_delay, trained_overlap_valid
from .factory import (
    InferenceEngineConfig,
    RemoteInferenceConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .policy_runner import PolicyRunner
from .prediction import ChunkPrediction, chunk_inference_context, predict_chunk
from .rtc import RTCInferenceEngine, supports_rtc_inference, validate_local_chunk_policy
from .sync import SyncInferenceEngine

__all__ = [
    "DEFAULT_MAX_TEXT_INPUT_CHARS",
    "DEFAULT_MAX_TEXT_OUTPUT_CHARS",
    "ActionChunk",
    "ActionProvenance",
    "ChunkPolicySpec",
    "ChunkPrediction",
    "ChunkRequest",
    "ChunkRuntime",
    "ExecutionMode",
    "FeatureSpec",
    "HoldState",
    "InferenceEngine",
    "InferenceEngineConfig",
    "InferenceRobot",
    "ObservationSnapshot",
    "PolicyCapabilities",
    "PolicyQuery",
    "PolicyRunner",
    "QueryAnswer",
    "QueryKind",
    "RTCInferenceConfig",
    "RTCInferenceEngine",
    "RemoteInferenceConfig",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "chunk_inference_context",
    "chunk_settings",
    "commit_action",
    "create_inference_engine",
    "dispatch_permitted",
    "dispatch_tick",
    "estimate_delay",
    "next_interpolated_action",
    "predict_chunk",
    "supports_rtc_inference",
    "trained_overlap_valid",
    "validate_local_chunk_policy",
]
