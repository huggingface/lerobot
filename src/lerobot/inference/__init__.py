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

"""Policy execution contracts, scheduling and synchronous/asynchronous backends.

Contracts are safe to import while policy classes are being defined. Backends and
policy runners load on demand: they depend on those same policy definitions, and
remote-only dependencies must not be required by ordinary local inference.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .base import InferenceEngine, InferenceRobot, PolicyQuery, QueryAnswer
from .contracts import (
    ActionChunk,
    ActionProvenance,
    ActionSource,
    ChunkPolicySpec,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
    QueryKind,
)

if TYPE_CHECKING:
    from .execution import ChunkRequest, ChunkRuntime, estimate_delay, trained_overlap_valid
    from .factory import (
        InferenceEngineConfig,
        RemoteInferenceConfig,
        RTCInferenceConfig,
        SyncInferenceConfig,
        create_inference_engine,
    )
    from .policy_runner import PolicyRunner
    from .prediction import ChunkPrediction, chunk_inference_context, predict_chunk
    from .remote import RemoteInferenceEngine
    from .rtc import RTCInferenceEngine, supports_rtc_inference
    from .sync import SyncInferenceEngine

_LAZY_EXPORTS = {
    "ChunkRequest": "execution",
    "ChunkRuntime": "execution",
    "estimate_delay": "execution",
    "trained_overlap_valid": "execution",
    "InferenceEngineConfig": "factory",
    "RemoteInferenceConfig": "factory",
    "RTCInferenceConfig": "factory",
    "SyncInferenceConfig": "factory",
    "create_inference_engine": "factory",
    "PolicyRunner": "policy_runner",
    "ChunkPrediction": "prediction",
    "chunk_inference_context": "prediction",
    "predict_chunk": "prediction",
    "RemoteInferenceEngine": "remote",
    "RTCInferenceEngine": "rtc",
    "supports_rtc_inference": "rtc",
    "SyncInferenceEngine": "sync",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ActionChunk",
    "ActionProvenance",
    "ActionSource",
    "ChunkPolicySpec",
    "ChunkPrediction",
    "ChunkRequest",
    "ChunkRuntime",
    "ExecutionMode",
    "FeatureSpec",
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
    "RemoteInferenceEngine",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "chunk_inference_context",
    "create_inference_engine",
    "estimate_delay",
    "predict_chunk",
    "supports_rtc_inference",
    "trained_overlap_valid",
]
