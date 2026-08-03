# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Inference engine package — backend-agnostic action production.

Concrete backends (``sync``, ``rtc``, ...) expose the same small interface so
rollout strategies never branch on which backend is in use.
"""

from .base import InferenceEngine
from .factory import (
    InferenceEngineConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

__all__ = [
    "InferenceEngine",
    "InferenceEngineConfig",
    "RTCInferenceConfig",
    "RTCInferenceEngine",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "create_inference_engine",
]
