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

"""Robot rollout orchestration with pluggable strategies and inference backends.

Strategies own the control loop and use ``ctx.policy.inference`` for local or remote inference.
Remote contexts have no local policy or policy processors. Interactive sessions add text I/O over
the lifecycle controller; strategies remain usable non-interactively. Shared contracts and local
backends live in :mod:`lerobot.inference`; remote execution lives in :mod:`lerobot.remote_inference`.
"""

from lerobot.utils.import_utils import require_package

require_package("datasets", extra="dataset")

from lerobot.inference import (
    InferenceEngine,
    InferenceEngineConfig,
    QueryAnswer,
    QueryKind,
    RTCInferenceConfig,
    RTCInferenceEngine,
    SyncInferenceConfig,
    SyncInferenceEngine,
    create_inference_engine,
)

from .configs import (
    BaseStrategyConfig,
    DAggerKeyboardConfig,
    DAggerPedalConfig,
    DAggerStrategyConfig,
    EpisodicStrategyConfig,
    HighlightStrategyConfig,
    RolloutConfig,
    RolloutStrategyConfig,
    SentryStrategyConfig,
)
from .context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RuntimeContext,
    build_rollout_context,
)
from .controller import (
    AskResult,
    LinkedEvent,
    RolloutController,
    RolloutEvent,
)
from .interactive import InteractiveSession
from .strategies import (
    BaseStrategy,
    DAggerStrategy,
    EpisodicStrategy,
    HighlightStrategy,
    RolloutStrategy,
    SentryStrategy,
    create_strategy,
)

__all__ = [
    "AskResult",
    "BaseStrategy",
    "BaseStrategyConfig",
    "DAggerKeyboardConfig",
    "DAggerPedalConfig",
    "DAggerStrategy",
    "DAggerStrategyConfig",
    "DatasetContext",
    "EpisodicStrategy",
    "EpisodicStrategyConfig",
    "HardwareContext",
    "HighlightStrategy",
    "HighlightStrategyConfig",
    "InferenceEngine",
    "InferenceEngineConfig",
    "InteractiveSession",
    "LinkedEvent",
    "PolicyContext",
    "ProcessorContext",
    "QueryAnswer",
    "QueryKind",
    "RTCInferenceConfig",
    "RTCInferenceEngine",
    "RolloutConfig",
    "RolloutContext",
    "RolloutController",
    "RolloutEvent",
    "RolloutStrategy",
    "RolloutStrategyConfig",
    "RuntimeContext",
    "SentryStrategy",
    "SentryStrategyConfig",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "build_rollout_context",
    "create_inference_engine",
    "create_strategy",
]
