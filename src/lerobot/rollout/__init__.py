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

"""Policy deployment engine with pluggable rollout strategies.

An interactive rollout is built from four components: :class:`InferenceEngine` (owns the policy),
:class:`RolloutStrategy` (the real-time tick loop), :class:`RolloutController` (lifecycle state machine)
and :class:`InteractiveSession` (text I/O over the controller).  Calls point downward only — session ->
controller -> {strategy, engine}, strategy -> engine — and nothing under ``strategies/`` or ``inference/``
references the controller, so strategies stay usable non-interactively.  Controller commands only record
intent under the controller lock; ``serve()`` is the only place intent becomes motion.
"""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports, require_package

require_package("datasets", extra="dataset")

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
from .controller import (
    AskResult,
    LinkedEvent,
    RolloutController,
    RolloutEvent,
)
from .inference import (
    InferenceEngine,
    InferenceEngineConfig,
    QueryAnswer,
    QueryKind,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .interactive import InteractiveSession

# These import torch, the dataset stack and the processors, so each is imported the first time it is used.
if TYPE_CHECKING:
    from .context import (
        DatasetContext,
        HardwareContext,
        PolicyContext,
        ProcessorContext,
        RolloutContext,
        RuntimeContext,
        build_rollout_context,
    )
    from .inference import RTCInferenceEngine, SyncInferenceEngine
    from .strategies import (
        BaseStrategy,
        DAggerStrategy,
        EpisodicStrategy,
        HighlightStrategy,
        RolloutStrategy,
        SentryStrategy,
        create_strategy,
    )
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "DatasetContext": ".context.DatasetContext",
            "HardwareContext": ".context.HardwareContext",
            "PolicyContext": ".context.PolicyContext",
            "ProcessorContext": ".context.ProcessorContext",
            "RolloutContext": ".context.RolloutContext",
            "RuntimeContext": ".context.RuntimeContext",
            "build_rollout_context": ".context.build_rollout_context",
            "RTCInferenceEngine": ".inference.RTCInferenceEngine",
            "SyncInferenceEngine": ".inference.SyncInferenceEngine",
            "BaseStrategy": ".strategies.BaseStrategy",
            "DAggerStrategy": ".strategies.DAggerStrategy",
            "EpisodicStrategy": ".strategies.EpisodicStrategy",
            "HighlightStrategy": ".strategies.HighlightStrategy",
            "RolloutStrategy": ".strategies.RolloutStrategy",
            "SentryStrategy": ".strategies.SentryStrategy",
            "create_strategy": ".strategies.create_strategy",
        },
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
