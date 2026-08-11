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

Architecture
------------
Interactive rollouts split across four components, each with a distinct
responsibility, lifetime, and thread:

- :class:`InferenceEngine` — owns the policy and the only thread allowed to
  touch it (observation -> action), plus every piece of state that thread
  must read safely: the task holder and the text-query/autosteer channel.
  Lives for the whole session; runs on the control thread (sync) or its own
  background thread (RTC).
- :class:`RolloutStrategy` — the real-time tick loop (obs -> action ->
  robot, optionally recording).  One ``run()`` call per segment, on the
  control thread.
- :class:`RolloutController` — the lifecycle state machine: serializes
  commands under one lock, runs segments, emits events.  Whole session;
  commands arrive from any thread, ``serve()`` blocks the control thread.
- :class:`InteractiveSession` + stdin listener — a replaceable text I/O
  adapter over the controller, with no control logic of its own.  Whole
  session, on the stdin-listener thread.

Calls point downward only — session -> controller -> {strategy, engine},
strategy -> engine; nothing in ``strategies/`` or ``inference/`` references
the controller.  The engine is called from two places by design: the
controller makes *control-plane* calls (``set_task``, ``ask``, ``pause``,
polling ``failed``), while the strategy makes *data-plane* calls
(``get_action``, ``pump_query``, ``resume``); ``ctx`` is the wiring harness
that lets them share the engine without referencing each other.  Upward
communication is deliberately narrow: the engine's answer-observer callback
(query answers, always fired on the serve thread from ``pump_query``) and
the :class:`LinkedEvent` installed as ``ctx.runtime.shutdown_event`` — the
controller sets its local flag to end a segment, strategies poll it every
tick, and a fatally failing engine sets it too, with ``engine.failed``
(polled by ``serve()`` and by segment startup) as the authoritative failure
signal.

When do things happen: a command never *does* anything on the caller's
thread — it records intent under the controller lock and returns.  The
control/policy threads observe that intent at defined points: segment
boundaries for start/reset/stop, the next inference for a task switch, the
next ``pump_query`` for text queries.  Read every controller method as
"record intent", and ``serve()`` as the only place intent becomes motion.
"""

from lerobot.utils.import_utils import require_package

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
from .inference import (
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
