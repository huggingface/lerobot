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

"""PI052 compatibility wrapper for the generic language-conditioned runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lerobot.policies.language_conditioned import (
    LanguageConditionedRuntime,
    RuntimeState,
    Tick,
    TickClock,
    ToolCall,
    VQAResult,
)

from .pi052_adapter import PI052PolicyAdapter


class PI052Runtime(LanguageConditionedRuntime):
    """Backwards-compatible PI052 runtime constructor."""

    def __init__(
        self,
        policy: Any,
        *,
        tools: dict[str, Any] | None = None,
        observation_provider: Callable[[], dict | None] | None = None,
        robot_executor: Callable[[Any], None] | None = None,
        event_collector: Callable[[RuntimeState], None] | None = None,
        chunk_hz: float = 4.0,
        ctrl_hz: float = 50.0,
        high_level_hz: float = 1.0,
        max_rate_hz: float = 50.0,
    ) -> None:
        super().__init__(
            policy_adapter=policy if isinstance(policy, PI052PolicyAdapter) else PI052PolicyAdapter(policy),
            observation_provider=observation_provider,
            action_executor=robot_executor,
            tools=tools or {},
            event_collector=event_collector,
            chunk_hz=chunk_hz,
            ctrl_hz=ctrl_hz,
            high_level_hz=high_level_hz,
            max_rate_hz=max_rate_hz,
        )


__all__ = [
    "LanguageConditionedRuntime",
    "PI052PolicyAdapter",
    "PI052Runtime",
    "RuntimeState",
    "Tick",
    "TickClock",
    "ToolCall",
    "VQAResult",
]
