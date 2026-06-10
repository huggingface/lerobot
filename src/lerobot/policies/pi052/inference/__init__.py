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
"""PI052 inference / runtime orchestration.

Multi-rate runtime that mirrors the recipe-time training shape:

  low_level_execution        → LowLevelForward + DispatchAction (high Hz)
  high_level_subtask         → HighLevelSubtaskFwd (~1 Hz)
  memory_update              → MemoryUpdateFwd (event: subtask_change)
  user_interjection_response → UserInterjectionFwd (event: stdin)
  ask_vqa_*                  → AskVQAFwd (event: stdin question)
  speech tool calls          → DispatchToolCalls (event: tool_call_pending)

The CLI ``lerobot-pi052-runtime`` builds a ``PI052Runtime`` and calls
``run()``.
"""

from .repl import StdinReader
from .runtime import PI052Runtime
from .runtime_state import initial_runtime_state, push_log, set_if_changed, take_event
from .steps import (
    AskVQAFwd,
    DispatchAction,
    DispatchToolCalls,
    HighLevelSubtaskFwd,
    InferenceStep,
    LowLevelForward,
    MemoryUpdateFwd,
    UserInterjectionFwd,
)
from .triggers import EventTrigger, HzTrigger, Tick, TickClock, Trigger
from .ui import make_state_panel, print_robot_lines, print_user_line

__all__ = [
    # runtime
    "PI052Runtime",
    "StdinReader",
    # state helpers
    "initial_runtime_state",
    "push_log",
    "set_if_changed",
    "take_event",
    # triggers
    "Trigger",
    "Tick",
    "TickClock",
    "HzTrigger",
    "EventTrigger",
    # steps
    "InferenceStep",
    "LowLevelForward",
    "DispatchAction",
    "HighLevelSubtaskFwd",
    "MemoryUpdateFwd",
    "UserInterjectionFwd",
    "AskVQAFwd",
    "DispatchToolCalls",
    # UI
    "make_state_panel",
    "print_robot_lines",
    "print_user_line",
]
