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
"""Runtime state passed between inference steps each tick.

The runtime threads a single dict through the pipeline; this module
documents the shape and provides factories. We use a plain ``dict``
rather than a frozen dataclass because steps freely add and remove
keys (``events_this_tick``, ``messages_pending``, ``tool_calls_pending``,
…) and dataclass field churn would just get in the way.

Stable keys (read by multiple steps):

  task          str             the current top-level task
  current_plan  str | None      latest plan emitted by the planner
  current_subtask str | None    latest subtask the policy is executing
  current_memory str | None     latest compressed memory
  recent_interjection str | None  most recent user interjection text (consumed)

  action_queue  collections.deque[Tensor]  pending action chunks
  tool_calls_pending list[dict]  parsed but not-yet-dispatched tool calls

  events_this_tick list[str]    triggers consumed this tick
  _tick         Tick            current tick (set by the loop)

  mode          str             "action" (run the robot) | "vlm" (VQA only,
                                 action loop paused)

  log_lines     list[str]       human-readable status lines printed each tick
"""

from __future__ import annotations

from collections import deque
from typing import Any


def initial_runtime_state(task: str | None = None) -> dict[str, Any]:
    """Build a fresh runtime state dict with sensible defaults."""
    return {
        "task": task,
        "current_plan": None,
        "current_subtask": None,
        "current_memory": None,
        "recent_interjection": None,
        "action_queue": deque(),
        "tool_calls_pending": [],
        "events_this_tick": [],
        "log_lines": [],
        "mode": "action",
        "stop": False,
    }


def take_event(state: dict[str, Any], event_name: str) -> bool:
    """Pop ``event_name`` from ``events_this_tick`` if present.

    Steps that consume an event call this so the same event doesn't
    re-fire on a sibling step within the same tick.
    """
    events: list[str] = state.get("events_this_tick") or []
    if event_name in events:
        events.remove(event_name)
        return True
    return False


def push_log(state: dict[str, Any], line: str) -> None:
    """Append ``line`` to the per-tick log buffer; the runtime prints
    it at the end of the tick."""
    state.setdefault("log_lines", []).append(line)


def set_if_changed(state: dict[str, Any], key: str, value: Any, label: str | None = None) -> bool:
    """Update ``state[key]`` and log a diff line if the value changed.

    Returns ``True`` if the value actually changed.
    """
    prev = state.get(key)
    if prev == value:
        return False
    state[key] = value
    if label is not None:
        push_log(state, f"  {label}: {value}")
    return True
