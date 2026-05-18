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
"""Stdin REPL event collector for the SmolVLA2 runtime.

Reads non-blocking stdin lines, classifies each one heuristically:

  "stop" / "quit" / "exit"               → state["stop"] = True
  "/action" / "/pause"                    → set state["mode"]
  ends with "?"                           → user_vqa_query event
  starts with "task:" or first line       → set runtime task
  anything else                           → user_interjection event

Plugged into the runtime via ``event_collector=StdinReader().poll``.

Note: the shipped CLI (``lerobot-smolvla2-runtime``) drives stdin
directly in its REPL / autonomous loops and does *not* wire this
collector; it's kept as the documented embedding hook and for tests.
"""

from __future__ import annotations

import select
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StdinReader:
    """Non-blocking stdin line collector for the runtime loop."""

    prompt: str = "> "
    _seen_first_line: bool = field(default=False, init=False)
    _prompted: bool = field(default=False, init=False)

    def poll(self, state: dict[str, Any]) -> None:
        """Drain pending stdin lines into runtime events."""
        # Print the input prompt once on every fresh tick if we don't
        # already have a pending line; matches the expected REPL feel.
        if not self._prompted:
            print(self.prompt, end="", flush=True)
            self._prompted = True

        # ``select`` with timeout=0 makes this non-blocking. Only works
        # for actual TTY / pipe stdins; CI / scripted runs hit EOF.
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except (ValueError, OSError):
            return
        if not ready:
            return

        line = sys.stdin.readline()
        if not line:  # EOF
            state["stop"] = True
            return
        line = line.strip()
        self._prompted = False  # we'll re-prompt next tick
        if not line:
            return

        lower = line.lower()
        if lower in {"stop", "quit", "exit"}:
            state["stop"] = True
            return

        # Slash commands flip the run mode. ``/pause`` stops the action
        # loop (the action steps gate on ``state["mode"]``); ``/action``
        # resumes it.
        if lower.split(" ", 1)[0] in {"/action", "/act", "/run"}:
            state["mode"] = "action"
            return
        if lower in {"/pause", "/p"}:
            state["mode"] = "paused"
            queue = state.get("action_queue")
            if hasattr(queue, "clear"):
                queue.clear()
            return

        # First non-control line sets the task if no task is active.
        if not state.get("task"):
            task = line[5:].strip() if lower.startswith("task:") else line
            state["task"] = task
            print(f"[smolvla2] Task: {task}", flush=True)
            self._seen_first_line = True
            return

        # Question → VQA; statement → interjection.
        if lower.endswith("?"):
            state["recent_vqa_query"] = line
            state.setdefault("events_this_tick", []).append("user_vqa_query")
        else:
            state["recent_interjection"] = line
            state.setdefault("events_this_tick", []).append("user_interjection")
