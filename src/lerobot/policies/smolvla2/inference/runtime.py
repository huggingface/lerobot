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
"""SmolVLA2 runtime loop.

Threads the multi-rate inference pipeline together with a stdin REPL
event collector, drives ticks through :class:`TickClock`, and prints
state-change updates to the user.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .runtime_state import initial_runtime_state, push_log
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
from .triggers import HzTrigger, TickClock

logger = logging.getLogger(__name__)


@dataclass
class SmolVLA2Runtime:
    """Compose the inference pipeline and drive it tick-by-tick."""

    policy: Any
    tools: dict[str, Any] = field(default_factory=dict)
    """Name → tool-instance dict, e.g. ``{"say": SayTool(...)}``. Read
    from :func:`lerobot.tools.get_tools(meta)` when wiring the
    runtime."""
    observation_provider: Callable[[], dict | None] | None = None
    """Closure returning the current preprocessed observation batch.
    ``None`` for dry-run / language-only sessions."""
    robot_executor: Callable[[Any], None] | None = None
    """Closure that takes one action chunk and forwards it to the
    robot. ``None`` for dry-run."""
    event_collector: Callable[[dict], None] | None = None
    """Per-tick hook that polls external sources (stdin, network) and
    appends event names to ``state["events_this_tick"]``."""
    chunk_hz: float = 4.0
    ctrl_hz: float = 50.0
    high_level_hz: float = 1.0
    max_rate_hz: float = 50.0

    pipeline: list[InferenceStep] = field(init=False)
    state: dict[str, Any] = field(init=False)
    _stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Pipeline order matters. Both ``HighLevelSubtaskFwd`` and
        # ``LowLevelForward`` are gated on "action queue is empty" so
        # the slow LLM call (select_message) doesn't starve dispatch.
        # If LowLevelForward runs first, it refills the queue and the
        # high-level step never sees ``queue == 0`` afterwards.
        #
        # Order is therefore: high-level steps that read state (subtask,
        # memory, interjection, vqa) → low-level chunk refresh → action
        # dispatch → tool dispatch. So on an empty-queue tick the
        # subtask refreshes first, the new subtask string flows into
        # the next chunk's prompt, and DispatchAction drains.
        self.pipeline = [
            HighLevelSubtaskFwd(
                trigger=HzTrigger(self.high_level_hz),
                policy=self.policy,
                observation_provider=self.observation_provider,
            ),
            MemoryUpdateFwd(
                policy=self.policy,
                observation_provider=self.observation_provider,
            ),
            UserInterjectionFwd(
                policy=self.policy,
                observation_provider=self.observation_provider,
            ),
            AskVQAFwd(
                policy=self.policy,
                observation_provider=self.observation_provider,
            ),
            LowLevelForward(
                trigger=HzTrigger(self.chunk_hz),
                policy=self.policy,
                observation_provider=self.observation_provider,
            ),
            DispatchAction(
                trigger=HzTrigger(self.ctrl_hz),
                robot_executor=self.robot_executor,
            ),
            DispatchToolCalls(tools=self.tools),
        ]
        self.state = initial_runtime_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_task(self, task: str) -> None:
        """Set or replace the active task. Logged for the REPL."""
        self.state["task"] = task
        push_log(self.state, f"Task: {task}")

    def stop(self) -> None:
        self._stop = True

    def run(self, *, max_ticks: int | None = None) -> None:
        """Main loop. Returns when ``stop()`` is called or after
        ``max_ticks`` ticks (useful for tests / dry-run)."""
        clock = TickClock(max_rate_hz=self.max_rate_hz)
        while not self._stop:
            tick = clock.advance()
            self.state["_tick"] = tick
            self.state["events_this_tick"] = []
            self.state["log_lines"] = []

            if self.event_collector is not None:
                self.event_collector(self.state)
            if self.state.get("stop"):
                self._stop = True
                break

            for step in self.pipeline:
                self.state = step(self.state)

            self._flush_logs()
            if max_ticks is not None and tick.index >= max_ticks:
                break

        self._on_shutdown()

    # ------------------------------------------------------------------
    # REPL helper: drive one full pipeline pass and return its logs
    # ------------------------------------------------------------------

    def step_once(self) -> list[str]:
        """Run one tick of the pipeline and return the log lines.

        Used by the interactive REPL: instead of a background thread,
        the CLI drives ticks synchronously after each user input. Logs
        are returned (not printed) so the caller can route them into
        the rich-Live chat scrollback.
        """
        from .triggers import Tick  # noqa: PLC0415

        # Synthesize a tick. We don't need the real wall-clock pacing
        # here — the REPL drives the runtime, not vice versa — but
        # ``HzTrigger`` uses ``tick.monotonic_seconds`` to gate, so we
        # bump it generously so every Hz-triggered step considers
        # itself due.
        import time as _time  # noqa: PLC0415

        prev_index = self.state.get("_tick").index if isinstance(self.state.get("_tick"), Tick) else 0
        self.state["_tick"] = Tick(index=prev_index + 1, monotonic_seconds=_time.monotonic())
        self.state["log_lines"] = []
        # ``events_this_tick`` is set up by the caller before
        # ``step_once`` (the REPL pushes user-driven events first).
        self.state.setdefault("events_this_tick", [])

        for step in self.pipeline:
            self.state = step(self.state)

        return list(self.state.get("log_lines") or [])

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _flush_logs(self) -> None:
        for line in self.state.get("log_lines") or []:
            print(f"[smolvla2] {line}", flush=True)

    def _on_shutdown(self) -> None:
        # Drain any queued action chunks safely.
        queue = self.state.get("action_queue")
        if isinstance(queue, deque):
            queue.clear()
        print("[smolvla2] runtime stopped", flush=True)
