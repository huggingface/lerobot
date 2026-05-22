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
"""Trigger primitives for SmolVLA2's multi-rate inference runtime.

Mirrors the plan's Section "Runtime orchestration": each
``InferenceStep`` is gated by a :class:`Trigger` that decides per tick
whether the step fires. Two trigger flavours cover all the cadences
the canonical recipe needs:

* :class:`HzTrigger` for periodic beats (action chunks at ~3-5 Hz,
  high-level subtask generation at ~1 Hz, action dispatch at ~50 Hz)
* :class:`EventTrigger` for one-shot reactions (subtask boundary →
  memory update; user interjection → plan refresh; user VQA query →
  vqa answer; pending tool call → dispatcher)

Triggers are stateless except for ``HzTrigger``'s last-fire timestamp.
The runtime stores the :class:`Tick` clock as ``state["_tick"]`` so
every step shares a single time source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Tick:
    """Single tick from :class:`TickClock`. Carries time references the
    runtime steps consume to gate themselves."""

    index: int
    """Monotonic counter — increments by one per tick."""

    monotonic_seconds: float
    """``time.monotonic()`` at the start of this tick."""


@dataclass
class TickClock:
    """Drives the runtime loop at up to ``max_rate_hz``.

    Sleeps just enough between :meth:`advance` calls to enforce the
    rate. With ``max_rate_hz=50`` the loop wakes ~every 20ms; the
    higher-level ``HzTrigger`` slices that timeline into sub-cadences.
    """

    max_rate_hz: float = 50.0
    _index: int = field(default=0, init=False)
    _last_seconds: float | None = field(default=None, init=False)

    def advance(self) -> Tick:
        period = 1.0 / max(self.max_rate_hz, 0.1)
        now = time.monotonic()
        if self._last_seconds is not None:
            sleep_for = (self._last_seconds + period) - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
        self._last_seconds = now
        self._index += 1
        return Tick(index=self._index, monotonic_seconds=now)


class Trigger(Protocol):
    """Decide whether the next ``InferenceStep`` should fire."""

    def should_fire(self, tick: Tick, state: dict[str, Any]) -> bool: ...


@dataclass
class HzTrigger:
    """Fire at most ``hz`` times per second.

    A step that gates further (e.g. ``HighLevelSubtaskFwd`` skipping
    when the action queue is non-empty) and wants the trigger to
    retry next tick instead of waiting a full period can call
    :meth:`rearm` from inside ``run``. Without this, a low-hz trigger
    (e.g. ``hz=0.2`` = once per 5 s) almost never coincides with the
    brief queue-empty window and the step never fires at all.
    """

    hz: float
    _last_seconds: float | None = field(default=None, init=False)

    def should_fire(self, tick: Tick, state: dict[str, Any]) -> bool:
        period = 1.0 / max(self.hz, 1e-6)
        if self._last_seconds is None or (tick.monotonic_seconds - self._last_seconds) >= period:
            self._last_seconds = tick.monotonic_seconds
            return True
        return False

    def rearm(self) -> None:
        """Mark the trigger as not having fired, so the next tick re-evaluates.

        Used by a step that decided to skip after ``should_fire`` already
        committed the firing — keeps the cadence honest without losing
        the slot.
        """
        self._last_seconds = None


@dataclass
class EventTrigger:
    """Fire when ``event_name`` is in ``state["events_this_tick"]``.

    The runtime fills ``events_this_tick`` once per tick from:

    * stdin / network input (``user_interjection``, ``user_vqa_query``,
      ``stop``)
    * internal state transitions (``subtask_change``,
      ``tool_call_pending``)

    The list is consumed (cleared at the end of the tick) so events
    fire at most once.
    """

    event_name: str

    def should_fire(self, tick: Tick, state: dict[str, Any]) -> bool:
        events: list[str] = state.get("events_this_tick") or []
        return self.event_name in events
