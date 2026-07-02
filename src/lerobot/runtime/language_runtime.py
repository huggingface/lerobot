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

"""Small reusable runtime for language-conditioned robot policies."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    """Explicit state shared by the runtime and policy adapter."""

    task: str = ""
    language_context: dict[str, str] = field(default_factory=dict)
    action_queue: deque[Any] = field(default_factory=deque)
    events: set[str] = field(default_factory=set)
    log_lines: list[str] = field(default_factory=list)
    mode: str = "action"
    stop: bool = False
    tick: Tick | None = None
    actions_dispatched: int = 0
    action_deadline: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _ALIASES = {
        "current_plan": ("language_context", "plan"),
        "current_subtask": ("language_context", "subtask"),
        "current_memory": ("language_context", "memory"),
        "events_this_tick": ("events", None),
        "_tick": ("tick", None),
    }

    def emit(self, event_name: str) -> None:
        self.events.add(event_name)

    def take_event(self, event_name: str) -> bool:
        if event_name not in self.events:
            return False
        self.events.remove(event_name)
        return True

    def log(self, line: str) -> None:
        self.log_lines.append(line)

    def set_context(self, key: str, value: str | None, *, label: str | None = None) -> bool:
        previous = self.language_context.get(key)
        if previous == value:
            return False
        if value is None:
            self.language_context.pop(key, None)
        else:
            self.language_context[key] = value
        if label is not None and value:
            self.log(f"  {label}: {value}")
        return True

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: Any = None) -> Any:
        current = self.get(key, None)
        if current is not None:
            return current
        self[key] = default
        return default

    def __getitem__(self, key: str) -> Any:
        alias = self._ALIASES.get(key)
        if alias is not None:
            target, subkey = alias
            value = getattr(self, target)
            return value if subkey is None else value.get(subkey)
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        alias = self._ALIASES.get(key)
        if alias is not None:
            target, subkey = alias
            if subkey is None:
                setattr(self, target, value)
            elif value is None:
                getattr(self, target).pop(subkey, None)
            else:
                getattr(self, target)[subkey] = value
            return
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value


class LanguageConditionedPolicyAdapter(Protocol):
    """Policy-specific bridge used by :class:`LanguageConditionedRuntime`."""

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any: ...

    def select_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str: ...


@dataclass
class Tick:
    index: int
    monotonic_seconds: float


@dataclass
class TickClock:
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


@dataclass
class _RateGate:
    hz: float
    _last_seconds: float | None = None

    def due(self, tick: Tick, *, force: bool = False) -> bool:
        if force:
            self._last_seconds = tick.monotonic_seconds
            return True
        period = 1.0 / max(self.hz, 1e-6)
        if self._last_seconds is None or tick.monotonic_seconds - self._last_seconds >= period:
            self._last_seconds = tick.monotonic_seconds
            return True
        return False

    def rearm(self) -> None:
        self._last_seconds = None


@dataclass
class LanguageConditionedRuntime:
    """Generic tick loop for language-conditioned robot policies."""

    policy_adapter: LanguageConditionedPolicyAdapter
    observation_provider: Callable[[], dict[str, Any] | None] | None = None
    action_executor: Callable[[Any], None] | None = None
    event_collector: Callable[[RuntimeState], None] | None = None
    chunk_hz: float = 4.0
    ctrl_hz: float = 50.0
    high_level_hz: float = 1.0
    max_rate_hz: float = 50.0

    state: RuntimeState = field(default_factory=RuntimeState)
    _chunk_gate: _RateGate = field(init=False)
    _ctrl_gate: _RateGate = field(init=False)
    _language_gate: _RateGate = field(init=False)
    _stop: bool = field(default=False, init=False)
    _last_dispatch_seconds: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._chunk_gate = _RateGate(self.chunk_hz)
        self._ctrl_gate = _RateGate(self.ctrl_hz)
        self._language_gate = _RateGate(self.high_level_hz)

    @property
    def policy(self) -> Any:
        return getattr(self.policy_adapter, "policy", self.policy_adapter)

    def set_task(self, task: str) -> None:
        self.state.task = task
        self.state.log(f"Task: {task}")

    def stop(self) -> None:
        self._stop = True
        self.state.stop = True

    def run(self, *, max_ticks: int | None = None) -> None:
        clock = TickClock(max_rate_hz=self.max_rate_hz)
        while not self._stop:
            tick = clock.advance()
            self._run_tick(tick)
            self._flush_logs()
            if self.state.stop:
                self._stop = True
            if max_ticks is not None and tick.index >= max_ticks:
                break
        self._on_shutdown()

    def step_once(self) -> list[str]:
        previous = self.state.tick.index if self.state.tick is not None else 0
        tick = Tick(index=previous + 1, monotonic_seconds=time.monotonic())
        self._run_tick(tick, force_rates=True)
        return list(self.state.log_lines)

    def _run_tick(self, tick: Tick, *, force_rates: bool = False) -> None:
        self.state.tick = tick
        self.state.log_lines = []
        if self.event_collector is not None:
            self.event_collector(self.state)
        self._handle_action_deadline()
        if self.state.stop:
            return
        self.maybe_update_language_state(force=force_rates)
        self.maybe_handle_user_events()
        self.maybe_enqueue_action_chunk(force=force_rates)
        self.dispatch_action(force=force_rates)
        self.state.events.clear()

    def _current_observation(self) -> dict[str, Any] | None:
        if self.observation_provider is None:
            return None
        try:
            return self.observation_provider()
        except Exception as exc:  # noqa: BLE001
            logger.debug("observation_provider failed: %s", exc)
            return None

    def maybe_update_language_state(self, *, force: bool = False) -> None:
        if self.state.mode != "action" or not self.state.task:
            return
        if self.state.action_queue:
            self._language_gate.rearm()
            return
        if self.state.tick is None or not self._language_gate.due(self.state.tick, force=force):
            return
        update = getattr(self.policy_adapter, "update_language_state", None)
        if update is None:
            return
        observation = self._current_observation()
        try:
            update(observation, self.state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("language update failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            self.state.log(f"  [warn] language update failed: {type(exc).__name__}: {exc}")

    def maybe_handle_user_events(self) -> None:
        if self.state.take_event("user_interjection"):
            self._handle_user_interjection()

    def _handle_user_interjection(self) -> None:
        text = str(self.state.extra.get("recent_interjection") or "")
        if not text:
            return
        observation = self._current_observation()
        out = self.policy_adapter.select_text("interjection", observation, self.state, user_text=text)
        if not out:
            return
        plan = getattr(self.policy_adapter, "plan_from_text", lambda value: value)(out)
        if plan:
            self.state.set_context("plan", plan, label="plan")
        self.state.extra["recent_interjection"] = None

    def maybe_enqueue_action_chunk(self, *, force: bool = False) -> None:
        if self.state.mode != "action" or not self.state.task:
            return
        if self.state.action_queue:
            return
        if self.state.tick is None or not self._chunk_gate.due(self.state.tick, force=force):
            return
        observation = self._current_observation()
        if observation is None:
            return
        try:
            chunk = self.policy_adapter.select_action(observation, self.state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("select_action failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            self.state.log(f"  [warn] select_action failed: {type(exc).__name__}: {exc}")
            return
        self._enqueue_chunk(chunk)

    def _enqueue_chunk(self, chunk: Any) -> None:
        if chunk is None:
            return
        chunk_iter = chunk[0] if getattr(chunk, "ndim", None) == 3 else chunk
        if getattr(chunk_iter, "ndim", None) == 1:
            chunk_iter = chunk_iter.unsqueeze(0)
        for step in chunk_iter:
            self.state.action_queue.append(step.unsqueeze(0) if hasattr(step, "unsqueeze") else step)
        try:
            self.state.extra["last_chunk_size"] = int(chunk_iter.shape[0])
        except Exception:  # noqa: BLE001
            self.state.extra["last_chunk_size"] = len(self.state.action_queue)

    def dispatch_action(self, *, force: bool = False) -> None:
        if self.state.mode != "action":
            self._last_dispatch_seconds = None
            return
        if self.state.tick is None or not self._ctrl_gate.due(self.state.tick, force=force):
            return
        queue = self.state.action_queue
        if not queue:
            self._last_dispatch_seconds = None
            return
        now = time.monotonic()
        if self._last_dispatch_seconds is None or self.ctrl_hz <= 0:
            n_to_pop = 1
        else:
            n_to_pop = max(1, min(len(queue), int(round((now - self._last_dispatch_seconds) * self.ctrl_hz))))
        self._last_dispatch_seconds = now
        latest = None
        for _ in range(n_to_pop):
            if not queue:
                break
            latest = queue.popleft()
            self.state.actions_dispatched += 1
        if latest is not None and self.action_executor is not None:
            self.action_executor(latest)

    def _handle_action_deadline(self) -> None:
        deadline = self.state.action_deadline
        if self.state.mode == "action" and deadline is not None and time.monotonic() >= deadline:
            self.state.mode = "paused"
            self.state.action_deadline = None
            self.state.action_queue.clear()
            self.state.log("timed action elapsed — paused")

    def _flush_logs(self) -> None:
        for line in self.state.log_lines:
            print(f"[runtime] {line}", flush=True)

    def _on_shutdown(self) -> None:
        self.state.action_queue.clear()
        print("[runtime] stopped", flush=True)
