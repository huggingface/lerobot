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

"""Inference engine ABC.

Rollout strategies consume actions through this small interface so they
do not need to know whether inference happens inline on the control thread
or asynchronously in a background thread (RTC).
"""

from __future__ import annotations

import abc
import logging
from threading import Lock

import torch

logger = logging.getLogger(__name__)


class InferenceEngine(abc.ABC):
    """Abstract backend for producing actions during rollout.

    Subclasses decide whether inference happens inline on the control
    thread or asynchronously in a background thread.  The contract is
    minimal so additional backends can be plugged in without touching
    rollout strategies.

    Lifecycle
    ---------
    ``start`` — prepare the backend (e.g. launch a background thread).
    ``stop`` — shut the backend down cleanly.
    ``reset`` — clear episode-scoped state (policy hidden state, queues…).

    Action production
    -----------------
    ``get_action(obs_frame)`` — return the next action tensor, or
    ``None`` if none is available (e.g. async queue empty).  Sync
    backends always compute from ``obs_frame``; async backends ignore
    it (they receive observations via ``notify_observation``).

    Task
    ----
    ``task`` / ``set_task`` hold the language instruction the policy is
    conditioned on.  ``set_task`` is safe to call from any thread (the
    interactive session's ``/subtask`` command calls it from its stdin
    reader); subclasses pick the new value up on their own inference
    thread via :meth:`_take_task`, so no policy state is ever mutated
    across threads.  ``dispatched_task`` names the instruction that
    generated the most recently returned action — it trails ``task``
    while actions produced under the previous instruction are still
    being consumed.

    Optional hooks
    --------------
    ``notify_observation`` / ``pause`` / ``resume`` have a no-op default
    so rollout strategies can invoke them unconditionally.

    Subclasses must call ``super().__init__(task=...)``; the task holder
    is set up there.
    """

    def __init__(self, task: str = "") -> None:
        self._task = task
        self._task_changed = False
        self._dispatched_task = task
        self._task_lock = Lock()

    # ------------------------------------------------------------------
    # Task (language instruction)
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """The language instruction currently conditioning inference."""
        with self._task_lock:
            return self._task

    def set_task(self, task: str) -> bool:
        """Set the instruction used from the next inference onwards.

        Callable from any thread.  Returns ``True`` when the value
        actually changed, so callers can report no-op switches.
        """
        with self._task_lock:
            if task == self._task:
                return False
            previous, self._task = self._task, task
            self._task_changed = True
        logger.info("Task changed: '%s' -> '%s'", previous, task)
        return True

    def _take_task(self) -> tuple[str, bool]:
        """Read the task and whether it changed since the last read.

        Call from the thread that runs inference: the "changed" edge is
        consumed here so the backend can drop actions precomputed under
        the previous instruction before using the new one.
        """
        with self._task_lock:
            changed, self._task_changed = self._task_changed, False
            return self._task, changed

    @property
    def dispatched_task(self) -> str:
        """Instruction that generated the most recently returned action.

        Unlike :attr:`task` — the *requested* instruction, which
        ``set_task`` changes immediately — this follows the actions the
        engine actually hands out: every successful ``get_action``
        records the task that generated the returned action.  Recording
        strategies label frames with it so actions still queued (or
        being interpolated) from a previous instruction keep that
        instruction's label.

        Only meaningful on the control thread right after it consumed
        ``get_action``; between a reset and the next dispatched action
        it holds the requested task.
        """
        with self._task_lock:
            return self._dispatched_task

    def _set_dispatched_task(self, task: str) -> None:
        """Record the task of the action a ``get_action`` call is returning."""
        with self._task_lock:
            self._dispatched_task = task

    def _discard_task_change(self) -> None:
        """Drop a pending task-change edge, e.g. from ``reset`` (state is already cleared).

        Also re-primes ``dispatched_task``: the caller just cleared any
        queued actions, so the next dispatched action can only come from
        the current instruction.
        """
        with self._task_lock:
            self._task_changed = False
            self._dispatched_task = self._task

    @abc.abstractmethod
    def start(self) -> None:
        """Initialise the backend."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear the backend down."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped state."""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action tensor, or ``None`` if unavailable."""

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """Publish the latest processed observation.  Default: no-op."""

    def pause(self) -> None:  # noqa: B027
        """Pause background inference.  Default: no-op."""

    def resume(self) -> None:  # noqa: B027
        """Resume background inference.  Default: no-op."""

    @property
    def ready(self) -> bool:
        """True once the backend can produce actions (e.g. warmup done)."""
        return True

    @property
    def failed(self) -> bool:
        """True if an unrecoverable error occurred in the backend."""
        return False

    @property
    def failure_traceback(self) -> str | None:
        """Formatted traceback of the unrecoverable error, when ``failed`` is True."""
        return None
