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
from copy import copy
from threading import Event, Lock
from typing import TYPE_CHECKING

import torch

from lerobot.configs import TextKind

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

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
    across threads.

    Text queries
    ------------
    Backends publish their latest policy-ready observation through
    :meth:`_publish_text_observation`.  The interactive ``/ask`` worker takes
    a shallow snapshot and calls :meth:`generate_text`; an engine-owned gate
    prevents language decoding from racing an action-model call.  Action
    inference uses a non-blocking acquire so the hardware loop keeps ticking
    while a text response is being decoded.

    Optional hooks
    --------------
    ``notify_observation`` / ``pause`` / ``resume`` have a no-op default
    so rollout strategies can invoke them unconditionally.

    Subclasses must call ``super().__init__(task=..., policy=...)``; the task
    holder and optional text-generation plumbing are set up there.
    """

    def __init__(self, task: str = "", policy: PreTrainedPolicy | None = None) -> None:
        self._task = task
        self._task_changed = False
        self._task_lock = Lock()
        self._policy = policy

        self._text_observation: dict | None = None
        self._text_observation_lock = Lock()
        self._text_observation_publication_enabled = True

        # A text query gets priority after the current action inference ends.
        # Action backends never block on this lock: they return "no action yet"
        # and let the control loop keep servicing hardware at its normal rate.
        self._policy_call_lock = Lock()
        self._text_query_pending = Event()
        self._text_query_serial_lock = Lock()

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

    def _discard_task_change(self) -> None:
        """Drop a pending task-change edge, e.g. from ``reset`` (state is already cleared)."""
        with self._task_lock:
            self._task_changed = False

    # ------------------------------------------------------------------
    # Optional text generation (interactive /ask)
    # ------------------------------------------------------------------

    def supports_text_generation(self) -> bool:
        """Whether the attached policy implements the optional text hook."""
        return self._policy is not None and self._policy.supports_text_generation()

    def _publish_text_observation(self, observation: dict) -> None:
        """Cache a policy-ready observation for a future background query."""
        with self._text_observation_lock:
            if self._text_observation_publication_enabled:
                self._text_observation = copy(observation)

    def invalidate_text_observation(self) -> None:
        """Discard the cached view and reject publications until the next reset.

        ``/reset`` calls this before its deferred main-thread homing begins.
        Keeping publication disabled matters because action inference may
        already be in flight and otherwise republish the pre-reset scene.
        """
        with self._text_observation_lock:
            self._text_observation = None
            self._text_observation_publication_enabled = False

    def _reset_text_observation(self) -> None:
        """Clear the cached view and allow the next control segment to publish."""
        with self._text_observation_lock:
            self._text_observation = None
            self._text_observation_publication_enabled = True

    def snapshot_text_observation(self) -> dict | None:
        """Return the latest policy-ready view without touching robot hardware."""
        with self._text_observation_lock:
            if self._text_observation is None:
                return None
            observation = copy(self._text_observation)
        # A /subtask may have arrived since this visual observation was cached.
        # Keep the image/state snapshot but pair it with the latest instruction.
        observation["task"] = self.task
        return observation

    def generate_text(
        self,
        observation: dict,
        *,
        kind: TextKind,
        user_text: str | None = None,
    ) -> str:
        """Run one policy text query outside the hardware loop.

        Only one query is admitted at a time.  Setting the pending edge before
        acquiring the shared policy gate prevents a busy action backend from
        repeatedly winning the lock and starving the question.
        """
        if self._policy is None or not self._policy.supports_text_generation():
            raise NotImplementedError("This policy does not support text generation.")
        with self._text_query_serial_lock:
            self._text_query_pending.set()
            try:
                with self._policy_call_lock, torch.inference_mode():
                    return self._policy.generate_text(observation, kind=kind, user_text=user_text)
            finally:
                self._text_query_pending.clear()

    def _try_begin_action_inference(self) -> bool:
        """Acquire policy ownership without blocking the hardware loop."""
        if self._text_query_pending.is_set():
            return False
        return self._policy_call_lock.acquire(blocking=False)

    def _end_action_inference(self) -> None:
        """Release policy ownership acquired by :meth:`_try_begin_action_inference`."""
        self._policy_call_lock.release()

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
