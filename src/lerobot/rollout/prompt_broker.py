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

"""Thread-safe prompt broker and listener for hot-switching the task string at runtime.

The :class:`PromptBroker` holds the current task string and is shared across
all strategies and inference engines.  A :class:`PromptListenerBase` subclass
updates the broker from a background daemon thread whenever a new prompt
arrives from an external source.

Supported sources
-----------------
``stdin``
    Each non-empty line typed (or piped) in the terminal becomes the new
    active task::

        # Interactive: type a new task and press Enter
        lerobot-rollout --online_task_switching=true --task="pick up cube" ...

        # Pipe from a voice-STT tool:
        whisper_stt_tool | lerobot-rollout --online_task_switching=true ...
"""

from __future__ import annotations

import abc
import logging
import select
import sys
from collections.abc import Callable
from threading import Event, Lock, Thread

logger = logging.getLogger(__name__)


class PromptBroker:
    """Thread-safe holder for the current task string.

    All strategies and inference engines share a single instance so that
    a call to :meth:`set_task` is immediately visible everywhere on the
    next :meth:`get_task` call.
    """

    def __init__(self, initial_task: str = "") -> None:
        self._lock = Lock()
        self._task = initial_task
        self._on_change_callbacks: list[Callable[[], None]] = []
        logger.info("PromptBroker initialised (task='%s')", initial_task)

    def register_on_change(self, callback: Callable[[], None]) -> None:
        """Register a zero-argument callable invoked whenever the task changes.

        Callbacks are called inside the listener thread immediately after the
        task is updated, still holding no lock.  Keep them fast and non-blocking.
        """
        self._on_change_callbacks.append(callback)

    def get_task(self) -> str:
        """Return the current task string (thread-safe)."""
        with self._lock:
            return self._task

    def set_task(self, task: str) -> None:
        """Update the task string and log the transition (thread-safe)."""
        with self._lock:
            old = self._task
            self._task = task
        if old != task:
            logger.info("Task switched: '%s' → '%s'", old, task)
            for cb in self._on_change_callbacks:
                cb()


class PromptListenerBase(abc.ABC):
    """Abstract base for prompt source listeners.

    Subclasses implement :meth:`_listen`, which runs in a daemon thread
    and calls ``broker.set_task(new_task)`` whenever a new prompt
    arrives from its source.
    """

    def start(self, broker: PromptBroker, shutdown_event: Event) -> None:
        """Launch the listener in a background daemon thread."""
        t = Thread(
            target=self._listen,
            args=(broker, shutdown_event),
            daemon=True,
            name=type(self).__name__,
        )
        t.start()
        logger.info("%s listener started", type(self).__name__)

    @abc.abstractmethod
    def _listen(self, broker: PromptBroker, shutdown_event: Event) -> None:
        """Block until shutdown, calling broker.set_task() on each new prompt."""


class StdinPromptListener(PromptListenerBase):
    """Reads new task prompts from ``stdin``.

    Each non-empty line becomes the new active task.  Uses
    ``select.select`` with a short timeout so the thread exits cleanly
    when ``shutdown_event`` is set without blocking indefinitely on a
    ``readline()`` call.

    Unix-pipe example::

        whisper_stt_tool | lerobot-rollout --online_task_switching=true ...
    """

    _SELECT_TIMEOUT_S: float = 0.5

    def _listen(self, broker: PromptBroker, shutdown_event: Event) -> None:
        logger.info(
            "StdinPromptListener active — type a new task and press Enter to switch (current: '%s')",
            broker.get_task(),
        )
        while not shutdown_event.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], self._SELECT_TIMEOUT_S)
            except (ValueError, OSError):
                # stdin was closed (e.g. the piped upstream process exited)
                break
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                # EOF — pipe closed or Ctrl-D
                logger.info("StdinPromptListener: stdin closed (EOF), exiting")
                break
            task = line.strip()
            if task:
                broker.set_task(task)
        logger.info("StdinPromptListener stopped")
