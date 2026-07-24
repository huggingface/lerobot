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

"""Unit tests for PromptBroker and StdinPromptListener."""

import os
import time
from contextlib import contextmanager
from threading import Event
from unittest.mock import patch

from lerobot.rollout.prompt_broker import PromptBroker, StdinPromptListener


@contextmanager
def _pipe_stdin(content: str):
    """Yield a read-end file pre-loaded with *content*.

    The write end is closed before yielding so the read end sees EOF after
    consuming all bytes — exactly like a finished pipe from a STT tool.
    Both objects are real file descriptors so ``select.select`` works.
    """
    r_fd, w_fd = os.pipe()
    with os.fdopen(w_fd, "w") as w:
        w.write(content)
    r = os.fdopen(r_fd, "r")
    try:
        yield r
    finally:
        r.close()


# ---------------------------------------------------------------------------
# PromptBroker
# ---------------------------------------------------------------------------


class TestPromptBroker:
    def test_initial_task_returned(self):
        broker = PromptBroker(initial_task="pick up cube")
        assert broker.get_task() == "pick up cube"

    def test_default_initial_task_is_empty(self):
        broker = PromptBroker()
        assert broker.get_task() == ""

    def test_set_task_updates_value(self):
        broker = PromptBroker(initial_task="task A")
        broker.set_task("task B")
        assert broker.get_task() == "task B"

    def test_set_task_same_value_no_change(self):
        broker = PromptBroker(initial_task="task A")
        broker.set_task("task A")
        assert broker.get_task() == "task A"

    def test_thread_safety_concurrent_writers(self):
        """Multiple threads writing simultaneously should not corrupt state."""
        import threading

        broker = PromptBroker(initial_task="initial")
        errors = []

        def writer(task: str):
            for _ in range(200):
                try:
                    broker.set_task(task)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"task_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions and final task is one of the valid values
        assert not errors
        assert broker.get_task().startswith("task_")

    def test_thread_safety_concurrent_readers_and_writers(self):
        """Readers should never see a partial/corrupt value."""
        import threading

        broker = PromptBroker(initial_task="start")
        errors = []
        stop = threading.Event()

        def writer():
            tasks = ["alpha", "beta", "gamma", "delta"]
            i = 0
            while not stop.is_set():
                broker.set_task(tasks[i % len(tasks)])
                i += 1

        def reader():
            while not stop.is_set():
                task = broker.get_task()
                if not isinstance(task, str):
                    errors.append(f"Expected str, got {type(task)}")

        threads = [threading.Thread(target=writer) for _ in range(2)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for t in threads:
            t.start()
        time.sleep(0.1)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        assert not errors


# ---------------------------------------------------------------------------
# StdinPromptListener
# ---------------------------------------------------------------------------


class TestStdinPromptListener:
    def test_listener_updates_broker_from_stdin(self):
        """Lines written to a mock stdin are forwarded to the broker."""
        broker = PromptBroker(initial_task="initial")
        shutdown = Event()

        with (
            _pipe_stdin("fold the towel\n") as fake_stdin,
            patch("lerobot.rollout.prompt_broker.sys.stdin", fake_stdin),
        ):
            listener = StdinPromptListener()
            listener.start(broker, shutdown)
            # Give the daemon thread time to process the line then hit EOF
            time.sleep(0.3)

        assert broker.get_task() == "fold the towel"

    def test_listener_ignores_empty_lines(self):
        """Blank / whitespace-only lines must not overwrite the task."""
        broker = PromptBroker(initial_task="original")
        shutdown = Event()

        with (
            _pipe_stdin("\n   \n\n") as fake_stdin,
            patch("lerobot.rollout.prompt_broker.sys.stdin", fake_stdin),
        ):
            listener = StdinPromptListener()
            listener.start(broker, shutdown)
            time.sleep(0.3)

        assert broker.get_task() == "original"

    def test_listener_processes_multiple_lines(self):
        """Each non-empty line updates the broker; final value is the last line."""
        broker = PromptBroker(initial_task="task0")
        shutdown = Event()

        with (
            _pipe_stdin("task1\ntask2\ntask3\n") as fake_stdin,
            patch("lerobot.rollout.prompt_broker.sys.stdin", fake_stdin),
        ):
            listener = StdinPromptListener()
            listener.start(broker, shutdown)
            time.sleep(0.3)

        assert broker.get_task() == "task3"

    def test_listener_stops_on_shutdown_event(self):
        """Setting shutdown_event causes the listener thread to exit cleanly."""
        broker = PromptBroker(initial_task="initial")
        shutdown = Event()

        # Blocking stdin — the thread should exit when shutdown is set
        read_pipe_r, read_pipe_w = __import__("os").pipe()
        import os

        fake_stdin_fd = os.fdopen(read_pipe_r, "r")

        try:
            with patch("lerobot.rollout.prompt_broker.sys.stdin", fake_stdin_fd):
                listener = StdinPromptListener()
                listener.start(broker, shutdown)
                time.sleep(0.05)
                shutdown.set()
                # Give thread up to 2× the select timeout to notice the event
                time.sleep(StdinPromptListener._SELECT_TIMEOUT_S * 2 + 0.1)
        finally:
            os.close(read_pipe_w)
            fake_stdin_fd.close()

        # Broker was never updated
        assert broker.get_task() == "initial"

    def test_listener_strips_whitespace(self):
        """Leading/trailing whitespace is stripped from prompts."""
        broker = PromptBroker(initial_task="initial")
        shutdown = Event()

        with (
            _pipe_stdin("  grasp the block  \n") as fake_stdin,
            patch("lerobot.rollout.prompt_broker.sys.stdin", fake_stdin),
        ):
            listener = StdinPromptListener()
            listener.start(broker, shutdown)
            time.sleep(0.3)

        assert broker.get_task() == "grasp the block"
