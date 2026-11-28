#!/usr/bin/env python

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

import multiprocessing
import os
import signal
import sys
import threading
from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.utils.process import ProcessSignalHandler  # noqa: E402

pytestmark = pytest.mark.dataset


# Fixture to reset shutdown_event_counter and original signal handlers before and after each test
_SIGNAL_NAMES = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")


def _send_signal_to_self(sig) -> None:
    """Deliver ``sig`` to the current process, portably.

    ``os.kill(pid, sig)`` cannot be used on Windows: only ``CTRL_C_EVENT`` /
    ``CTRL_BREAK_EVENT`` are delivered, anything else goes through
    ``TerminateProcess`` and would kill the test run instead of invoking the handler.
    ``signal.raise_signal`` goes through the C runtime and does reach the handler there.
    """
    if sys.platform == "win32":
        signal.raise_signal(sig)
    else:
        os.kill(os.getpid(), sig)


@pytest.fixture(autouse=True)
def reset_globals_and_handlers():
    # Store original signal handlers. Resolving the names with `getattr` keeps this
    # runnable where a signal does not exist (SIGHUP/SIGQUIT on Windows): referencing
    # `signal.SIGHUP` itself raises AttributeError before the `hasattr` filter can run.
    original_handlers = {
        getattr(signal, name): signal.getsignal(getattr(signal, name))
        for name in _SIGNAL_NAMES
        if hasattr(signal, name)
    }

    yield

    # Restore original signal handlers
    for sig, handler in original_handlers.items():
        signal.signal(sig, handler)


def test_setup_process_handlers_event_with_threads():
    """Test that setup_process_handlers returns the correct event type."""
    handler = ProcessSignalHandler(use_threads=True)
    shutdown_event = handler.shutdown_event
    assert isinstance(shutdown_event, threading.Event), "Should be a threading.Event"
    assert not shutdown_event.is_set(), "Event should initially be unset"


def test_setup_process_handlers_event_with_processes():
    """Test that setup_process_handlers returns the correct event type."""
    handler = ProcessSignalHandler(use_threads=False)
    shutdown_event = handler.shutdown_event
    assert isinstance(shutdown_event, type(multiprocessing.Event())), "Should be a multiprocessing.Event"
    assert not shutdown_event.is_set(), "Event should initially be unset"


@pytest.mark.parametrize("use_threads", [True, False])
@pytest.mark.parametrize(
    "sig",
    [
        signal.SIGINT,
        signal.SIGTERM,
        # SIGHUP and SIGQUIT are not reliably available on all platforms (e.g. Windows).
        # `getattr` keeps this list importable there: `signal.SIGHUP` itself would raise
        # AttributeError at collection time, before the skipif mark can do its job.
        pytest.param(
            getattr(signal, "SIGHUP", None),
            marks=pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP not available"),
        ),
        pytest.param(
            getattr(signal, "SIGQUIT", None),
            marks=pytest.mark.skipif(not hasattr(signal, "SIGQUIT"), reason="SIGQUIT not available"),
        ),
    ],
)
def test_signal_handler_sets_event(use_threads, sig):
    """Test that the signal handler sets the event on receiving a signal."""
    handler = ProcessSignalHandler(use_threads=use_threads)
    shutdown_event = handler.shutdown_event

    assert handler.counter == 0

    _send_signal_to_self(sig)

    # In some environments, the signal might take a moment to be handled.
    shutdown_event.wait(timeout=1.0)

    assert shutdown_event.is_set(), f"Event should be set after receiving signal {sig}"

    # Ensure the internal counter was incremented
    assert handler.counter == 1


@pytest.mark.parametrize("use_threads", [True, False])
@patch("sys.exit")
def test_force_shutdown_on_second_signal(mock_sys_exit, use_threads):
    """Test that a second signal triggers a force shutdown."""
    handler = ProcessSignalHandler(use_threads=use_threads)

    _send_signal_to_self(signal.SIGINT)
    # Give a moment for the first signal to be processed
    import time

    time.sleep(0.1)
    _send_signal_to_self(signal.SIGINT)

    time.sleep(0.1)

    assert handler.counter == 2
    mock_sys_exit.assert_called_once_with(1)
