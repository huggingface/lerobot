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

# SIGHUP and SIGQUIT are not defined on every platform (e.g. Windows). Look them up with
# getattr: a bare `signal.SIGHUP` raises AttributeError while this module is being imported,
# which aborts collection of the whole test session before any skip mark can apply.
SIGHUP = getattr(signal, "SIGHUP", None)
SIGQUIT = getattr(signal, "SIGQUIT", None)

# On Windows, os.kill() only delivers CTRL_C_EVENT / CTRL_BREAK_EVENT; for any other signal it
# calls TerminateProcess, so a process cannot send itself a catchable signal. Tests that rely on
# it would kill the pytest process instead of exercising the handler.
requires_self_signal = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill() cannot deliver a catchable signal to the current process on Windows",
)


# Fixture to reset shutdown_event_counter and original signal handlers before and after each test
@pytest.fixture(autouse=True)
def reset_globals_and_handlers():
    # Store original signal handlers
    original_handlers = {
        sig: signal.getsignal(sig)
        for sig in [signal.SIGINT, signal.SIGTERM, SIGHUP, SIGQUIT]
        if sig is not None
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


@requires_self_signal
@pytest.mark.parametrize("use_threads", [True, False])
@pytest.mark.parametrize(
    "sig",
    [
        signal.SIGINT,
        signal.SIGTERM,
        # SIGHUP and SIGQUIT are not reliably available on all platforms (e.g. Windows)
        pytest.param(
            SIGHUP,
            marks=pytest.mark.skipif(SIGHUP is None, reason="SIGHUP not available"),
            id="SIGHUP",
        ),
        pytest.param(
            SIGQUIT,
            marks=pytest.mark.skipif(SIGQUIT is None, reason="SIGQUIT not available"),
            id="SIGQUIT",
        ),
    ],
)
def test_signal_handler_sets_event(use_threads, sig):
    """Test that the signal handler sets the event on receiving a signal."""
    handler = ProcessSignalHandler(use_threads=use_threads)
    shutdown_event = handler.shutdown_event

    assert handler.counter == 0

    os.kill(os.getpid(), sig)

    # In some environments, the signal might take a moment to be handled.
    shutdown_event.wait(timeout=1.0)

    assert shutdown_event.is_set(), f"Event should be set after receiving signal {sig}"

    # Ensure the internal counter was incremented
    assert handler.counter == 1


@requires_self_signal
@pytest.mark.parametrize("use_threads", [True, False])
@patch("sys.exit")
def test_force_shutdown_on_second_signal(mock_sys_exit, use_threads):
    """Test that a second signal triggers a force shutdown."""
    handler = ProcessSignalHandler(use_threads=use_threads)

    os.kill(os.getpid(), signal.SIGINT)
    # Give a moment for the first signal to be processed
    import time

    time.sleep(0.1)
    os.kill(os.getpid(), signal.SIGINT)

    time.sleep(0.1)

    assert handler.counter == 2
    mock_sys_exit.assert_called_once_with(1)
