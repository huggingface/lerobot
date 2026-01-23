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
import threading
from unittest.mock import patch

import pytest

from lerobot.rl.process import ProcessSignalHandler


# Fixture to reset shutdown_event_counter and original signal handlers before and after each test
@pytest.fixture(autouse=True)
def reset_globals_and_handlers():
    # Store original signal handlers
    original_handlers = {
        sig: signal.getsignal(sig)
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]
        if hasattr(signal, sig.name)
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
        # SIGHUP and SIGQUIT are not reliably available on all platforms (e.g. Windows)
        pytest.param(
            signal.SIGHUP,
            marks=pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP not available"),
        ),
        pytest.param(
            signal.SIGQUIT,
            marks=pytest.mark.skipif(not hasattr(signal, "SIGQUIT"), reason="SIGQUIT not available"),
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
