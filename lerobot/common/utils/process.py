#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import logging
import os
import signal
import sys


class ProcessSignalHandler:
    """Utility class to attach graceful shutdown signal handlers.

    The class exposes a shutdown_event attribute that is set when a shutdown
    signal is received. A counter tracks how many shutdown signals have been
    caught. On the second signal the process exits with status 1.
    """

    _SUPPORTED_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")

    def __init__(self, use_threads: bool, display_pid: bool = False):
        # TODO: Check if we can use Event from threading since Event from
        # multiprocessing is the a clone of threading.Event.
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Event
        if use_threads:
            from threading import Event
        else:
            from multiprocessing import Event

        self.shutdown_event = Event()
        self._counter: int = 0
        self._display_pid = display_pid

        self._register_handlers()

    @property
    def counter(self) -> int:  # pragma: no cover – simple accessor
        """Number of shutdown signals that have been intercepted."""
        return self._counter

    def _register_handlers(self):
        """Attach the internal _signal_handler to a subset of POSIX signals."""

        def _signal_handler(signum, frame):
            pid_str = ""
            if self._display_pid:
                pid_str = f"[PID: {os.getpid()}]"
            logging.info(f"{pid_str} Shutdown signal {signum} received. Cleaning up…")
            self.shutdown_event.set()
            self._counter += 1

            # On a second Ctrl-C (or any supported signal) force the exit to
            # mimic the previous behaviour while giving the caller one chance to
            # shutdown gracefully.
            # TODO: Investigate if we need it later
            if self._counter > 1:
                logging.info("Force shutdown")
                sys.exit(1)

        for sig_name in self._SUPPORTED_SIGNALS:
            sig = getattr(signal, sig_name, None)
            if sig is None:
                # The signal is not available on this platform (Windows for
                # instance does not provide SIGHUP, SIGQUIT…). Skip it.
                continue
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):  # pragma: no cover – unlikely but safe
                # Signal not supported or we are in a non-main thread.
                continue
