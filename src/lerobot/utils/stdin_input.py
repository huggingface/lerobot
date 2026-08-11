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

"""Non-blocking, line-oriented stdin reading.

This complements :mod:`lerobot.utils.keyboard_input`, which serves discrete
hotkeys: :class:`TerminalKeyListener` reads single raw bytes in cbreak mode,
whereas :class:`StdinCommandListener` here assembles whole typed lines and
leaves the terminal in canonical (line-buffered, echoing) mode — the operator
is typing chat-style commands, not pressing hotkeys.  The two cannot share
stdin at the same time.  More generally the listener must be the stream's
*sole* consumer: in select mode it reads the file descriptor directly with
``os.read``, so nothing else in the process may read the same stream while it
runs, and bytes already slurped into a buffered wrapper (e.g. by an earlier
``input()`` call) are invisible to it.

Environment support: reading works over SSH (the session's pty is a regular
TTY file descriptor), in headless setups (no display server is involved,
unlike the ``pynput`` keyboard backend), and from piped stdin.  End-of-file
means "no more commands": an interactive Ctrl-D, an exhausted piped script,
or ``stdin`` redirected from ``/dev/null`` all trigger ``on_eof``, as does a
missing ``sys.stdin`` (e.g. a daemonized process).
"""

from __future__ import annotations

import logging
import os
import select
import sys
from collections.abc import Callable
from threading import Thread
from typing import IO

logger = logging.getLogger(__name__)


class StdinCommandListener:
    """Daemon thread that reads input lines and forwards them to a callback.

    On POSIX the reader polls the stream with ``select`` so ``stop()`` can
    end the thread promptly; elsewhere (or for file-like objects without a
    file descriptor) it falls back to a blocking ``readline`` daemon thread
    that dies with the process.  Blank lines are skipped; end-of-file and
    unexpected read errors trigger ``on_eof`` — a dead command channel must
    never leave the consumer waiting for input that can no longer arrive.
    """

    def __init__(
        self,
        on_line: Callable[[str], None],
        on_eof: Callable[[], None] | None = None,
        stream: IO[str] | IO[bytes] | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        self._on_line = on_line
        self._on_eof = on_eof
        # sys.stdin can itself be None (pythonw, daemonized processes);
        # start() treats that as an immediately-closed stream.
        self._stream = stream if stream is not None else sys.stdin
        self._poll_interval_s = poll_interval_s
        self._running = False
        self._thread: Thread | None = None
        self._use_select = False
        if os.name == "posix":
            try:
                self._stream.fileno()
                self._use_select = True
            except (OSError, ValueError, AttributeError):
                pass

    def start(self) -> None:
        """Start the reader thread (idempotent).

        Callbacks fire on the reader thread — except when the stream is
        missing (``sys.stdin`` is ``None``), in which case ``on_eof`` fires
        synchronously on the caller's thread before ``start()`` returns.
        """
        if self._thread is not None:
            return
        if self._stream is None:
            logger.warning("No stdin available for command input — treating as EOF")
            self._emit_eof()
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="StdinCommandListener")
        self._thread.start()
        if not self._use_select:
            logger.info("stdin listener running in blocking mode (select unavailable for this stream)")

    def stop(self) -> None:
        """Stop the reader thread.

        Blocking-mode threads may be stuck inside ``readline`` and cannot be
        joined; they are daemons and die with the process. Late lines are
        ignored via the ``_running`` flag either way.
        """
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and self._use_select:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        if self._use_select:
            self._run_select()
        else:
            self._run_blocking()

    def _run_select(self) -> None:
        """Poll the file descriptor and split lines from raw bytes.

        Reading raw bytes (instead of ``stream.readline()``) matters: a
        buffered file object can slurp several lines off the descriptor at
        once, after which ``select`` reports the drained fd as not-ready and
        the buffered lines would never be delivered — breaking pasted or
        piped command sequences.
        """
        try:
            fd = self._stream.fileno()
        except (OSError, ValueError):  # closed between construction and thread start
            self._emit_read_error()
            return
        buffer = b""
        while self._running:
            try:
                ready, _, _ = select.select([fd], [], [], self._poll_interval_s)
            except (OSError, ValueError):  # stream closed underneath us
                self._emit_read_error()
                return
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                self._emit_read_error()
                return
            if not self._running:
                return
            if chunk == b"":  # EOF: Ctrl-D or the piped input ended
                # A final command without trailing newline still counts.
                self._emit_line(buffer.decode(errors="replace"))
                self._emit_eof()
                return
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                self._emit_line(raw.decode(errors="replace"))

    def _run_blocking(self) -> None:
        while self._running:
            try:
                line = self._stream.readline()
            except (OSError, ValueError, AttributeError):
                self._emit_read_error()
                return
            if not self._running:
                return
            if not line:  # EOF: "" on text streams, b"" on bytes streams
                self._emit_eof()
                return
            self._emit_line(line if isinstance(line, str) else line.decode(errors="replace"))

    def _emit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            self._on_line(line)
        except Exception:  # never let a handler error kill the reader thread
            logger.exception("Error while handling input line %r", line)

    def _emit_eof(self) -> None:
        logger.info("Input stream closed (EOF)")
        if self._on_eof is not None:
            try:
                self._on_eof()
            except Exception:
                logger.exception("Error while handling input EOF")

    def _emit_read_error(self) -> None:
        """Treat an unexpected read failure like EOF so consumers shut down.

        A dead command channel must not leave the consumer running with no
        way to reach it.  Deliberate ``stop()`` calls clear ``_running``
        first and do not reach this path.
        """
        if self._running:
            logger.warning("Input stream failed — treating as EOF")
            self._emit_eof()
