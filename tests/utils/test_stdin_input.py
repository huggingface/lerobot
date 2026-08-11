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

"""Tests for the non-blocking stdin line reader."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from threading import Event

from lerobot.utils.stdin_input import StdinCommandListener


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@contextlib.contextmanager
def _pipe_stream():
    """A held-open pipe so the listener never sees EOF until we close it."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r")
    writer = os.fdopen(write_fd, "w")
    try:
        yield reader, writer
    finally:
        with contextlib.suppress(OSError, ValueError):
            writer.close()
        with contextlib.suppress(OSError, ValueError):
            reader.close()


def test_stdin_listener_reads_lines_and_eof():
    lines: list[str] = []
    eof = Event()

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=reader)
        listener.start()
        writer.write("/start\n")
        writer.write("   \n")  # blank lines are skipped
        writer.write("/help\n")
        writer.flush()
        assert _wait_for(lambda: len(lines) == 2)
        assert lines == ["/start", "/help"]

        writer.close()
        assert _wait_for(eof.is_set)
        listener.stop()


def test_stdin_listener_delivers_batched_lines():
    """Several lines arriving in one chunk (paste, piped script) are all delivered."""
    lines: list[str] = []

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, stream=reader)
        listener.start()
        writer.write("/start\n/subtask grab the cube\n/stop\n")
        writer.flush()
        assert _wait_for(lambda: len(lines) == 3)
        assert lines == ["/start", "/subtask grab the cube", "/stop"]
        listener.stop()


def test_stdin_listener_handler_errors_do_not_kill_reader():
    lines: list[str] = []

    def flaky(line: str) -> None:
        if line == "/boom":
            raise RuntimeError("boom")
        lines.append(line)

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(flaky, stream=reader)
        listener.start()
        writer.write("/boom\n/start\n")
        writer.flush()
        assert _wait_for(lambda: lines == ["/start"])
        listener.stop()


def test_stdin_listener_blocking_fallback():
    """Streams without a file descriptor (e.g. StringIO) use the blocking readline path."""
    lines: list[str] = []
    eof = Event()
    listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=io.StringIO("/start\n\n/help\n"))
    assert not listener._use_select
    listener.start()
    assert _wait_for(eof.is_set)
    assert lines == ["/start", "/help"]
    listener.stop()


def test_stdin_listener_blocking_bytes_stream_reaches_eof():
    """A bytes stream on the blocking path must fire on_eof and end the thread.

    ``readline()`` returns ``b""`` at EOF there, which a ``== ""`` check misses
    — the regression this guards against was a 100% CPU busy loop that never
    signalled EOF.
    """
    lines: list[str] = []
    eof = Event()
    listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=io.BytesIO(b"/start\n"))
    assert not listener._use_select
    listener.start()
    assert _wait_for(eof.is_set)
    assert lines == ["/start"]
    assert _wait_for(lambda: not listener._thread.is_alive())
    listener.stop()


def test_stdin_listener_final_line_without_newline_is_flushed_at_eof():
    """A trailing command with no newline (printf '/stop') is delivered before on_eof."""
    lines: list[str] = []
    eof = Event()

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=reader)
        listener.start()
        writer.write("/start\n/stop")
        writer.flush()
        assert _wait_for(lambda: lines == ["/start"])

        writer.close()
        assert _wait_for(eof.is_set)
        assert lines == ["/start", "/stop"]
        listener.stop()


def test_stdin_listener_accumulates_line_split_across_reads():
    """A line arriving in two chunks is delivered once, whole."""
    lines: list[str] = []

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, stream=reader)
        listener.start()
        writer.write("/subtask grab ")
        writer.flush()
        time.sleep(0.05)  # let the listener consume the partial chunk
        assert lines == []

        writer.write("the cube\n")
        writer.flush()
        assert _wait_for(lambda: lines == ["/subtask grab the cube"])
        listener.stop()


def test_stdin_listener_none_stdin_treated_as_eof(monkeypatch):
    """A missing sys.stdin (daemonized process) must fire on_eof, not hang silently."""
    monkeypatch.setattr(sys, "stdin", None)
    eof = Event()
    listener = StdinCommandListener(lambda line: None, on_eof=eof.set)
    listener.start()
    assert eof.is_set()
    listener.stop()


def test_stdin_listener_broken_stream_treated_as_eof():
    """A stream whose readline blows up mid-run must fire on_eof (dead command channel)."""

    class _BrokenStream:
        def readline(self):
            raise AttributeError("broken")

    eof = Event()
    listener = StdinCommandListener(lambda line: None, on_eof=eof.set, stream=_BrokenStream())
    listener.start()
    assert _wait_for(eof.is_set)
    listener.stop()
