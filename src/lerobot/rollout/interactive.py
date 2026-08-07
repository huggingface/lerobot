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

"""Interactive rollout session: chat-style stdin commands for ``lerobot-rollout``.

Enabled with ``--interactive=true``, this module lets the operator control a
rollout from the terminal while hardware and policy stay connected and warm:

    /start   start (or restart) the policy control loop
    /reset   stop movement and return the robot to its initial position
    /stop    end the session and run the normal shutdown routines
    /help    show the available commands

Threading model (mirrors the DAgger events pattern): a daemon
:class:`StdinCommandListener` thread reads lines and only ever sets
thread-safe flags — it never touches hardware or the inference engine.  The
:class:`InteractiveSession` driver runs on the main thread and executes
``strategy.run(ctx)`` in *segments*: each ``/start`` begins a segment, and
``/reset`` / ``/stop`` end it by setting the session's :class:`LinkedEvent`,
which every strategy control loop already polls as
``ctx.runtime.shutdown_event``.  Real shutdown signals (SIGINT/SIGTERM)
propagate through the linked event's parent, so Ctrl-C behaves exactly as in
non-interactive runs.

The command table is intentionally a name → handler mapping so future
commands (``/subtask``, ``/ask`` — see the language-runtime work in
PR #4183/#4234) can be registered without restructuring the parser or the
session loop.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread
from typing import IO, TYPE_CHECKING

from lerobot.utils.utils import log_say

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)

_BANNER_RULE = "─" * 60


class LinkedEvent(Event):
    """A ``threading.Event`` whose ``is_set`` also reflects a parent event.

    ``set``/``clear`` act only on the local flag, so the interactive session
    can raise and clear its own segment-stop requests without masking (or
    accidentally re-arming) the process-wide shutdown event carried by
    ``parent``.  Every rollout strategy control loop polls
    ``ctx.runtime.shutdown_event.is_set()``, so installing a ``LinkedEvent``
    there makes the loops react both to session commands and to real
    shutdown signals.
    """

    _WAIT_SLICE_S = 0.05

    def __init__(self, parent: Event) -> None:
        super().__init__()
        self.parent = parent

    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for either the local or the parent flag.

        The base ``Event.wait`` only watches the local flag, so poll in short
        slices to also observe the parent.  Strategy loops only call
        ``is_set()``; this coarse wait exists for API completeness.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None and remaining <= 0:
                return False
            wait_slice = self._WAIT_SLICE_S if remaining is None else min(self._WAIT_SLICE_S, remaining)
            super().wait(wait_slice)
        return True


@dataclass(frozen=True)
class InteractiveCommand:
    """A parsed ``/name args`` line from the interactive prompt."""

    name: str
    args: str = ""


def parse_command(line: str) -> InteractiveCommand | None:
    """Parse an input line into an :class:`InteractiveCommand`.

    Commands are ``/name`` optionally followed by free-text arguments
    (unused by the built-in commands, but the grammar already supports
    future ones like ``/subtask grab the red cube``).  Returns ``None`` for
    lines that are not commands (no leading ``/`` or a bare ``/``).
    """
    line = line.strip()
    if not line.startswith("/"):
        return None
    head, *rest = line.split(maxsplit=1)
    name = head[1:].lower()
    if not name:
        return None
    return InteractiveCommand(name=name, args=rest[0].strip() if rest else "")


class StdinCommandListener:
    """Daemon thread that reads input lines and forwards them to a callback.

    On POSIX the reader polls the stream with ``select`` so ``stop()`` can
    end the thread promptly; elsewhere (or for file-like objects without a
    file descriptor) it falls back to a blocking ``readline`` daemon thread
    that dies with the process.  Blank lines are skipped; end-of-file and
    unexpected read errors trigger ``on_eof`` (an interactive Ctrl-D or an
    exhausted piped script both mean "no more commands" — the session must
    not keep the robot running with no way to command it).

    Unlike :class:`lerobot.utils.keyboard_input.TerminalKeyListener`, this
    reader leaves the terminal in canonical (line-buffered, echoing) mode —
    the operator is typing chat-style commands, not pressing hotkeys.
    """

    def __init__(
        self,
        on_line: Callable[[str], None],
        on_eof: Callable[[], None] | None = None,
        stream: IO[str] | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        self._on_line = on_line
        self._on_eof = on_eof
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
        """Start the reader thread (idempotent)."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="InteractiveStdin")
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
        fd = self._stream.fileno()
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
                self._emit_line(buffer)  # a final command without trailing newline still counts
                self._emit_eof()
                return
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                self._emit_line(raw)

    def _run_blocking(self) -> None:
        while self._running:
            try:
                line = self._stream.readline()
            except (OSError, ValueError):
                self._emit_read_error()
                return
            if not self._running:
                return
            if line == "":  # EOF
                self._emit_eof()
                return
            self._emit_line(line.encode() if isinstance(line, str) else line)

    def _emit_line(self, raw: bytes) -> None:
        line = raw.decode(errors="replace").strip()
        if not line:
            return
        try:
            self._on_line(line)
        except Exception:  # never let a handler error kill the reader thread
            logger.exception("Error while handling interactive input %r", line)

    def _emit_eof(self) -> None:
        logger.info("Interactive input stream closed (EOF)")
        if self._on_eof is not None:
            try:
                self._on_eof()
            except Exception:
                logger.exception("Error while handling interactive input EOF")

    def _emit_read_error(self) -> None:
        """Treat an unexpected read failure like EOF so the session shuts down.

        A dead command channel must not leave the robot running with no way
        to stop it.  Deliberate ``stop()`` calls clear ``_running`` first and
        do not reach this path.
        """
        if self._running:
            logger.warning("Interactive input stream failed — treating as EOF")
            self._emit_eof()


class InteractiveSession:
    """Drive a rollout strategy from chat-style stdin commands.

    The session owns the outer lifecycle: after ``strategy.setup(ctx)`` the
    robot stays idle until ``/start``.  Each run *segment* executes
    ``strategy.run(ctx)`` on the calling (main) thread until the operator
    interrupts it or the strategy returns on its own (e.g. ``--duration``
    elapsed).  ``/reset`` pauses the inference engine and returns the robot
    to its initial position while hardware and policy stay warm; ``/stop``
    ends the session so the caller can run ``strategy.teardown(ctx)`` — the
    same shutdown routine as non-interactive rollouts.

    Requires ``ctx.runtime.shutdown_event`` to be a :class:`LinkedEvent`
    (installed by ``lerobot-rollout`` when ``--interactive=true``): the
    session sets the local flag to end a segment, and process signals still
    propagate through the parent.

    Commands are last-write-wins: ``/reset`` and ``/stop`` cancel a pending
    ``/start`` so the robot never starts moving after the operator's final
    command asked it not to.  End-of-file on the command stream stops the
    session (a closed stdin means there is no way left to command the
    robot), so piped scripts must keep stdin open for the intended session
    duration, e.g. ``(printf '/start\\n'; sleep 60; printf '/stop\\n') |
    lerobot-rollout ... --interactive=true``.
    """

    _POLL_INTERVAL_S = 0.2

    def __init__(
        self,
        strategy: RolloutStrategy,
        ctx: RolloutContext,
        input_stream: IO[str] | None = None,
    ) -> None:
        stop_event = ctx.runtime.shutdown_event
        if not isinstance(stop_event, LinkedEvent):
            raise TypeError(
                "InteractiveSession requires ctx.runtime.shutdown_event to be a LinkedEvent so "
                "/reset can end a run segment without triggering process shutdown. Build the "
                "rollout context with build_rollout_context(cfg, LinkedEvent(shutdown_event))."
            )
        self._strategy = strategy
        self._ctx = ctx
        self._segment_stop = stop_event
        self._global_shutdown = stop_event.parent
        self._listener = StdinCommandListener(self._handle_line, on_eof=self._handle_eof, stream=input_stream)

        # Written by the listener thread, consumed by the main loop.
        self._start_requested = Event()
        self._reset_requested = Event()
        self._stop_requested = Event()
        self._wake = Event()
        self._running = Event()

        # name -> (handler, help line); /help and the banner render from this
        # table, so future commands (/subtask, /ask) stay documented for free.
        self._commands: dict[str, tuple[Callable[[InteractiveCommand], None], str]] = {
            "start": (self._cmd_start, "start (or restart) the policy control loop"),
            "reset": (self._cmd_reset, "stop movement and return the robot to its initial position"),
            "stop": (self._cmd_stop, "end the session and shut down"),
            "help": (self._cmd_help, "show this help"),
        }

    # ------------------------------------------------------------------
    # Main-thread session loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the session until ``/stop``, EOF, engine failure, or a shutdown signal."""
        play_sounds = self._ctx.runtime.cfg.play_sounds
        self._print(self._render_banner())
        self._listener.start()
        try:
            while not self._global_shutdown.is_set():
                if self._ctx.policy.inference.failed:
                    self._print("Inference engine failed — shutting down. See the log for the error.")
                    break
                if self._stop_requested.is_set():
                    break
                if self._reset_requested.is_set():
                    self._reset_requested.clear()
                    self._reset_robot()
                    continue
                if self._start_requested.is_set():
                    self._start_requested.clear()
                    self._run_segment()
                    continue
                self._wake.wait(timeout=self._POLL_INTERVAL_S)
                self._wake.clear()
        finally:
            self._listener.stop()
            log_say("Interactive session ended", play_sounds)

    def _run_segment(self) -> None:
        """Execute one ``strategy.run`` segment until interrupted or finished."""
        engine = self._ctx.policy.inference
        # Clear the local flag *before* checking the request flags: command
        # handlers set their flag first and the segment-stop event second, so
        # a /reset or /stop racing with this /start is either seen here or
        # ends the freshly started loop on its first tick.
        self._segment_stop.clear()
        if self._stop_requested.is_set() or self._reset_requested.is_set() or self._global_shutdown.is_set():
            return
        self._strategy.reset_control_state()
        log_say("Starting rollout", self._ctx.runtime.cfg.play_sounds)
        self._print("Rollout running — /reset to pause and return to initial position, /stop to shut down.")
        self._running.set()
        try:
            self._strategy.run(self._ctx)
        finally:
            self._running.clear()
            engine.pause()
        if engine.failed:
            return  # the session loop reports the failure and shuts down
        if not (
            self._stop_requested.is_set() or self._reset_requested.is_set() or self._global_shutdown.is_set()
        ):
            self._print(
                "Rollout run ended on its own (duration reached). Robot is holding position — "
                "/start to run again, /reset to return to initial position, /stop to shut down."
            )

    def _reset_robot(self) -> None:
        """Pause inference and return the robot to its initial position."""
        self._ctx.policy.inference.pause()
        log_say("Resetting robot to initial position", self._ctx.runtime.cfg.play_sounds)
        if self._ctx.hardware.initial_position:
            self._strategy.return_to_initial_position(self._ctx.hardware)
            self._print("Robot reset — holding at initial position. /start to run.")
        else:
            logger.warning("No initial position captured — skipping the return move")
            self._print("Robot paused — no initial position captured, holding current pose. /start to run.")

    # ------------------------------------------------------------------
    # Command handlers (called from the listener thread; only set flags)
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        cmd = parse_command(line)
        if cmd is None:
            self._print("Input not recognized — commands start with '/'. Type /help for the list.")
            return
        entry = self._commands.get(cmd.name)
        if entry is None:
            self._print(f"Unknown command '/{cmd.name}'. Type /help for the list.")
            return
        handler, _ = entry
        handler(cmd)

    def _handle_eof(self) -> None:
        self._print("Input stream closed — stopping the session.")
        self._request_stop()

    def _cmd_start(self, cmd: InteractiveCommand) -> None:
        if self._running.is_set():
            self._print("Already running — /reset to pause first, or /stop to shut down.")
            return
        self._start_requested.set()
        self._wake.set()

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        # Last command wins: a /start still waiting to be serviced is cancelled
        # so the robot never starts moving after the operator asked it not to.
        # Flag first, segment-stop second (see the ordering note in _run_segment).
        self._start_requested.clear()
        self._reset_requested.set()
        self._segment_stop.set()
        self._wake.set()

    def _cmd_stop(self, cmd: InteractiveCommand) -> None:
        self._request_stop()

    def _request_stop(self) -> None:
        self._start_requested.clear()  # last command wins, see _cmd_reset
        self._stop_requested.set()
        self._segment_stop.set()
        self._wake.set()

    def _cmd_help(self, cmd: InteractiveCommand) -> None:
        self._print(self._render_help())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_help(self) -> str:
        width = max(len(name) for name in self._commands)
        lines = [f"  /{name:<{width}}   {help_line}" for name, (_, help_line) in self._commands.items()]
        return "Available commands:\n" + "\n".join(lines)

    def _render_banner(self) -> str:
        return (
            f"{_BANNER_RULE}\n"
            "Interactive rollout session — the robot will NOT move until you type /start.\n"
            f"{self._render_help()}\n"
            f"{_BANNER_RULE}"
        )

    @staticmethod
    def _print(message: str) -> None:
        """User-facing chat output; logging stays on stderr, replies on stdout."""
        print(message, flush=True)
