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

    /start           start (or restart) the policy control loop
    /subtask <text>  change the instruction the policy follows, mid-run
    /ask <question>  ask the policy about the latest view without stopping
    /reset           stop movement, return the robot to its initial position,
                     and restore the instruction passed on the command line
    /stop            end the session and run the normal shutdown routines
    /help            show the available commands

Threading model (mirrors the DAgger events pattern): a daemon
:class:`StdinCommandListener` thread reads lines and only ever publishes
thread-safe state — flags for the session loop, and the instruction string
via :meth:`InferenceEngine.set_task`; it never touches hardware, and never
mutates policy state (the engine applies a task change on its own inference
thread).  ``/ask`` snapshots the latest policy-ready observation and hands it
to one background text-query worker; the worker never reads robot hardware.
The :class:`InteractiveSession` driver runs on the main thread and executes
``strategy.run(ctx)`` in *segments*: each ``/start`` begins a segment, and
``/reset`` / ``/stop`` end it by setting the session's :class:`LinkedEvent`,
which every strategy control loop already polls as
``ctx.runtime.shutdown_event``.  Real shutdown signals (SIGINT/SIGTERM)
propagate through the linked event's parent, so Ctrl-C behaves exactly as in
non-interactive runs.

While the session runs, console log handlers are muted (including
non-propagating library loggers like ``transformers``) and Python warnings
are suppressed, so system output does not interleave with the chat prompt;
only the session's own output is shown.  File log handlers are unaffected,
and console logging resumes when the session ends (so teardown logs are
visible).  A fatal inference-engine error is still surfaced: the session
prints the engine's captured traceback.  Run without ``--interactive`` to
see the full live log output.

The command table is intentionally a name → (handler, argument hint, help)
mapping so further commands can be registered without restructuring the
parser, the help output, or the session loop.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import IO, TYPE_CHECKING

from lerobot.configs import TextKind
from lerobot.utils.utils import log_say

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)

_BANNER_RULE = "─" * 60


def _mute_console_log_handlers() -> list[tuple[logging.Handler, int]]:
    """Mute console log handlers for the interactive session.

    System logs (policy, robot, control loop) contend with the chat prompt
    for the terminal, so raise every console handler above ``CRITICAL``
    while the session runs.  All loggers are covered, not just the root:
    libraries like ``transformers`` and ``datasets`` attach their own
    stderr handlers with ``propagate=False``.  File handlers are left
    untouched — anyone who wants a persistent log can attach one — and the
    previous levels are returned so :func:`_restore_log_handlers` can undo
    the muting.
    """
    loggers = [logging.getLogger()]
    loggers += [lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    muted = []
    for lg in loggers:
        for handler in lg.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                muted.append((handler, handler.level))
                handler.setLevel(logging.CRITICAL + 1)
    return muted


def _restore_log_handlers(muted: list[tuple[logging.Handler, int]]) -> None:
    """Restore handler levels changed by :func:`_mute_console_log_handlers`."""
    for handler, level in muted:
        handler.setLevel(level)


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


def _format_task(task: str) -> str:
    """Render a task string for the operator, naming the empty case explicitly."""
    return repr(task) if task else "(none — set one with /subtask <text>)"


def _strip_quotes(text: str) -> str:
    """Drop one layer of matching surrounding quotes from a command argument."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


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


@dataclass(frozen=True)
class TextQueryRequest:
    """One immutable policy question paired with the view seen at submission."""

    question: str
    observation: dict


class TextQueryWorker:
    """Single background worker for non-blocking interactive policy questions.

    At most one request may be queued or running.  This bounds the number of
    retained image tensors (the observation can live on the GPU) and gives the
    operator an explicit "busy" response instead of accumulating questions
    against increasingly stale views.
    """

    _STOP = object()
    _JOIN_TIMEOUT_S = 5.0

    def __init__(
        self,
        answer: Callable[[TextQueryRequest], str],
        on_answer: Callable[[TextQueryRequest, str], None],
        on_error: Callable[[TextQueryRequest, Exception], None],
    ) -> None:
        self._answer = answer
        self._on_answer = on_answer
        self._on_error = on_error
        self._queue: Queue[TextQueryRequest | object] = Queue(maxsize=1)
        self._state_lock = Lock()
        self._busy = False
        self._stopping = Event()
        self._stop_enqueued = False
        self._thread: Thread | None = None

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._busy

    def start(self) -> None:
        """Start the worker (idempotent)."""
        with self._state_lock:
            if self._thread is not None or self._stopping.is_set():
                return
            self._thread = Thread(target=self._run, daemon=True, name="InteractiveTextQuery")
            self._thread.start()

    def submit(self, request: TextQueryRequest) -> bool:
        """Queue ``request`` without blocking; return ``False`` when busy or stopping."""
        with self._state_lock:
            if self._busy or self._stopping.is_set():
                return False
            self._busy = True
            # stop() takes the same lock before publishing its sentinel, so a
            # successful admission cannot race with shutdown and hit Queue.Full.
            self._queue.put_nowait(request)
        return True

    def cancel(self) -> None:
        """Reject new work, discard a queued request, and suppress late callbacks."""
        with self._state_lock:
            self._stopping.set()
            discard_request = not self._stop_enqueued
        if discard_request:
            self._discard_queued_request()

    def stop(self, timeout_s: float = _JOIN_TIMEOUT_S) -> bool:
        """Cancel queued work and give an active model call bounded time to finish.

        Returns ``False`` when decoding is still stuck after ``timeout_s``. The
        worker is a daemon and callbacks stay suppressed, allowing hardware
        teardown to proceed instead of hanging indefinitely.
        """
        self.cancel()
        with self._state_lock:
            thread = self._thread
            enqueue_stop = thread is not None and not self._stop_enqueued
            self._stop_enqueued = self._stop_enqueued or enqueue_stop
        if thread is None:
            return True
        if enqueue_stop:
            self._queue.put(self._STOP)
        thread.join(timeout=timeout_s)
        stopped = not thread.is_alive()
        if stopped:
            with self._state_lock:
                if self._thread is thread:
                    self._thread = None
        return stopped

    def _discard_queued_request(self) -> None:
        try:
            item = self._queue.get_nowait()
        except Empty:
            return
        self._queue.task_done()
        if item is self._STOP:
            # A concurrent/repeated cancel must not consume the sentinel that
            # an earlier stop() already published for the worker.
            self._queue.put_nowait(self._STOP)
            return
        with self._state_lock:
            self._busy = False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                request = item
                assert isinstance(request, TextQueryRequest)
                if self._stopping.is_set():
                    continue
                try:
                    answer = self._answer(request)
                except Exception as exc:  # a language failure must not end robot control
                    self._deliver(self._on_error, request, exc)
                else:
                    self._deliver(self._on_answer, request, answer)
            finally:
                if item is not self._STOP:
                    with self._state_lock:
                        self._busy = False
                self._queue.task_done()

    def _deliver(self, callback: Callable[..., None], *args) -> None:
        """Linearize a result callback with cancellation."""
        with self._state_lock:
            if not self._stopping.is_set():
                callback(*args)


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
        # The instruction the rollout was launched with; /reset restores it.
        self._initial_task = ctx.policy.inference.task
        self._listener = StdinCommandListener(self._handle_line, on_eof=self._handle_eof, stream=input_stream)
        self._text_query = TextQueryWorker(
            self._answer_text_query,
            self._report_text_answer,
            self._report_text_error,
        )

        # Written by the listener thread, consumed by the main loop.
        self._start_requested = Event()
        self._reset_requested = Event()
        self._stop_requested = Event()
        self._wake = Event()
        self._running = Event()

        # name -> (handler, argument hint, help line); /help and the banner
        # render from this table, so future commands stay documented for free.
        self._commands: dict[str, tuple[Callable[[InteractiveCommand], None], str, str]] = {
            "start": (self._cmd_start, "", "start (or restart) the policy control loop"),
            "subtask": (self._cmd_subtask, " <text>", "set the instruction the policy follows"),
            "ask": (self._cmd_ask, " <question>", "ask the policy about the latest view"),
            "reset": (self._cmd_reset, "", "stop movement, return to initial position, restore the task"),
            "stop": (self._cmd_stop, "", "end the session and shut down"),
            "help": (self._cmd_help, "", "show this help"),
        }

    # ------------------------------------------------------------------
    # Main-thread session loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the session until ``/stop``, EOF, engine failure, or a shutdown signal."""
        play_sounds = self._ctx.runtime.cfg.play_sounds
        muted_handlers: list[tuple[logging.Handler, int]] = []
        saved_warning_filters = warnings.filters[:]
        try:
            muted_handlers = _mute_console_log_handlers()
            warnings.simplefilter("ignore")
            self._print(self._render_banner())
            self._text_query.start()
            self._listener.start()
            while not self._global_shutdown.is_set():
                if self._ctx.policy.inference.failed:
                    self._report_engine_failure()
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
            # A model call cannot be force-cancelled safely. Give it a bounded
            # grace period, then prioritize hardware teardown if it is wedged.
            if not self._text_query.stop():
                self._print(
                    "Policy question did not finish within 5 seconds — "
                    "continuing hardware shutdown; its daemon thread will be abandoned."
                )
            # Restore before log_say so teardown logs are visible again.
            _restore_log_handlers(muted_handlers)
            warnings.filters[:] = saved_warning_filters
            log_say("Interactive session ended", play_sounds)

    def _report_engine_failure(self) -> None:
        """Surface a fatal engine error despite the muted console logging."""
        self._print("Inference engine failed — shutting down.")
        failure_traceback = self._ctx.policy.inference.failure_traceback
        if failure_traceback:
            self._print(failure_traceback)
        else:
            self._print("Re-run without --interactive=true to see the error output.")

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
        self._print(
            f"Rollout running — task {_format_task(engine.task)}. "
            "/subtask <text> to change it, /ask <question> to query the policy, "
            "/reset to return to initial position, /stop to shut down."
        )
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
        """Pause inference and return the robot home (the task was restored by ``/reset``)."""
        self._print("Resetting — returning the robot to its initial position...")
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
        handler = entry[0]
        handler(cmd)

    def _handle_eof(self) -> None:
        self._print("Input stream closed — stopping the session.")
        self._request_stop()

    def _cmd_start(self, cmd: InteractiveCommand) -> None:
        if self._running.is_set():
            self._print("Already running — /reset to pause first, or /stop to shut down.")
            return
        if self._text_query.busy:
            self._print("A policy question is still finishing — wait for it before /start.")
            return
        self._start_requested.set()
        self._wake.set()

    def _cmd_subtask(self, cmd: InteractiveCommand) -> None:
        engine = self._ctx.policy.inference
        if not cmd.args:
            self._print(f"Current task: {_format_task(engine.task)}")
            return
        task = _strip_quotes(cmd.args)
        previous = engine.task
        # Publishing the string is all this thread does: the engine applies the
        # switch on its own inference thread.
        if engine.set_task(task):
            self._print(
                f"Task: {_format_task(previous)} → {_format_task(task)} "
                "(applies from the next policy inference)"
            )
        else:
            self._print(f"Task unchanged: {_format_task(task)}")

    def _cmd_ask(self, cmd: InteractiveCommand) -> None:
        question = _strip_quotes(cmd.args)
        if not question:
            self._print("Usage: /ask <question>")
            return
        engine = self._ctx.policy.inference
        if not engine.supports_text_generation():
            self._print("This policy does not support /ask (it has no text-generation head).")
            return
        if not self._running.is_set():
            self._print("The rollout is not running — /start it before using /ask.")
            return
        observation = engine.snapshot_text_observation()
        if observation is None:
            self._print("No policy observation is available yet — /start the rollout and try again.")
            return
        request = TextQueryRequest(question=question, observation=observation)
        if not self._text_query.submit(request):
            self._print("A policy question is already being answered — try again when it finishes.")
            return
        self._print(f"Question queued: {question!r} (the rollout keeps running)")

    def _answer_text_query(self, request: TextQueryRequest) -> str:
        return self._ctx.policy.inference.generate_text(
            request.observation,
            kind=TextKind.VQA,
            user_text=request.question,
        )

    def _report_text_answer(self, request: TextQueryRequest, answer: str) -> None:
        if answer:
            self._print(f"[policy] {answer}")
        else:
            self._print(f"The policy returned no answer for {request.question!r}.")

    def _report_text_error(self, request: TextQueryRequest, exc: Exception) -> None:
        self._print(f"Policy question failed ({type(exc).__name__}): {exc}")

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        # Last command wins: a /start still waiting to be serviced is cancelled
        # so the robot never starts moving after the operator asked it not to.
        # Flag first, segment-stop second (see the ordering note in _run_segment).
        self._start_requested.clear()
        # Restore the task here rather than in _reset_robot (which runs later, on
        # the main thread) so that both task writers run on this thread and are
        # ordered by command order — otherwise a /subtask typed right after
        # /reset would be silently reverted by the deferred restore.
        engine = self._ctx.policy.inference
        if engine.set_task(self._initial_task):
            self._print(f"Task restored to {_format_task(self._initial_task)}")
        # Homing changes the scene outside normal inference. Invalidate the
        # cached VLM input synchronously so a following /ask cannot capture
        # the pre-reset view while the main thread is still unwinding.
        engine.invalidate_text_observation()
        self._reset_requested.set()
        self._segment_stop.set()
        self._wake.set()

    def _cmd_stop(self, cmd: InteractiveCommand) -> None:
        self._request_stop()

    def _request_stop(self) -> None:
        # Suppress a queued/finishing answer as soon as /stop or EOF is
        # observed; stop() in the session's finally block gives the model call
        # bounded time to finish before hardware teardown continues.
        self._text_query.cancel()
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
        usages = {name: f"/{name}{entry[1]}" for name, entry in self._commands.items()}
        width = max(len(usage) for usage in usages.values())
        lines = [f"  {usages[name]:<{width}}   {entry[2]}" for name, entry in self._commands.items()]
        return "Available commands:\n" + "\n".join(lines)

    def _render_banner(self) -> str:
        return (
            f"{_BANNER_RULE}\n"
            "Interactive rollout session — the robot will NOT move until you type /start.\n"
            f"Task: {_format_task(self._initial_task)}\n"
            f"{self._render_help()}\n"
            "System logs and warnings are muted during the session; they resume when it ends.\n"
            f"{_BANNER_RULE}"
        )

    @staticmethod
    def _print(message: str) -> None:
        """User-facing chat output; logging stays on stderr, replies on stdout."""
        print(message, flush=True)
