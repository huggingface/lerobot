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

"""Programmatic control of a rollout: start, pause, re-instruct, and stop a policy while
hardware and policy stay connected and warm.

:class:`RolloutController` is the embedding-friendly core: it has no I/O of its own, so it can be
driven from a CLI (:class:`lerobot.rollout.interactive.InteractiveSession`), a network server, or a
notebook.  See ``docs/source/inference.mdx`` for a worked embedding example.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from enum import Enum
from threading import Event, Lock
from typing import TYPE_CHECKING

from .inference import QueryAnswer, QueryKind

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)


class LinkedEvent(Event):
    """A ``threading.Event`` whose ``is_set`` also reflects a parent event.

    ``set``/``clear`` act only on the local flag, so a controller can raise and clear its own
    segment-stop requests without masking (or re-arming) the shutdown event carried by ``parent``.
    """

    _WAIT_SLICE_S = 0.05

    def __init__(self, parent: Event) -> None:
        super().__init__()
        self.parent = parent

    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for either the local or the parent flag, polling in short slices."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None and remaining <= 0:
                return False
            wait_slice = self._WAIT_SLICE_S if remaining is None else min(self._WAIT_SLICE_S, remaining)
            super().wait(wait_slice)
        return True


class AskResult(Enum):
    """Outcome of :meth:`RolloutController.ask`."""

    QUEUED = "queued"
    """Accepted; the answer arrives as a ``QUERY_ANSWERED`` event."""

    NOT_RUNNING = "not_running"
    """Rejected: no segment is running, so no fresh observation is flowing."""

    BUSY = "busy"
    """Rejected: another question or autosteer turn holds the single-slot channel."""

    UNSUPPORTED = "unsupported"
    """Rejected: the policy has no text head; unlike the others, permanent for the session."""


class RolloutEvent(Enum):
    """Lifecycle notifications emitted by :class:`RolloutController`.

    All events fire on the thread running :meth:`RolloutController.serve`; callbacks must be
    quick and must not call back into the controller's blocking methods.
    """

    SEGMENT_STARTED = "segment_started"
    """A control-loop segment is about to run (control state freshly reset)."""

    SEGMENT_ENDED = "segment_ended"
    """The segment returned on its own (e.g. ``--duration`` elapsed); the robot is holding."""

    RESET_STARTED = "reset_started"
    """A reset is being executed: inference paused, robot about to move home."""

    RESET_DONE = "reset_done"
    """The robot is back at its initial position, holding."""

    RESET_SKIPPED = "reset_skipped"
    """No initial position was captured; the robot holds its current pose."""

    RESET_FAILED = "reset_failed"
    """The return move errored partway: the robot may be holding an arbitrary pose, *not* the
    initial position."""

    QUERY_ANSWERED = "query_answered"
    """A text query resolved (an :meth:`RolloutController.ask` question or an autosteer turn); the
    payload is a :class:`~lerobot.rollout.inference.QueryAnswer`, check ``ok`` before ``answer``."""

    ENGINE_FAILED = "engine_failed"
    """The engine hit an unrecoverable error; ``serve()`` is returning.  Read
    :attr:`RolloutController.failure_traceback` (same for ``STRATEGY_FAILED``)."""

    STRATEGY_FAILED = "strategy_failed"
    """``strategy.run()`` raised mid-segment (robot I/O, recording, ...); ``serve()`` is returning."""

    STOPPED = "stopped"
    """``serve()`` is returning (stop, front-end EOF, a failure, or a parent shutdown signal)."""


class RolloutController:
    """Drive a rollout strategy through thread-safe start/reset/stop/set_task calls.

    The robot is idle until :meth:`start`; each run *segment* executes ``strategy.run(ctx)`` on the
    thread that called :meth:`serve`, while ``strategy.setup``/``teardown`` stay with the caller.

    - ``ctx.runtime.shutdown_event`` must be a :class:`LinkedEvent`, so ending a segment does not
      trigger process shutdown: ``build_rollout_context(cfg, LinkedEvent(shutdown_event))``.
    - The control methods are callable from any thread and serialized by an internal lock, so calls
      issued in order from one thread keep that order.  Events fire on the :meth:`serve` thread.
    - One-shot: once :meth:`serve` returns the controller is terminally :attr:`stopped`, the control
      methods refuse with ``False``, and a second :meth:`serve` call raises.
    """

    _POLL_INTERVAL_S = 0.2

    def __init__(
        self,
        strategy: RolloutStrategy,
        ctx: RolloutContext,
        on_event: Callable[[RolloutEvent, QueryAnswer | None], None] | None = None,
    ) -> None:
        stop_event = ctx.runtime.shutdown_event
        if not isinstance(stop_event, LinkedEvent):
            raise TypeError(
                "RolloutController requires ctx.runtime.shutdown_event to be a LinkedEvent so "
                "reset() can end a run segment without triggering process shutdown. Build the "
                "rollout context with build_rollout_context(cfg, LinkedEvent(shutdown_event))."
            )
        if not strategy.config.supports_interactive:
            # One-shot strategies finalize their dataset when run() exits, so a second start()
            # would record into a finalized dataset (same guard as RolloutConfig.__post_init__).
            raise ValueError(
                f"RolloutController drives strategy.run() in restartable segments, but "
                f"'{strategy.config.type}' is a one-shot strategy "
                f"(supports_interactive is False). Use a strategy that honours the "
                f"restartable-run() contract (see RolloutStrategy in strategies/core.py)."
            )
        self._strategy = strategy
        self._ctx = ctx
        self._segment_stop = stop_event
        self._global_shutdown = stop_event.parent
        self._on_event = on_event
        self._initial_task = ctx.policy.inference.task
        self._autosteer_interval_s = ctx.runtime.cfg.autosteer_interval_s

        # Serializes the control methods so multi-writer task updates keep their call order.
        self._control_lock = Lock()

        # Written by control methods (any thread), consumed by the serve loop.
        self._start_requested = Event()
        self._reset_requested = Event()
        self._stop_requested = Event()
        self._wake = Event()
        self._running = Event()
        # Latched (never cleared) when serve() exits; the control methods then refuse.
        self._stopped = Event()
        self._strategy_failure_traceback: str | None = None

        # Answers only leave the engine through pump_query(), called on the serve thread, so this
        # observer keeps the "events fire on the serve thread" guarantee.
        ctx.policy.inference.set_answer_observer(self._on_query_answer)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """The language instruction currently conditioning inference."""
        return self._ctx.policy.inference.task

    @property
    def initial_task(self) -> str:
        """The instruction the rollout was launched with (restored by :meth:`reset`)."""
        return self._initial_task

    @property
    def running(self) -> bool:
        """True while a control-loop segment is executing."""
        return self._running.is_set()

    @property
    def stopped(self) -> bool:
        """True once :meth:`serve` has returned; the controller is terminal (one-shot)."""
        return self._stopped.is_set()

    @property
    def failed(self) -> bool:
        """True if the engine or the strategy hit an unrecoverable error."""
        return self._ctx.policy.inference.failed or self._strategy_failure_traceback is not None

    @property
    def failure_traceback(self) -> str | None:
        """Formatted traceback of the failure, when :attr:`failed` is True."""
        return self._strategy_failure_traceback or self._ctx.policy.inference.failure_traceback

    # ------------------------------------------------------------------
    # Control methods (callable from any thread)
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Request a control-loop segment, which executes on the :meth:`serve` thread.

        Returns ``True`` when the segment was scheduled, ``False`` when one is already running, after a
        failure, or once the controller is stopping or stopped (an accepted start would never run).
        """
        with self._control_lock:
            if self._stopped.is_set() or self._stop_requested.is_set() or self.failed:
                return False
            if self._running.is_set():
                return False
            self._start_requested.set()
            self._wake.set()
            return True

    def reset(self) -> bool:
        """Stop movement, return the robot to its initial position, restore the launch task.

        Hardware and policy stay warm; call :meth:`start` to run again.  Returns ``True`` when the
        task was restored (i.e. it had been changed), ``False`` when it was already the launch task
        or the controller is stopping or stopped.
        """
        with self._control_lock:
            if self._stopped.is_set() or self._stop_requested.is_set():
                return False
            # Last command wins: cancel a pending start().  Flag first, segment-stop second
            # (see the ordering note in _run_segment).
            self._start_requested.clear()
            # Back to square one, so the sequencer stops too: it would overwrite the restored task.
            self._ctx.policy.inference.stop_autosteer()
            # Restore here, not later on the serve thread, so a following set_task() survives.
            restored = self._ctx.policy.inference.set_task(self._initial_task)
            self._reset_requested.set()
            self._segment_stop.set()
            self._wake.set()
            return restored

    def stop(self) -> None:
        """End :meth:`serve` so the caller can run ``strategy.teardown(ctx)``.  Idempotent."""
        with self._control_lock:
            if self._stopped.is_set():
                return
            self._start_requested.clear()  # last command wins, see reset()
            self._stop_requested.set()
            self._segment_stop.set()
            self._wake.set()

    def set_task(self, task: str) -> bool:
        """Change the instruction the policy follows, effective from the next inference.

        Returns ``True`` when the value actually changed.  Safe to call while a segment is running:
        the engine applies the switch on its own inference thread (sync backends also drop actions
        precomputed under the previous instruction).  Refused (``False``, engine untouched) once the
        controller is stopping or stopped.  Stops :meth:`autosteer`, which would overwrite this instruction.
        """
        with self._control_lock:
            if self._stopped.is_set() or self._stop_requested.is_set():
                return False
            self._ctx.policy.inference.stop_autosteer()
            return self._ctx.policy.inference.set_task(task)

    def ask(self, question: str) -> AskResult:
        """Queue a question about what the robot currently sees.

        Returns immediately; the answer arrives as a :attr:`RolloutEvent.QUERY_ANSWERED` event, and
        the policy is never touched on the caller's thread.  Rejected with
        :attr:`AskResult.UNSUPPORTED` (no text head), :attr:`AskResult.NOT_RUNNING` (no segment
        running, so no observation to answer from), or :attr:`AskResult.BUSY` (channel taken).
        """
        # A static capability: checked first, and outside the control lock.
        if not self._ctx.policy.inference.supports_text_queries:
            return AskResult.UNSUPPORTED
        with self._control_lock:
            # Same lock _run_segment clears _running under, so a question is never left orphaned.
            if not self._running.is_set():
                return AskResult.NOT_RUNNING
            if not self._ctx.policy.inference.ask(question):
                return AskResult.BUSY
            return AskResult.QUEUED

    @property
    def autosteer_goal(self) -> str | None:
        """The high-level goal currently driving the task, if any."""
        return self._ctx.policy.inference.autosteer_goal

    def autosteer(self, goal: str) -> AskResult:
        """Let the policy decompose ``goal`` and drive its own subtasks.

        Every ``autosteer_interval_s`` seconds the engine asks the policy for the next subtask and
        applies it through the *engine's* ``set_task``.  Plan progress lives in the policy, so the
        sequencer does not survive a segment; it is also stopped by :meth:`reset` and
        :meth:`set_task`.  Same guards and rejection values as :meth:`ask`.
        """
        if not self._ctx.policy.inference.supports_text_queries:
            return AskResult.UNSUPPORTED
        with self._control_lock:
            if not self._running.is_set():
                return AskResult.NOT_RUNNING
            self._ctx.policy.inference.start_autosteer(goal, self._autosteer_interval_s)
            return AskResult.QUEUED

    def stop_autosteer(self) -> str | None:
        """Stop the sequencer, returning the goal it was driving (or ``None``)."""
        with self._control_lock:
            return self._ctx.policy.inference.stop_autosteer()

    # ------------------------------------------------------------------
    # Serve loop (blocks the calling thread)
    # ------------------------------------------------------------------

    def serve(self) -> None:
        """Service control requests until :meth:`stop`, a failure, or parent shutdown.

        Blocks the calling thread; run segments execute here and :class:`RolloutEvent`
        notifications are emitted through ``on_event``.  One-shot: once it returns the controller
        is terminally :attr:`stopped` and calling ``serve()`` again raises.
        """
        if self._stopped.is_set():
            raise RuntimeError(
                "RolloutController.serve() is one-shot: this controller has already stopped. "
                "Build a new controller to run again."
            )
        try:
            while not self._global_shutdown.is_set():
                if self._ctx.policy.inference.failed:
                    self._emit(RolloutEvent.ENGINE_FAILED)
                    break
                if self._strategy_failure_traceback is not None:
                    self._emit(RolloutEvent.STRATEGY_FAILED)
                    break
                if self._stop_requested.is_set():
                    break
                if self._reset_requested.is_set():
                    self._reset_requested.clear()
                    self._reset_robot()
                    continue
                if self._start_requested.is_set():
                    # Consume the request and mark the segment running in one atomic step, so a
                    # concurrent start() cannot re-arm the flag behind the running segment.
                    with self._control_lock:
                        starting = self._start_requested.is_set()
                        if starting:
                            self._start_requested.clear()
                            self._running.set()
                    if starting:
                        self._run_segment()
                    continue
                # Idle counterpart of the per-tick pump in the control loop: deliver an answer
                # that landed just as the segment ended.
                self._ctx.policy.inference.pump_query()
                self._wake.wait(timeout=self._POLL_INTERVAL_S)
                self._wake.clear()
        finally:
            # Latch before announcing, so an observer reacting to STOPPED sees a stopped controller.
            self._stopped.set()
            self._emit(RolloutEvent.STOPPED)

    def _run_segment(self) -> None:
        """Execute one ``strategy.run`` segment until interrupted or finished.

        The serve loop has already set ``_running`` (under the control lock), so this method must
        clear it on every exit path.
        """
        engine = self._ctx.policy.inference
        try:
            # Clear the local flag *before* checking the request flags: control methods set their flag
            # first and the event second, so a racing reset()/stop() is either seen here or ends the
            # fresh loop at once.  The clear also absorbs a dying engine's signal: hence engine.failed.
            self._segment_stop.clear()
            if (
                self._stop_requested.is_set()
                or self._reset_requested.is_set()
                or self._global_shutdown.is_set()
                or engine.failed
            ):
                return
            self._strategy.reset_control_state()
            self._emit(RolloutEvent.SEGMENT_STARTED)
            try:
                self._strategy.run(self._ctx)
            except Exception:
                # Route to the same public failure surface as an engine failure, instead of
                # unwinding through serve() as a clean-looking STOPPED.
                self._strategy_failure_traceback = traceback.format_exc()
                logger.exception("Rollout strategy failed mid-segment")
            finally:
                engine.pause()
        finally:
            # Clear and drop together under the control lock: ask() gates on _running under the same
            # lock, so a question either lands before this and is dropped, or is rejected outright.
            with self._control_lock:
                self._running.clear()
                # The sequencer cannot outlive the segment: its plan progress lives in the policy.
                engine.stop_autosteer()
                dropped = engine.drop_pending_query()
                # Else the idle pump would announce a subtask after the sequencer ended; VQA stays.
                engine.drop_ready_subtask_answers()
            # Only an operator question is worth reporting.
            if dropped is not None and dropped.kind is QueryKind.VQA:
                self._emit(
                    RolloutEvent.QUERY_ANSWERED,
                    QueryAnswer(question=dropped.text, error="the run ended before it could be answered"),
                )
        if engine.failed or self._strategy_failure_traceback is not None:
            return  # the serve loop emits the failure event and shuts down
        if not (
            self._stop_requested.is_set() or self._reset_requested.is_set() or self._global_shutdown.is_set()
        ):
            self._emit(RolloutEvent.SEGMENT_ENDED)

    def _reset_robot(self) -> None:
        """Pause inference and return the robot home (the task was restored by :meth:`reset`)."""
        self._emit(RolloutEvent.RESET_STARTED)
        self._ctx.policy.inference.pause()
        if not self._ctx.hardware.initial_position:
            logger.warning("No initial position captured — skipping the return move")
            self._emit(RolloutEvent.RESET_SKIPPED)
        elif self._strategy.return_to_initial_position(self._ctx.hardware):
            self._emit(RolloutEvent.RESET_DONE)
        else:
            # RESET_DONE guarantees "back at the initial position"; a failed move must not claim it.
            self._emit(RolloutEvent.RESET_FAILED)

    def _on_query_answer(self, answer: QueryAnswer) -> None:
        """Engine answer observer — runs on the serve thread (see ``__init__``)."""
        self._emit(RolloutEvent.QUERY_ANSWERED, answer)

    def _emit(self, event: RolloutEvent, payload: QueryAnswer | None = None) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event, payload)
        except Exception:  # a broken observer must not kill the serve loop
            logger.exception("Error in RolloutController event callback for %s", event)
