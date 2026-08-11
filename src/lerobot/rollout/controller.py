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

"""Programmatic control of a rollout: start, pause, re-instruct, and stop a
policy while hardware and policy stay connected and warm.

:class:`RolloutController` is the embedding-friendly core of interactive
rollouts.  It has no I/O of its own — no stdin, no printing, no log
manipulation — so it can be driven from any application code: a CLI
(:class:`lerobot.rollout.interactive.InteractiveSession` is exactly that), a
network server, a voice front-end, or a notebook.

Typical embedding::

    from threading import Event, Thread
    from lerobot.rollout import (
        LinkedEvent,
        RolloutController,
        build_rollout_context,
        create_strategy,
    )

    parent = Event()  # your application's shutdown signal
    ctx = build_rollout_context(cfg, LinkedEvent(parent))
    strategy = create_strategy(cfg.strategy)
    strategy.setup(ctx)

    controller = RolloutController(strategy, ctx)
    serve_thread = Thread(target=controller.serve)
    serve_thread.start()  # or call serve() on your main thread

    controller.start()  # robot starts executing the policy
    controller.set_task("grab the red cube")  # re-instruct mid-run
    controller.reset()  # stop movement, return home, stay warm
    controller.stop()  # end serve()

    serve_thread.join()
    strategy.teardown(ctx)  # teardown stays with the caller
"""

from __future__ import annotations

import logging
import time
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

    ``set``/``clear`` act only on the local flag, so a controller can raise
    and clear its own segment-stop requests without masking (or accidentally
    re-arming) the process-wide shutdown event carried by ``parent``.  Every
    rollout strategy control loop polls ``ctx.runtime.shutdown_event.is_set()``,
    so installing a ``LinkedEvent`` there makes the loops react both to
    controller commands and to real shutdown signals.
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


class AskResult(Enum):
    """Outcome of :meth:`RolloutController.ask`."""

    QUEUED = "queued"
    """The question was accepted; the answer arrives as a ``QUERY_ANSWERED`` event."""

    NOT_RUNNING = "not_running"
    """Rejected: no segment is running, so no fresh observation is flowing."""

    BUSY = "busy"
    """Rejected: another query holds the channel: a previous question, or
    an autosteer subtask query queued but not yet served."""

    UNSUPPORTED = "unsupported"
    """Rejected: the policy has no text head, so it can neither answer
    questions nor plan subtasks.  Unlike the other rejections this one is
    permanent for the session."""


class RolloutEvent(Enum):
    """Lifecycle notifications emitted by :class:`RolloutController`.

    All events are emitted on the thread running :meth:`RolloutController.serve`;
    callbacks must be quick and must not call back into the controller's
    blocking methods.
    """

    SEGMENT_STARTED = "segment_started"
    """A control-loop segment is about to run (control state freshly reset)."""

    SEGMENT_ENDED = "segment_ended"
    """The segment returned on its own (e.g. ``--duration`` elapsed); the
    controller is idle again and the robot is holding position."""

    RESET_STARTED = "reset_started"
    """A reset is being executed: inference paused, robot about to move home."""

    RESET_DONE = "reset_done"
    """The robot is back at its initial position, holding."""

    RESET_SKIPPED = "reset_skipped"
    """No initial position was captured; the robot holds its current pose."""

    QUERY_ANSWERED = "query_answered"
    """A text query was resolved: a question queued with
    :meth:`RolloutController.ask`, or one of the autosteer sequencer's turns
    (kind ``NEXT_SUBTASK`` — a success carries the subtask just applied to the
    task, a failure the reason the sequencer stopped).  The event payload is a
    :class:`~lerobot.rollout.inference.QueryAnswer`; check its ``ok`` before
    reading ``answer``, since it also reports questions the policy could not
    handle and ones dropped when the run ended first."""

    ENGINE_FAILED = "engine_failed"
    """The inference engine hit an unrecoverable error; ``serve()`` is about
    to return.  Read :attr:`RolloutController.failure_traceback` for details."""

    STOPPED = "stopped"
    """``serve()`` is returning (after :meth:`RolloutController.stop`, EOF of
    the driving front-end, an engine failure, or a parent shutdown signal)."""


class RolloutController:
    """Drive a rollout strategy through thread-safe start/reset/stop/set_task calls.

    The controller owns the outer lifecycle between ``strategy.setup(ctx)``
    and ``strategy.teardown(ctx)`` (both stay with the caller): the robot is
    idle until :meth:`start`, each run *segment* executes ``strategy.run(ctx)``
    on the thread that called :meth:`serve` until interrupted or until the
    strategy returns on its own (e.g. ``--duration`` elapsed).  :meth:`reset`
    pauses the inference engine, returns the robot to its initial position,
    and restores the launch task, while hardware and policy stay warm.
    :meth:`stop` ends :meth:`serve` so the caller can run
    ``strategy.teardown(ctx)``.  :meth:`ask` queues a question for the
    policy's text head and reports the answer through ``on_event``.

    Requires ``ctx.runtime.shutdown_event`` to be a :class:`LinkedEvent`: the
    controller sets the local flag to end a segment, and process shutdown
    signals still propagate through the parent.  Build the context with
    ``build_rollout_context(cfg, LinkedEvent(shutdown_event))``.

    Thread safety: the control methods (:meth:`start`, :meth:`reset`,
    :meth:`stop`, :meth:`set_task`) may be called from any thread and are
    serialized by an internal lock, so calls issued in order from one thread
    keep that order — e.g. a ``set_task`` right after a ``reset`` is not
    clobbered by the reset's task restore.  Commands are last-write-wins:
    ``reset`` and ``stop`` cancel a still-pending ``start`` so the robot
    never starts moving after the caller's most recent command asked it not
    to.  Events are emitted on the :meth:`serve` thread via ``on_event``.
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
        self._strategy = strategy
        self._ctx = ctx
        self._segment_stop = stop_event
        self._global_shutdown = stop_event.parent
        self._on_event = on_event
        # The instruction the rollout was launched with; reset() restores it.
        self._initial_task = ctx.policy.inference.task
        self._autosteer_interval_s = ctx.runtime.cfg.autosteer_interval_s

        # Serializes the control methods so multi-writer task updates (e.g.
        # reset()'s restore followed by a set_task()) keep their call order.
        self._control_lock = Lock()

        # Written by control methods (any thread), consumed by the serve loop.
        self._start_requested = Event()
        self._reset_requested = Event()
        self._stop_requested = Event()
        self._wake = Event()
        self._running = Event()

        # Answers are produced on whichever thread owns the policy, but the
        # engine only hands them over from ``pump_query`` — which the control
        # loop and the idle poll below both call on the serve thread — so this
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
    def failed(self) -> bool:
        """True if the inference engine hit an unrecoverable error."""
        return self._ctx.policy.inference.failed

    @property
    def failure_traceback(self) -> str | None:
        """Formatted traceback of the engine failure, when :attr:`failed` is True."""
        return self._ctx.policy.inference.failure_traceback

    # ------------------------------------------------------------------
    # Control methods (callable from any thread)
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Request a control-loop segment.

        Returns ``False`` when a segment is already running (the request is
        ignored); ``True`` when the segment was scheduled.  The segment itself
        executes on the :meth:`serve` thread.
        """
        with self._control_lock:
            if self._running.is_set():
                return False
            self._start_requested.set()
            self._wake.set()
            return True

    def reset(self) -> bool:
        """Stop movement, return the robot to its initial position, restore the launch task.

        Hardware and policy stay warm; call :meth:`start` to run again.
        Returns ``True`` when the task was restored to the launch task (i.e.
        it had been changed), ``False`` when it was already the launch task.
        """
        with self._control_lock:
            # Last command wins: a start() still waiting to be serviced is
            # cancelled so the robot never starts moving after the caller
            # asked it not to.  Flag first, segment-stop second (see the
            # ordering note in _run_segment).
            self._start_requested.clear()
            # Reset means back to square one, so the sequencer stops too —
            # otherwise it would overwrite the launch task being restored below.
            self._ctx.policy.inference.stop_autosteer()
            # Restore the task here, under the control lock, rather than in
            # _reset_robot (which runs later, on the serve thread) so that a
            # set_task() issued right after this reset() is not silently
            # reverted by a deferred restore.
            restored = self._ctx.policy.inference.set_task(self._initial_task)
            self._reset_requested.set()
            self._segment_stop.set()
            self._wake.set()
            return restored

    def stop(self) -> None:
        """End :meth:`serve`; the caller then runs ``strategy.teardown(ctx)``."""
        with self._control_lock:
            self._start_requested.clear()  # last command wins, see reset()
            self._stop_requested.set()
            self._segment_stop.set()
            self._wake.set()

    def set_task(self, task: str) -> bool:
        """Change the instruction the policy follows, effective from the next inference.

        Returns ``True`` when the value actually changed.  Safe to call while
        a segment is running: the engine applies the switch on its own
        inference thread (sync backends also drop actions precomputed under
        the previous instruction).

        Setting the instruction by hand stops :meth:`autosteer`: the operator
        is taking the wheel, and leaving the sequencer on would silently
        overwrite this instruction at the next interval.
        """
        with self._control_lock:
            self._ctx.policy.inference.stop_autosteer()
            return self._ctx.policy.inference.set_task(task)

    def ask(self, question: str) -> AskResult:
        """Queue a question about what the robot currently sees.

        Returns immediately; the answer arrives later as a
        :attr:`RolloutEvent.QUERY_ANSWERED` event carrying a
        :class:`~lerobot.rollout.inference.QueryAnswer`.  The policy is
        never touched on the caller's thread — the question is answered by
        whichever thread owns the policy, which for async backends means the
        robot keeps executing already-queued actions and then holds while the
        text head runs.

        Refused outright (:attr:`AskResult.UNSUPPORTED`) when the policy has
        no text head — checked here, before queueing, so the operator learns
        immediately rather than from an error answer a tick later.

        Only accepted while a segment is running (:attr:`running`).  Engines
        are fed observations by the control loop and by nothing else, so an
        idle engine has no current view to answer from — and an async
        backend's inference thread is parked and would never pick the
        question up at all.
        """
        # A static capability, so checked outside the control lock — and
        # first, so the operator is not told to /start a policy that could
        # never answer.
        if not self._ctx.policy.inference.supports_text_queries:
            return AskResult.UNSUPPORTED
        with self._control_lock:
            # Checked under the same lock that _run_segment clears _running
            # and drops the pending question under, so a question can never
            # slip past this guard and be left orphaned in the slot.
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

        Every ``autosteer_interval_s`` seconds the engine asks the policy
        for the next subtask and applies it through :meth:`set_task`,
        so the switch takes the usual instruction-change path.  Progress
        through the plan lives in the policy — each query re-sends the same
        goal — which is why the sequencer does not survive a segment:
        restarting a segment resets the policy, and with it the plan.

        Subject to the same capability and running-only guards as :meth:`ask`,
        and stopped by :meth:`reset`, :meth:`set_task`, and the end of a
        segment.  Returns :attr:`AskResult.NOT_RUNNING` when no segment is
        running and :attr:`AskResult.UNSUPPORTED` when the policy has no text
        head to plan with.
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
        """Service control requests until :meth:`stop`, engine failure, or parent shutdown.

        Blocks the calling thread; run segments execute here.  Emits
        :class:`RolloutEvent` notifications through ``on_event``.
        """
        try:
            while not self._global_shutdown.is_set():
                if self._ctx.policy.inference.failed:
                    self._emit(RolloutEvent.ENGINE_FAILED)
                    break
                if self._stop_requested.is_set():
                    break
                if self._reset_requested.is_set():
                    self._reset_requested.clear()
                    self._reset_robot()
                    continue
                if self._start_requested.is_set():
                    # Consume the request and mark the segment running in one
                    # atomic step: start() gates on _running, so a concurrent
                    # start() is rejected for the entire startup sequence
                    # (reset_control_state, SEGMENT_STARTED emission), not just
                    # once strategy.run() begins — otherwise it could re-arm
                    # _start_requested behind the running segment and the robot
                    # would start again, uncommanded, when the segment ends.
                    with self._control_lock:
                        starting = self._start_requested.is_set()
                        if starting:
                            self._start_requested.clear()
                            self._running.set()
                    if starting:
                        self._run_segment()
                    continue
                # Deliver an answer that landed just as the segment ended.
                # While a segment runs the control loop pumps every tick; this
                # is the idle counterpart.  No observation to offer, so a
                # pending question stays queued rather than being answered
                # blind — _run_segment has already dropped it anyway.
                self._ctx.policy.inference.pump_query()
                self._wake.wait(timeout=self._POLL_INTERVAL_S)
                self._wake.clear()
        finally:
            self._emit(RolloutEvent.STOPPED)

    def _run_segment(self) -> None:
        """Execute one ``strategy.run`` segment until interrupted or finished.

        The serve loop has already set ``_running`` (under the control lock),
        so this method must clear it on every exit path.
        """
        engine = self._ctx.policy.inference
        try:
            # Clear the local flag *before* checking the request flags: control
            # methods set their flag first and the segment-stop event second, so
            # a reset() or stop() racing with this start() is either seen here or
            # ends the freshly started loop on its first tick.
            self._segment_stop.clear()
            if (
                self._stop_requested.is_set()
                or self._reset_requested.is_set()
                or self._global_shutdown.is_set()
            ):
                return
            self._strategy.reset_control_state()
            self._emit(RolloutEvent.SEGMENT_STARTED)
            try:
                self._strategy.run(self._ctx)
            finally:
                engine.pause()
        finally:
            # Clear and drop together under the control lock: ask() gates on
            # _running under the same lock, so a question either lands before
            # this (and is dropped here) or is rejected outright.  Left in the
            # slot it would be answered against a completely different scene
            # whenever the robot next starts.
            with self._control_lock:
                self._running.clear()
                # The sequencer cannot outlive the segment: restarting one
                # resets the policy, and the plan's progress lives there.
                engine.stop_autosteer()
                dropped = engine.drop_pending_query()
            # Only an operator question is worth reporting — a dropped
            # next-subtask query is just the sequencer stopping with it.
            if dropped is not None and dropped.kind is QueryKind.VQA:
                self._emit(
                    RolloutEvent.QUERY_ANSWERED,
                    QueryAnswer(question=dropped.text, error="the run ended before it could be answered"),
                )
        if engine.failed:
            return  # the serve loop emits ENGINE_FAILED and shuts down
        if not (
            self._stop_requested.is_set() or self._reset_requested.is_set() or self._global_shutdown.is_set()
        ):
            self._emit(RolloutEvent.SEGMENT_ENDED)

    def _reset_robot(self) -> None:
        """Pause inference and return the robot home (the task was restored by :meth:`reset`)."""
        self._emit(RolloutEvent.RESET_STARTED)
        self._ctx.policy.inference.pause()
        if self._ctx.hardware.initial_position:
            self._strategy.return_to_initial_position(self._ctx.hardware)
            self._emit(RolloutEvent.RESET_DONE)
        else:
            logger.warning("No initial position captured — skipping the return move")
            self._emit(RolloutEvent.RESET_SKIPPED)

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
