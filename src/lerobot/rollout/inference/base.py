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

"""Inference engine ABC.

Rollout strategies consume actions through this small interface so they
do not need to know whether inference happens inline on the control thread
or asynchronously in a background thread (RTC).
"""

from __future__ import annotations

import abc
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock

import torch

from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

logger = logging.getLogger(__name__)


class QueryKind(Enum):
    """What the policy's text head is being asked for.

    Both kinds return text, but they are preprocessed differently — a
    high-level goal is formatted so the reply is exactly one subtask — so
    the kind travels in the batch alongside the query text (see
    :meth:`InferenceEngine._mark_query`).
    """

    VQA = "vqa"
    """A free-form question about the current scene; the reply goes to the operator."""

    NEXT_SUBTASK = "next_subtask"
    """A high-level goal; the reply is the next subtask and is fed to ``set_task``."""


@dataclass(frozen=True)
class PolicyQuery:
    """A queued request for the policy's text head."""

    kind: QueryKind
    text: str


@dataclass(frozen=True)
class QueryAnswer:
    """Result of a policy text query.

    Exactly one of ``answer`` / ``error`` is set: ``error`` carries the
    reason the query could not be served (policy without a text head,
    inference failure, run ended first) so the caller can report it instead
    of silently dropping it.

    ``NEXT_SUBTASK`` answers report the sequencer's turns: a success carries
    the subtask the engine has *already applied* through ``set_task`` (the
    receiver only announces it, never applies it again), a failure the reason
    the sequencer just stopped.
    """

    question: str
    answer: str | None = None
    error: str | None = None
    kind: QueryKind = QueryKind.VQA

    @property
    def ok(self) -> bool:
        """True when the policy produced an answer."""
        return self.error is None


class InferenceEngine(abc.ABC):
    """Abstract backend for producing actions during rollout.

    Subclasses decide whether inference happens inline on the control
    thread or asynchronously in a background thread.  The contract is
    minimal so additional backends can be plugged in without touching
    rollout strategies.

    Lifecycle
    ---------
    ``start`` — prepare the backend (e.g. launch a background thread).
    ``stop`` — shut the backend down cleanly.
    ``reset`` — clear episode-scoped state (policy hidden state, queues…).

    Action production
    -----------------
    ``get_action(obs_frame)`` — return the next action tensor, or
    ``None`` if none is available (e.g. async queue empty).  Sync
    backends always compute from ``obs_frame``; async backends ignore
    it (they receive observations via ``notify_observation``).

    Task
    ----
    ``task`` / ``set_task`` hold the language instruction the policy is
    conditioned on.  ``set_task`` is safe to call from any thread (the
    interactive session's ``/subtask`` command calls it from its stdin
    reader); subclasses pick the new value up on their own inference
    thread via :meth:`_take_task`, so no policy state is ever mutated
    across threads.  ``dispatched_task`` names the instruction that
    generated the most recently returned action — it trails ``task``
    while actions produced under the previous instruction are still
    being consumed.

    Text queries
    ------------
    ``ask`` queues a question for the policy's text head.  Like
    ``set_task`` it is safe to call from any thread (the interactive
    session's ``/vqa`` command calls it from its stdin reader) and never
    touches the policy itself: the question is answered by the thread
    that *owns* the policy — the control thread for sync backends, the
    background inference thread for async ones — through
    :meth:`_service_query`.  The answer lands in a slot and is handed to
    the observer registered via :meth:`set_answer_observer` when the
    control thread next calls :meth:`pump_query`, so observers never fire
    on a background inference thread.  Whether the channel can be served
    at all is reported by :attr:`supports_text_queries`, which callers
    check *before* queueing so a policy without a text head is refused up
    front instead of failing one tick later.

    Optional hooks
    --------------
    ``notify_observation`` / ``pause`` / ``resume`` have a no-op default
    so rollout strategies can invoke them unconditionally.

    Subclasses must call ``super().__init__(task=...)``; the task and
    query holders are set up there.
    """

    def __init__(self, task: str = "") -> None:
        self._task = task
        self._task_changed = False
        self._dispatched_task = task
        self._task_lock = Lock()

        # Text-query channel.  Its own lock, so a slow generate never
        # blocks a concurrent ``set_task`` (and vice versa).
        self._query_lock = Lock()
        self._pending_query: PolicyQuery | None = None
        self._ready_answer: QueryAnswer | None = None
        self._answer_observer: Callable[[QueryAnswer], None] | None = None

        # Autosteer sequencer state (same lock: it writes _pending_query).
        self._autosteer_goal: str | None = None
        self._autosteer_interval_s: float = 0.0
        self._autosteer_due_at: float = 0.0

    # ------------------------------------------------------------------
    # Task (language instruction)
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """The language instruction currently conditioning inference."""
        with self._task_lock:
            return self._task

    def set_task(self, task: str) -> bool:
        """Set the instruction used from the next inference onwards.

        Callable from any thread.  Returns ``True`` when the value
        actually changed, so callers can report no-op switches.
        """
        with self._task_lock:
            if task == self._task:
                return False
            previous, self._task = self._task, task
            self._task_changed = True
        logger.info("Task changed: '%s' -> '%s'", previous, task)
        return True

    def _take_task(self) -> tuple[str, bool]:
        """Read the task and whether it changed since the last read.

        Call from the thread that runs inference: the "changed" edge is
        consumed here so the backend can drop actions precomputed under
        the previous instruction before using the new one.
        """
        with self._task_lock:
            changed, self._task_changed = self._task_changed, False
            return self._task, changed

    @property
    def dispatched_task(self) -> str:
        """Instruction that generated the most recently returned action.

        Unlike :attr:`task` — the *requested* instruction, which
        ``set_task`` changes immediately — this follows the actions the
        engine actually hands out: every successful ``get_action``
        records the task that generated the returned action.  Recording
        strategies label frames with it so actions still queued (or
        being interpolated) from a previous instruction keep that
        instruction's label.

        Only meaningful on the control thread right after it consumed
        ``get_action``; between a reset and the next dispatched action
        it holds the requested task.
        """
        with self._task_lock:
            return self._dispatched_task

    def _set_dispatched_task(self, task: str) -> None:
        """Record the task of the action a ``get_action`` call is returning."""
        with self._task_lock:
            self._dispatched_task = task

    def _discard_task_change(self) -> None:
        """Drop a pending task-change edge, e.g. from ``reset`` (state is already cleared).

        Also re-primes ``dispatched_task``: the caller just cleared any
        queued actions, so the next dispatched action can only come from
        the current instruction.
        """
        with self._task_lock:
            self._task_changed = False
            self._dispatched_task = self._task

    # ------------------------------------------------------------------
    # Text queries (VQA)
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when this backend's policy can serve text queries.

        Backends holding a policy report ``policy.supports_text_generation()``;
        the default is False so a backend without a text path never accepts a
        query it cannot serve.  Callers (the controller's ``ask`` /
        ``autosteer``) check this before queueing, so the operator is refused
        immediately instead of receiving an error answer a tick later.
        """
        return False

    def set_answer_observer(self, observer: Callable[[QueryAnswer], None] | None) -> None:
        """Register the callback :meth:`pump_query` hands ready answers to."""
        with self._query_lock:
            self._answer_observer = observer

    @property
    def has_pending_query(self) -> bool:
        """True while a query is queued and not yet served."""
        with self._query_lock:
            return self._pending_query is not None

    @property
    def autosteer_goal(self) -> str | None:
        """The high-level goal currently driving the task, if any."""
        with self._query_lock:
            return self._autosteer_goal

    def ask(self, question: str) -> bool:
        """Queue a free-form ``question`` for the policy's text head.

        Callable from any thread.  Returns ``False`` when a query is
        already pending — the channel holds one at a time, so a second
        query cannot silently displace an unserved one.
        """
        return self._queue_query(PolicyQuery(kind=QueryKind.VQA, text=question))

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        """Drive the task from ``goal``, re-planning every ``interval_s`` seconds.

        Callable from any thread.  The engine asks the policy for the next
        subtask and applies it through :meth:`set_task`, so the normal
        instruction-switch path (queue blending on RTC, action drop on
        sync) handles the transition.  Progress through the plan lives in
        the policy, not here: every query re-sends the same goal.

        The interval is measured from the moment a subtask is *applied*,
        not from when its query is queued, so a generation slower than the
        interval cannot queue the next one back-to-back and starve the
        robot of motion.
        """
        with self._query_lock:
            self._autosteer_goal = goal
            self._autosteer_interval_s = max(0.0, interval_s)
            # Due immediately: the first subtask is requested on the very next
            # control tick rather than a full interval from now.
            self._autosteer_due_at = time.perf_counter()
        logger.info("Autosteer started for goal '%s' (every %.1fs)", goal, interval_s)

    def stop_autosteer(self) -> str | None:
        """Stop the sequencer, returning the goal it was driving (or ``None``)."""
        with self._query_lock:
            goal, self._autosteer_goal = self._autosteer_goal, None
        if goal is not None:
            logger.info("Autosteer stopped (goal was '%s')", goal)
        return goal

    def drop_pending_query(self) -> PolicyQuery | None:
        """Discard an unserved query, returning it (or ``None``).

        The controller calls this when a run segment ends: the query would
        otherwise sit in the slot and be served against a completely
        different scene the next time the robot starts.
        """
        with self._query_lock:
            dropped, self._pending_query = self._pending_query, None
        return dropped

    @abc.abstractmethod
    def pump_query(self, obs_processed: dict | None = None) -> None:
        """Service the query channel.  Call from the control thread only.

        Implementations must run, in order: :meth:`_poll_autosteer` (so
        the sequencer can queue its next-subtask query), then
        :meth:`_service_query` — but *only* when this backend's policy is
        owned by the control thread; async backends answer on their
        inference thread instead and skip it here — and finally
        :meth:`_deliver_answer`, so observers always fire on this thread.

        ``obs_processed`` is the processed observation of the current
        control tick; pass ``None`` when none is available (the
        controller's idle poll does).  A pending query is then left queued
        rather than served without a view, and the autosteer sequencer does
        not advance.
        """

    def _queue_query(self, query: PolicyQuery) -> bool:
        with self._query_lock:
            if self._pending_query is not None:
                return False
            self._pending_query = query
        return True

    def _poll_autosteer(self, obs_processed: dict | None) -> None:
        """Queue the next-subtask query if the sequencer is due.

        Driven from the control loop rather than a timer thread so it only
        advances while the robot is actually running: ``obs_processed`` is
        ``None`` on the controller's idle poll, which is not a control loop.
        """
        if obs_processed is None:
            return
        with self._query_lock:
            if self._autosteer_goal is None:
                return
            if time.perf_counter() < self._autosteer_due_at:
                return
            if self._pending_query is not None:
                # The operator's /vqa (or our own previous query) is still in
                # flight.  The deadline stays in the past, so the next tick
                # retries rather than losing this turn entirely.
                return
            self._pending_query = PolicyQuery(kind=QueryKind.NEXT_SUBTASK, text=self._autosteer_goal)

    def _schedule_next_autosteer(self) -> None:
        """Arm the next query, measured from now.  Called once a subtask lands."""
        with self._query_lock:
            if self._autosteer_goal is None:
                return  # stopped while this subtask was being generated
            self._autosteer_due_at = time.perf_counter() + self._autosteer_interval_s

    def _take_query(self) -> PolicyQuery | None:
        """Claim the pending query.  Call from the policy-owning thread."""
        with self._query_lock:
            query, self._pending_query = self._pending_query, None
            return query

    def _service_query(self, obs_processed: dict | None) -> None:
        """Serve a pending query.  Call ONLY from the policy-owning thread.

        Failures are captured into an answer rather than raised: a query
        the policy cannot handle must not take down the control loop or
        the background inference thread.
        """
        if obs_processed is None:
            return
        query = self._take_query()
        if query is None:
            return
        try:
            text = self._generate_text(obs_processed, query)
        except Exception as e:
            logger.exception("Policy text query failed (%s) for %r", query.kind.value, query.text)
            if query.kind is QueryKind.NEXT_SUBTASK:
                # A sequencer that cannot get its next subtask is broken; stop
                # it rather than failing again every interval.
                self.stop_autosteer()
            self._publish_answer(
                QueryAnswer(question=query.text, error=f"{type(e).__name__}: {e}", kind=query.kind)
            )
            return
        if query.kind is QueryKind.NEXT_SUBTASK:
            # Applied here, on the policy-owning thread, so the subtask is live
            # for the very next inference.
            self.set_task(text)
            # Armed only now, so the interval measures robot motion between
            # subtasks rather than wall-clock that a slow generate ate into.
            self._schedule_next_autosteer()
        # Published after being applied, so an observer announcing the subtask
        # never gets ahead of the task it describes.
        self._publish_answer(QueryAnswer(question=query.text, answer=text, kind=query.kind))

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on ``obs_processed``.  Backend-specific.

        Implementations build the batch, stamp the query into it with
        :meth:`_mark_query`, run the preprocessor, and call
        ``policy.generate_text(batch)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text queries — no /vqa or /autosteer on this backend."
        )

    @staticmethod
    def _mark_query(batch: dict, query: PolicyQuery) -> dict:
        """Stamp ``batch`` with the query's kind and text, for the preprocessor.

        Call between ``prepare_observation_for_inference`` and the
        preprocessor pipeline.  ``QUERY_KIND`` and ``QUERY_TEXT`` are in the
        converters' complementary-data allowlist, so they survive
        ``batch_to_transition`` and land beside ``task`` — where a
        ``ComplementaryDataProcessorStep`` can read the kind and rewrite
        ``QUERY_TEXT`` in place into this policy's prompt format before
        ``generate_text`` consumes it.  Stored as plain strings so processor
        steps need not import this module, and unbatched because they
        describe the request, not a sample.
        """
        batch[QUERY_KIND] = query.kind.value
        batch[QUERY_TEXT] = query.text
        return batch

    def _publish_answer(self, answer: QueryAnswer) -> None:
        with self._query_lock:
            self._ready_answer = answer

    def _deliver_answer(self) -> None:
        with self._query_lock:
            answer, self._ready_answer = self._ready_answer, None
            observer = self._answer_observer
        if answer is None or observer is None:
            return
        try:
            observer(answer)
        except Exception:  # a broken observer must not kill the control loop
            logger.exception("Error in inference-engine answer observer")

    @abc.abstractmethod
    def start(self) -> None:
        """Initialise the backend."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear the backend down."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped state."""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action tensor, or ``None`` if unavailable."""

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """Publish the latest processed observation.  Default: no-op."""

    def pause(self) -> None:  # noqa: B027
        """Pause background inference.  Default: no-op."""

    def resume(self) -> None:  # noqa: B027
        """Resume background inference.  Default: no-op."""

    @property
    def ready(self) -> bool:
        """True once the backend can produce actions (e.g. warmup done)."""
        return True

    @property
    def failed(self) -> bool:
        """True if an unrecoverable error occurred in the backend."""
        return False

    @property
    def failure_traceback(self) -> str | None:
        """Formatted traceback of the unrecoverable error, when ``failed`` is True."""
        return None
