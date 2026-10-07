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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock

import torch

from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

logger = logging.getLogger(__name__)


class QueryKind(Enum):
    """What the policy's text head is being asked for."""

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

    Exactly one of ``answer`` / ``error`` is set.  A ``NEXT_SUBTASK`` success carries a
    subtask the engine has *already applied*, so the receiver only announces it.
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
    ``set_task`` is callable from any thread; subclasses pick the value up on their own
    inference thread via :meth:`_take_task`, so policy state is never mutated across threads.

    Text queries
    ------------
    ``ask`` is callable from any thread and never touches the policy: queries are served
    by :meth:`_service_query` on the thread owning the policy (see
    :attr:`control_thread_owns_policy`), and answers reach observers only from
    :meth:`pump_query` on the control thread.

    Optional hooks
    --------------
    ``notify_observation`` / ``pause`` / ``resume`` have a no-op default
    so rollout strategies can invoke them unconditionally.

    Subclasses must call ``super().__init__(task=...)``.
    """

    def __init__(self, task: str = "") -> None:
        self._task = task
        self._task_changed = False
        self._dispatched_task = task
        self._task_lock = Lock()

        # Text-query channel.  Its own lock, never held across a text generation.
        self._query_lock = Lock()
        self._pending_query: PolicyQuery | None = None
        # Set from the claim (``_take_query``) until the answer is published or the turn
        # is discarded, so the autosteer poll cannot queue a duplicate turn meanwhile.
        self._query_in_flight = False
        # Answers awaiting delivery; a queue so an undelivered one is never overwritten.
        self._ready_answers: deque[QueryAnswer] = deque()
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

        Callable from any thread.  Returns ``True`` when the value actually changed.
        """
        with self._task_lock:
            if task == self._task:
                return False
            previous, self._task = self._task, task
            self._task_changed = True
        logger.info("Task changed: '%s' -> '%s'", previous, task)
        return True

    def _take_task(self) -> tuple[str, bool]:
        """Read the task and consume the "changed" edge.  Call from the inference thread."""
        with self._task_lock:
            changed, self._task_changed = self._task_changed, False
            return self._task, changed

    @property
    def dispatched_task(self) -> str:
        """Instruction that generated the most recently returned action.

        Trails :attr:`task` (the *requested* instruction) while actions from a previous
        instruction are still being consumed; recording strategies label frames with it.
        Only meaningful on the control thread right after ``get_action``; after a reset it
        holds the requested task.
        """
        with self._task_lock:
            return self._dispatched_task

    def _set_dispatched_task(self, task: str) -> None:
        """Record the task of the action a ``get_action`` call is returning."""
        with self._task_lock:
            self._dispatched_task = task

    def _discard_task_change(self) -> None:
        """Drop a pending task-change edge, e.g. from ``reset`` (state is already cleared).

        Also re-primes ``dispatched_task``: with queued actions gone, the next dispatched
        action can only come from the current instruction.
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

        Default False, so a backend without a text path never accepts a query it cannot
        serve; callers check this before queueing.
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

        Callable from any thread.  Returns ``False`` when one is already pending: the
        channel holds a single query at a time.
        """
        return self._queue_query(PolicyQuery(kind=QueryKind.VQA, text=question))

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        """Drive the task from ``goal``, re-planning every ``interval_s`` seconds.

        Callable from any thread.  Each turn asks the policy for the next subtask and
        applies it through :meth:`set_task`; every query re-sends the same goal, so plan
        progress lives in the policy.  The interval is measured from when a subtask is
        *applied*, so a slow generation cannot starve the robot of motion.
        """
        with self._query_lock:
            self._autosteer_goal = goal
            self._autosteer_interval_s = max(0.0, interval_s)
            # Due immediately: first subtask requested on the next control tick.
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

        Called when a run segment ends, so the query is not served against a completely
        different scene the next time the robot starts.
        """
        with self._query_lock:
            dropped, self._pending_query = self._pending_query, None
        return dropped

    @property
    @abc.abstractmethod
    def control_thread_owns_policy(self) -> bool:
        """Whether the control thread is the one allowed to touch the policy.

        True (inline backends): :meth:`pump_query` serves pending queries itself.  False
        (async backends): their inference thread must call :meth:`_service_query`, and
        :meth:`pump_query` only advances the sequencer and delivers finished answers.
        """

    def pump_query(self, obs_processed: dict | None = None) -> bool:
        """Advance the text-query channel by one tick.  Control thread only.

        Polls the autosteer sequencer, serves a pending query when
        :attr:`control_thread_owns_policy` (async backends answer on their own thread),
        then delivers ready answers, so observers always fire on this thread.  Called at
        the end of a tick rather than from :meth:`get_action`: a text generation far
        outlasts a control tick.  Returns ``True`` when a query was served inline.  With
        ``obs_processed=None`` (the controller's idle poll) a pending query stays queued
        and the sequencer does not advance.
        """
        self._poll_autosteer(obs_processed)
        served = False
        if self.control_thread_owns_policy:
            served = self._service_query(obs_processed)
        self._deliver_answer()
        return served

    def _queue_query(self, query: PolicyQuery) -> bool:
        with self._query_lock:
            if self._pending_query is not None:
                return False
            self._pending_query = query
        return True

    def _poll_autosteer(self, obs_processed: dict | None) -> None:
        """Queue the next-subtask query if the sequencer is due (control loop only)."""
        if obs_processed is None:
            return
        with self._query_lock:
            if self._autosteer_goal is None:
                return
            if time.perf_counter() < self._autosteer_due_at:
                return
            if self._pending_query is not None or self._query_in_flight:
                # A /vqa (or our own previous query) is still queued or being generated.
                # The deadline stays in the past, so the next tick retries this turn.
                return
            self._pending_query = PolicyQuery(kind=QueryKind.NEXT_SUBTASK, text=self._autosteer_goal)

    def _take_query(self) -> PolicyQuery | None:
        """Claim the pending query.  Call from the policy-owning thread."""
        with self._query_lock:
            query, self._pending_query = self._pending_query, None
            if query is not None:
                self._query_in_flight = True
            return query

    def _service_query(self, obs_processed: dict | None) -> bool:
        """Serve a pending query.  Call ONLY from the policy-owning thread.

        Failures become error answers instead of exceptions, so a bad query never takes
        down the calling thread.  Returns ``True`` when a query was claimed and served.
        """
        if obs_processed is None:
            return False
        query = self._take_query()
        if query is None:
            return False
        try:
            text = self._generate_text(obs_processed, query)
            if not isinstance(text, str) or not text.strip():
                # Fail here so garbage becomes an error answer instead of steering the
                # robot and labeling recorded frames.
                raise TypeError(
                    f"generate_text() must return a non-empty str, got {text!r} ({type(text).__name__})"
                )
        except Exception as e:
            logger.exception("Policy text query failed (%s) for %r", query.kind.value, query.text)
            if query.kind is QueryKind.NEXT_SUBTASK and not self._fail_subtask(query):
                return True  # the sequencer this turn belonged to is gone; discard
            self._publish_answer(
                QueryAnswer(question=query.text, error=f"{type(e).__name__}: {e}", kind=query.kind)
            )
            return True
        if query.kind is QueryKind.NEXT_SUBTASK and not self._apply_subtask(query, text):
            return True  # sequencer stopped meanwhile; the turn was discarded
        # Published after being applied, so an announcing observer never gets ahead of
        # the task it describes.
        self._publish_answer(QueryAnswer(question=query.text, answer=text, kind=query.kind))
        return True

    def _fail_subtask(self, query: PolicyQuery) -> bool:
        """Stop the sequencer after a failed turn — unless it stopped or retargeted meanwhile.

        A sequencer that cannot get its next subtask must stop rather than fail every
        interval — but only if it is still the one that requested this turn.  Returns
        ``True`` when the failure answer should be published.
        """
        with self._query_lock:
            live = self._autosteer_goal == query.text
            if live:
                self._autosteer_goal = None
            else:
                self._query_in_flight = False  # no answer will be published
        if live:
            logger.info("Autosteer stopped (goal was '%s') — planning failed", query.text)
        else:
            logger.info(
                "Discarding failed autosteer turn for %r — the sequencer stopped or was "
                "retargeted while it was being generated",
                query.text,
            )
        return live

    def _apply_subtask(self, query: PolicyQuery, subtask: str) -> bool:
        """Apply a generated subtask, unless the sequencer stopped meanwhile.

        The generation ran lock-free for seconds, so check and apply happen atomically
        under ``_query_lock`` (``_task_lock`` nests inside it, never the reverse) or a
        stale plan could overwrite a newer instruction.  Returns ``True`` when applied.
        """
        with self._query_lock:
            live = self._autosteer_goal == query.text
            if live:
                self.set_task(subtask)
                # Armed only now, so the interval measures motion between subtasks.
                self._autosteer_due_at = time.perf_counter() + self._autosteer_interval_s
            else:
                self._query_in_flight = False  # no answer will be published
        if not live:
            logger.info(
                "Discarding autosteer subtask %r — the sequencer stopped while it was being generated",
                subtask,
            )
        return live

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on ``obs_processed``.  Backend-specific.

        Implementations build the batch, stamp it with :meth:`_mark_query`, preprocess it,
        and call ``policy.generate_text``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text queries — no /vqa or /autosteer on this backend."
        )

    @staticmethod
    def _mark_query(batch: dict, query: PolicyQuery) -> dict:
        """Stamp ``batch`` with the query's kind and text, for the preprocessor.

        Call between ``prepare_observation_for_inference`` and the preprocessor pipeline.
        ``QUERY_KIND`` / ``QUERY_TEXT`` are allowlisted complementary data, so they land
        beside ``task``: a policy-specific ``ComplementaryDataProcessorStep`` can read the
        kind there and rewrite ``QUERY_TEXT`` into its prompt format.
        """
        batch[QUERY_KIND] = query.kind.value
        batch[QUERY_TEXT] = query.text
        return batch

    def drop_ready_subtask_answers(self) -> None:
        """Discard undelivered ``NEXT_SUBTASK`` answers.

        Called at segment end, right after stopping the sequencer, so no later
        announcement describes a sequencer that no longer drives anything.  VQA answers
        stay deliverable.
        """
        with self._query_lock:
            kept = [a for a in self._ready_answers if a.kind is not QueryKind.NEXT_SUBTASK]
            dropped = len(self._ready_answers) - len(kept)
            self._ready_answers = deque(kept)
        if dropped:
            logger.debug("Dropped %d undelivered autosteer answer(s) at segment end", dropped)

    def _publish_answer(self, answer: QueryAnswer) -> None:
        with self._query_lock:
            self._query_in_flight = False
            self._ready_answers.append(answer)

    def _deliver_answer(self) -> None:
        with self._query_lock:
            answers = list(self._ready_answers)
            self._ready_answers.clear()
            observer = self._answer_observer
        if observer is None:
            return
        for answer in answers:
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
